import os
import time
import json
import gc

import pandas as pd
import torch
import sacrebleu
import mlflow
import matplotlib.pyplot as plt
import transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mlflow.pyfunc import PythonModel


# ==================================================
# STEP 1: PROJECT SETTINGS
# ==================================================

MODEL_A_ID = "Helsinki-NLP/opus-mt-hi-en"
MODEL_B_ID = "facebook/nllb-200-distilled-600M"

DATASET_PATH = "data/eval_inputs.csv"
OUTPUT_DIR = "outputs"

EXPERIMENT_NAME = "hf-hindi-english-champion-challenger"
REGISTERED_MODEL_NAME = "hindi-english-translation"

SEED = 42
MAX_LENGTH = 256
BLEU_MARGIN = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STUDENT_USN = "1MS25CI109"

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

print("Device:", DEVICE)
print("MLflow tracking URI:", mlflow.get_tracking_uri())


# ==================================================
# STEP 2: LOAD AND VALIDATE EVALUATION DATASET
# ==================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Could not find {DATASET_PATH}. Run this script from the project folder."
    )

df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

required_columns = {"id", "hindi", "reference_english"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

if len(df) != 100:
    raise ValueError(f"Expected exactly 100 evaluation examples, found {len(df)}.")

if df[list(required_columns)].isna().any().any():
    raise ValueError("The evaluation dataset contains missing values.")

if df["id"].duplicated().any():
    raise ValueError("The evaluation dataset contains duplicate IDs.")

hindi_sentences = df["hindi"].astype(str).tolist()
references = df["reference_english"].astype(str).tolist()

print("Evaluation examples:", len(df))


# ==================================================
# STEP 3: LOAD TRANSLATION MODELS
# ==================================================

def load_model(model_id, model_type):
    """Load the selected Hugging Face translation model."""

    print(f"\nLoading model: {model_id}")

    if model_type == "opus":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_type == "nllb":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            src_lang="hin_Deva"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ==================================================
# STEP 4: TRANSLATION FUNCTION
# ==================================================

def translate_sentences(tokenizer, model, sentences, model_type):
    """Translate sentences and measure per-sentence inference latency."""

    predictions = []
    individual_times = []

    target_id = None

    if model_type == "nllb":
        target_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    for sentence in sentences:
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            if model_type == "nllb":
                output = model.generate(
                    **inputs,
                    forced_bos_token_id=target_id,
                    max_length=MAX_LENGTH
                )
            else:
                output = model.generate(
                    **inputs,
                    max_length=MAX_LENGTH
                )

        translation = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        predictions.append(translation)
        individual_times.append(elapsed)

    return predictions, individual_times


# ==================================================
# STEP 5: CUSTOM MLFLOW PYFUNC MODEL
# ==================================================

class TranslationPyFunc(PythonModel):
    """A reloadable MLflow model that translates Hindi input."""

    def load_context(self, context):
        with open(context.artifacts["config"], "r", encoding="utf-8") as file:
            config = json.load(file)

        self.model_id = config["model_id"]
        self.model_type = config["model_type"]

        self.tokenizer, self.model = load_model(
            self.model_id,
            self.model_type
        )

    def predict(self, context, model_input, params=None):
        if "hindi" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'hindi' column.")

        sentences = model_input["hindi"].astype(str).tolist()

        predictions, _ = translate_sentences(
            self.tokenizer,
            self.model,
            sentences,
            self.model_type
        )

        return pd.DataFrame({
            "hindi": sentences,
            "prediction": predictions
        })


# ==================================================
# STEP 6: EVALUATE ONE MODEL
# ==================================================

def evaluate_model(model_id, model_type, run_name):
    tokenizer, model = load_model(model_id, model_type)

    predictions, individual_times = translate_sentences(
        tokenizer,
        model,
        hindi_sentences,
        model_type
    )

    total_time = sum(individual_times)
    average_latency = (total_time / len(predictions)) * 1000

    bleu_score = sacrebleu.corpus_bleu(
        predictions,
        [references]
    ).score

    chrf_score = sacrebleu.corpus_chrf(
        predictions,
        [references]
    ).score

    metrics = {
        "BLEU": float(bleu_score),
        "chrF": float(chrf_score),
        "avg_latency_ms": float(average_latency),
        "total_inference_s": float(total_time)
    }

    results = pd.DataFrame({
        "id": df["id"],
        "hindi": hindi_sentences,
        "reference_english": references,
        "prediction": predictions
    })

    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"{run_name}_predictions.csv"
    )
    results.to_csv(
        prediction_file,
        index=False,
        encoding="utf-8-sig"
    )

    metrics_file = os.path.join(
        OUTPUT_DIR,
        f"{run_name}_metrics.json"
    )
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    config_file = os.path.join(
        OUTPUT_DIR,
        f"{run_name}_model_config.json"
    )
    with open(config_file, "w", encoding="utf-8") as file:
        json.dump({
            "model_id": model_id,
            "model_type": model_type
        }, file, indent=4)

    # Log this model's evaluation as one MLflow run.
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("hf_model_id", model_id)
        mlflow.log_param("task", "hindi_to_english_translation")
        mlflow.log_param("role", "candidate")
        mlflow.log_param("eval_samples", len(df))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("batch_size", 1)

        mlflow.log_metric("BLEU", metrics["BLEU"])
        mlflow.log_metric("chrF", metrics["chrF"])
        mlflow.log_metric("avg_latency_ms", metrics["avg_latency_ms"])
        mlflow.log_metric("total_inference_s", metrics["total_inference_s"])

        mlflow.set_tag("student_usn", STUDENT_USN)
        mlflow.set_tag("dataset_name", "Hindi-English-100")
        mlflow.set_tag("transformers_version", transformers.__version__)
        mlflow.set_tag("torch_version", torch.__version__)
        mlflow.set_tag("mlflow_version", mlflow.__version__)

        mlflow.log_artifact(prediction_file)
        mlflow.log_artifact(metrics_file)
        mlflow.log_artifact(DATASET_PATH)

        # The model reloads its Hugging Face weights from the model ID.
        # Leave torch unpinned here so pip can select a compatible build.
        mlflow.pyfunc.log_model(
            name="translation_model",
            python_model=TranslationPyFunc(),
            artifacts={"config": config_file},
            pip_requirements=[
                "mlflow==3.18.0",
                "torch",
                "transformers==4.57.3",
                "sentencepiece",
                "pandas"
            ]
        )

    print(f"\nModel: {model_id}")
    print("BLEU:", round(metrics["BLEU"], 4))
    print("chrF:", round(metrics["chrF"], 4))
    print("Average latency (ms):", round(metrics["avg_latency_ms"], 2))
    print("Total inference time (s):", round(metrics["total_inference_s"], 2))

    del model, tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, predictions


# ==================================================
# STEP 7: RUN MODEL A AND MODEL B
# ==================================================

print("\n========== MODEL A: OPUS-MT ==========")

metrics_a, predictions_a = evaluate_model(
    MODEL_A_ID,
    "opus",
    "A_translation_opus_mt"
)

print("\n========== MODEL B: NLLB-200 ==========")

metrics_b, predictions_b = evaluate_model(
    MODEL_B_ID,
    "nllb",
    "B_translation_nllb_200"
)


# ==================================================
# STEP 8: CHAMPION-CHALLENGER DECISION
# ==================================================

bleu_difference = metrics_b["BLEU"] - metrics_a["BLEU"]
chrf_difference = metrics_b["chrF"] - metrics_a["chrF"]

# Predefined rule:
# 1. NLLB wins if its BLEU is at least 1.0 point higher.
# 2. If the absolute BLEU difference is less than 1.0,
#    the higher chrF score decides.
# 3. If chrF is tied, OPUS-MT remains Champion.
# 4. If OPUS-MT leads in BLEU by at least 1.0, it wins.

if bleu_difference >= BLEU_MARGIN:
    winner = "NLLB-200"
elif abs(bleu_difference) < BLEU_MARGIN:
    if chrf_difference > 0:
        winner = "NLLB-200"
    else:
        winner = "OPUS-MT"
else:
    winner = "OPUS-MT"

print("\n========== CHAMPION DECISION ==========")
print("BLEU difference (NLLB - OPUS):", round(bleu_difference, 4))
print("chrF difference (NLLB - OPUS):", round(chrf_difference, 4))
print("Selected Champion:", winner)


# ==================================================
# STEP 9: COMPARISON TABLE AND JSON SUMMARY
# ==================================================

comparison = pd.DataFrame([
    {"Model": "OPUS-MT", **metrics_a},
    {"Model": "NLLB-200", **metrics_b}
])

comparison_file = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
comparison.to_csv(comparison_file, index=False, encoding="utf-8-sig")

metrics_summary_json = os.path.join(
    OUTPUT_DIR,
    "metrics_summary.json"
)

with open(metrics_summary_json, "w", encoding="utf-8") as file:
    json.dump({
        "winner": winner,
        "primary_metric": "BLEU",
        "BLEU_margin": BLEU_MARGIN,
        "BLEU_difference_NLLB_minus_OPUS": float(bleu_difference),
        "chrF_difference_NLLB_minus_OPUS": float(chrf_difference),
        "OPUS_MT": metrics_a,
        "NLLB_200": metrics_b
    }, file, indent=4)

print("\n========== FINAL COMPARISON ==========")
print(comparison.to_string(index=False))


# ==================================================
# STEP 10: CREATE EVALUATION PLOTS
# ==================================================

# Plot 1: BLEU and chrF comparison
ax = comparison.set_index("Model")[["BLEU", "chrF"]].plot(
    kind="bar",
    figsize=(8, 5)
)
ax.set_title("OPUS-MT vs NLLB-200: Translation Metrics")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
plt.xticks(rotation=0)
plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_DIR,
    "translation_metrics_comparison.png"
)
plt.savefig(plot1)
plt.close()

# Plot 2: Average inference time comparison
plt.figure(figsize=(8, 5))
plt.bar(
    comparison["Model"],
    comparison["avg_latency_ms"]
)
plt.title("Average Translation Inference Time")
plt.ylabel("Milliseconds per sentence")
plt.xlabel("Model")
plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_DIR,
    "inference_time_comparison.png"
)
plt.savefig(plot2)
plt.close()


# ==================================================
# STEP 11: QUALITATIVE DISAGREEMENT EXAMPLES
# ==================================================

disagreements = pd.DataFrame({
    "hindi": hindi_sentences,
    "reference": references,
    "opus_prediction": predictions_a,
    "nllb_prediction": predictions_b
})

disagreements = disagreements[
    disagreements["opus_prediction"].str.strip().str.lower()
    != disagreements["nllb_prediction"].str.strip().str.lower()
].head(5).copy()

disagreements["comment"] = (
    "The models produced different translations. "
    "Compare both outputs with the reference for meaning and fluency."
)

disagreement_file = os.path.join(
    OUTPUT_DIR,
    "qualitative_examples.csv"
)

disagreements.to_csv(
    disagreement_file,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# STEP 12: SUMMARY RUN IN MLFLOW
# ==================================================

with mlflow.start_run(run_name="compare_translation"):
    mlflow.log_param("primary_metric", "BLEU")
    mlflow.log_param("margin", BLEU_MARGIN)
    mlflow.log_param("tie_breaker", "chrF")
    mlflow.log_param("eval_samples", len(df))
    mlflow.log_param("seed", SEED)

    mlflow.log_metric("delta_primary_metric", float(bleu_difference))
    mlflow.log_metric("delta_chrF", float(chrf_difference))

    mlflow.set_tag("winner", winner)
    mlflow.set_tag("selection_rule", "BLEU with chrF tie-breaker")

    mlflow.log_artifact(comparison_file)
    mlflow.log_artifact(metrics_summary_json)
    mlflow.log_artifact(plot1)
    mlflow.log_artifact(plot2)
    mlflow.log_artifact(disagreement_file)
    mlflow.log_artifact(DATASET_PATH)


# ==================================================
# STEP 13: FINISH
# ==================================================

print("\nEvaluation completed successfully!")
print("Champion:", winner)
print("Files saved in:", OUTPUT_DIR)

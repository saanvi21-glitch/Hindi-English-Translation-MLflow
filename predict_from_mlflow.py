import os
import pandas as pd
import mlflow
from mlflow import MlflowClient


# ==================================================
# STEP 1: PROJECT SETTINGS
# ==================================================

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "hindi-english-translation"
MODEL_ALIAS = "champion"

INPUT_FILE = "data/new_data.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "new_data_predictions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)


# ==================================================
# STEP 2: CHECK AND LOAD NEW INPUT DATA
# ==================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Create data/new_data.csv before running this script."
    )

new_data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = {"id", "hindi"}
missing_columns = required_columns - set(new_data.columns)

if missing_columns:
    raise ValueError(
        f"new_data.csv is missing these columns: {sorted(missing_columns)}"
    )

if new_data.empty:
    raise ValueError("new_data.csv has no prediction examples.")

if new_data[["id", "hindi"]].isna().any().any():
    raise ValueError("new_data.csv contains missing IDs or Hindi sentences.")

if len(new_data) < 5 or len(new_data) > 10:
    raise ValueError(
        f"Expected 5 to 10 new examples, but found {len(new_data)}."
    )

print("New examples loaded:", len(new_data))


# ==================================================
# STEP 3: FIND CHAMPION MODEL VERSION
# ==================================================

try:
    champion_version = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS
    )
except Exception as error:
    raise RuntimeError(
        "Could not find the 'champion' alias in the MLflow Model Registry. "
        "Check that register_champion.py completed successfully."
    ) from error

model_version = champion_version.version
source_run_id = champion_version.run_id

if not source_run_id:
    raise RuntimeError("The Champion model version has no source MLflow run ID.")

source_run = client.get_run(source_run_id)
hf_model_id = source_run.data.params.get("hf_model_id", "Not recorded")

print("\n========== CHAMPION MODEL DETAILS ==========")
print("Registered model:", REGISTERED_MODEL_NAME)
print("Alias:", MODEL_ALIAS)
print("Model version:", model_version)
print("Hugging Face model ID:", hf_model_id)


# ==================================================
# STEP 4: LOAD CHAMPION FROM MLFLOW
# ==================================================

model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

print("\nLoading Champion from MLflow:")
print(model_uri)

# Load the registered MLflow model through its alias.
# Do not load or call a Hugging Face pipeline directly here.
champion_model = mlflow.pyfunc.load_model(model_uri)

print("Champion model loaded successfully!")


# ==================================================
# STEP 5: GENERATE PREDICTIONS ON NEW DATA
# ==================================================

model_input = new_data[["hindi"]].copy()

predictions = champion_model.predict(model_input)

if isinstance(predictions, pd.DataFrame):
    prediction_values = predictions["prediction"].astype(str).tolist()
else:
    prediction_values = pd.Series(predictions).astype(str).tolist()

if len(prediction_values) != len(new_data):
    raise ValueError(
        "The number of predictions does not match the number of input sentences."
    )

results = new_data.copy()
results["prediction_english"] = prediction_values
results["model_name"] = hf_model_id
results["model_version"] = str(model_version)
results["model_alias"] = MODEL_ALIAS


# ==================================================
# STEP 6: DISPLAY AND SAVE RESULTS
# ==================================================

print("\n========== NEW DATA PREDICTIONS ==========")
print(results.to_string(index=False))

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nPredictions saved successfully!")
print("Output file:", OUTPUT_FILE)
print("Model used:", hf_model_id)
print("MLflow model version:", model_version)

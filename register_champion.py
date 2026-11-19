import mlflow
from mlflow import MlflowClient

# ==================================================
# SETTINGS
# ==================================================

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-hindi-english-champion-challenger"
REGISTERED_MODEL_NAME = "hindi-english-translation"

# Your evaluation selected NLLB-200 as Champion.
CHAMPION_MODEL_ID = "facebook/nllb-200-distilled-600M"
CHALLENGER_MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)


# ==================================================
# STEP 1: FIND THE EXPERIMENT
# ==================================================

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' was not found. "
        "Check that the MLflow server is running."
    )

print("Experiment found:", experiment.name)


# ==================================================
# STEP 2: FIND THE TWO COMPLETED EVALUATION RUNS
# ==================================================

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    max_results=100
)

def find_run_by_model_id(model_id):
    matching_runs = [
        run for run in runs
        if run.data.params.get("hf_model_id") == model_id
    ]

    if not matching_runs:
        raise ValueError(
            f"No completed MLflow run found for Hugging Face model: {model_id}"
        )

    # Select the most recently created matching run.
    return max(matching_runs, key=lambda run: run.info.start_time)


champion_run = find_run_by_model_id(CHAMPION_MODEL_ID)
challenger_run = find_run_by_model_id(CHALLENGER_MODEL_ID)

print("\nChampion run ID:", champion_run.info.run_id)
print("Challenger run ID:", challenger_run.info.run_id)


# ==================================================
# STEP 3: FIND THE LOGGED MODEL FOR EACH RUN
# ==================================================

def find_logged_model_uri(run_id, expected_model_id):
    """
    Find the MLflow 3 logged model belonging to a run.
    The evaluate.py script logged it with the name 'translation_model'.
    """

    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        max_results=100
    )

    if not logged_models:
        raise ValueError(
            f"No MLflow logged model was found for run {run_id}. "
            "Open that run in MLflow and check its Artifacts/Models section."
        )

    # Prefer the model whose logged-model name is translation_model.
    selected_model = None

    for logged_model in logged_models:
        model_name = getattr(logged_model, "name", None)
        if model_name == "translation_model":
            selected_model = logged_model
            break

    if selected_model is None:
        selected_model = logged_models[0]

    model_id = selected_model.model_id
    model_uri = f"models:/{model_id}"

    print(f"\nFound logged model for {expected_model_id}")
    print("Logged model ID:", model_id)
    print("Model URI:", model_uri)

    return model_uri


champion_model_uri = find_logged_model_uri(
    champion_run.info.run_id,
    CHAMPION_MODEL_ID
)

challenger_model_uri = find_logged_model_uri(
    challenger_run.info.run_id,
    CHALLENGER_MODEL_ID
)


# ==================================================
# STEP 4: CREATE REGISTERED MODEL IF NEEDED
# ==================================================

registered_models = [
    model.name for model in client.search_registered_models()
]

if REGISTERED_MODEL_NAME not in registered_models:
    client.create_registered_model(
        REGISTERED_MODEL_NAME,
        description=(
            "Hindi-to-English translation model. "
            "Champion: NLLB-200; Challenger: OPUS-MT."
        )
    )
    print("\nCreated registered model:", REGISTERED_MODEL_NAME)
else:
    print("\nRegistered model already exists:", REGISTERED_MODEL_NAME)


# ==================================================
# STEP 5: REGISTER BOTH MODEL VERSIONS
# ==================================================

def register_model_version(model_uri, model_label):
    # Avoid registering the same source URI more than once.
    existing_versions = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}'"
    )

    for version in existing_versions:
        if version.source == model_uri:
            print(
                f"{model_label} is already registered "
                f"as version {version.version}."
            )
            return version.version

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print(
        f"Registered {model_label} as version {registered.version}."
    )
    return registered.version


champion_version = register_model_version(
    champion_model_uri,
    "NLLB-200 Champion"
)

challenger_version = register_model_version(
    challenger_model_uri,
    "OPUS-MT Challenger"
)


# ==================================================
# STEP 6: ASSIGN MODEL ALIASES
# ==================================================

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="champion",
    version=champion_version
)

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="challenger",
    version=challenger_version
)

print("\n========== MODEL REGISTRATION COMPLETE ==========")
print("Registered model:", REGISTERED_MODEL_NAME)
print("Champion: NLLB-200, version", champion_version)
print("Challenger: OPUS-MT, version", challenger_version)
print("\nChampion URI:")
print(f"models:/{REGISTERED_MODEL_NAME}@champion")
print("\nChallenger URI:")
print(f"models:/{REGISTERED_MODEL_NAME}@challenger")

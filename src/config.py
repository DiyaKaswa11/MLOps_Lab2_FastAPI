"""Central configuration: file paths and dataset constants."""
from pathlib import Path

# Absolute paths, so the code works no matter which folder it is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

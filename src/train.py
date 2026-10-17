"""Trains a Decision Tree on Iris, evaluates it, and saves the model plus metadata."""
import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.config import CLASS_NAMES, FEATURE_NAMES, METADATA_PATH, MODEL_DIR, MODEL_PATH
from src.data import load_data, split_data

HYPERPARAMETERS = {"max_depth": 3, "random_state": 12}


def train_model(X_train, y_train):
    """Fits a Decision Tree classifier."""
    model = DecisionTreeClassifier(**HYPERPARAMETERS)
    model.fit(X_train, y_train)
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    now = datetime.now(timezone.utc)
    metadata = {
        "model_type": type(model).__name__,
        "model_version": now.strftime("%Y%m%d%H%M%S"),
        "trained_at": now.isoformat(timespec="seconds"),
        "scikit_learn_version": sklearn.__version__,
        "hyperparameters": HYPERPARAMETERS,
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": round(float(test_accuracy), 4),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=4))

    print(f"Model saved to {MODEL_PATH}")
    print(f"Test accuracy: {metadata['test_accuracy']}")
    print(f"Model version: {metadata['model_version']}")


if __name__ == "__main__":
    main()

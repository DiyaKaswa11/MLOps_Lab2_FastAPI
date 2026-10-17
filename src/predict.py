"""Model wrapper: loads the trained model once and makes predictions."""
import json

import joblib
import numpy as np

from src.config import CLASS_NAMES, METADATA_PATH, MODEL_PATH


class IrisModel:
    """Holds the trained model in memory so it is not reloaded on every request."""

    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Train it first with: python -m src.train"
            )
        self.model = joblib.load(model_path)
        self.metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )

    def predict(self, rows):
        """Predicts species for a list of [sepal_length, sepal_width, petal_length, petal_width] rows."""
        X = np.asarray(rows, dtype=float)
        class_ids = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        for class_id, probs in zip(class_ids, probabilities):
            results.append({
                "class_id": int(class_id),
                "species": CLASS_NAMES[int(class_id)],
                "confidence": round(float(probs.max()), 4),
                "probabilities": {
                    name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)
                },
            })
        return results

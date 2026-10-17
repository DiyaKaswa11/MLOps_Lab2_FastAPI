"""Iris species prediction API built with FastAPI."""
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from src.predict import IrisModel

MAX_BATCH_SIZE = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model once when the server starts, instead of on every request."""
    try:
        app.state.model = IrisModel()
        app.state.load_error = None
    except FileNotFoundError as error:
        app.state.model = None
        app.state.load_error = str(error)
    yield


app = FastAPI(
    title="Iris Species Prediction API",
    description="Predicts the species of an iris flower from its measurements (in cm).",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------- Request and response schemas ----------

class IrisFeatures(BaseModel):
    """Flower measurements in centimeters. Values must be greater than 0 and at most 10."""
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length (cm)", examples=[5.1])
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width (cm)", examples=[3.5])
    petal_length: float = Field(..., gt=0, le=10, description="Petal length (cm)", examples=[1.4])
    petal_width: float = Field(..., gt=0, le=10, description="Petal width (cm)", examples=[0.2])

    def to_row(self):
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class Prediction(BaseModel):
    class_id: int
    species: str
    confidence: float
    probabilities: Dict[str, float]


class BatchRequest(BaseModel):
    samples: List[IrisFeatures] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    count: int
    predictions: List[Prediction]


def get_model(request: Request) -> IrisModel:
    """Returns the loaded model, or a 503 error if it is not available."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.load_error,
        )
    return model


# ---------- Endpoints ----------

@app.get("/")
def root():
    """Welcome message with links to the main endpoints."""
    return {
        "message": "Iris Species Prediction API",
        "docs": "/docs",
        "endpoints": ["/health", "/model/info", "/predict", "/predict/batch"],
    }


@app.get("/health")
def health(request: Request, response: Response):
    """Reports whether the model is loaded. Returns 503 if it is not."""
    model_loaded = request.app.state.model is not None
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "healthy" if model_loaded else "unhealthy", "model_loaded": model_loaded}


@app.get("/model/info")
def model_info(request: Request):
    """Returns details about the loaded model: type, version, accuracy and training date."""
    return get_model(request).metadata


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures, request: Request):
    """Predicts the species of one flower."""
    model = get_model(request)
    return model.predict([features.to_row()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    """Predicts species for up to 100 flowers in one request."""
    model = get_model(request)
    predictions = model.predict([sample.to_row() for sample in batch.samples])
    return {"count": len(predictions), "predictions": predictions}

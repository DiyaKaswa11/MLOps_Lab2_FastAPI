# MLOps Lab 2 – Iris Species Prediction API with FastAPI

A REST API that serves a trained machine learning model. Send the measurements of an iris flower and the API returns the predicted species, the model's confidence, and the probability of each species.

Based on **API_Labs / FastAPI_Labs** from the MLOps course (IE7305). The original lab served a Decision Tree through a single `/predict` endpoint that returned only a class number. This version rewrites the API layer: the model is loaded once at startup, inputs are validated, responses are richer, there are new endpoints, and the API is covered by automated tests. All changes are listed in [Changes from the original lab](#changes-from-the-original-lab).

---

## 1. Requirements

| Tool | Version used | Check with |
|---|---|---|
| Python | 3.13.7 (3.10 or newer recommended) | `python3 --version` |
| Git | any recent version | `git --version` |

Python packages (installed in step 3, pinned in `requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| fastapi[standard] | 0.143.3 | Web framework for the API |
| uvicorn | 0.54.0 | Server that runs the API |
| scikit-learn | 1.9.1 | Model training and prediction |
| joblib | 1.6.0 | Saving and loading the model |
| pytest | 9.1.1 | Automated tests |
| httpx | 0.28.1 | Used by the FastAPI test client |

Note: the original lab pinned `scikit-learn==1.5.1`, which does not install on Python 3.13, so all versions were updated.

---

## 2. Get the code

    git clone https://github.com/DiyaKaswa11/MLOps_Lab2_FastAPI.git
    cd MLOps_Lab2_FastAPI

All commands below are run from this project folder.

## 3. Set up the environment

macOS / Linux:

    python3 -m venv fastapi_env
    source fastapi_env/bin/activate
    pip install -r requirements.txt

Windows:

    python -m venv fastapi_env
    fastapi_env\Scripts\activate
    pip install -r requirements.txt

When the environment is active, your terminal line starts with `(fastapi_env)`.

## 4. Train the model

    python -m src.train

Expected output (the version number will be the current date and time):

    Model saved to .../MLOps_Lab2_FastAPI/model/iris_model.pkl
    Test accuracy: 0.9778
    Model version: 20261017193434

This creates `model/iris_model.pkl` and `model/model_metadata.json`. A trained model is already included in the repo, so this step is optional, but it shows the full pipeline.

## 5. Run the tests

    pytest -v

Expected result: `11 passed`.

## 6. Start the API

    uvicorn src.main:app --reload

Expected output includes:

    Uvicorn running on http://127.0.0.1:8000
    Application startup complete.

Keep this terminal open while using the API. Press `Ctrl + C` to stop it.

## 7. Use the API

### Option A: in the browser

Open http://127.0.0.1:8000/docs. This interactive page lists every endpoint. Click an endpoint, then **Try it out**, then **Execute**.

### Option B: from a second terminal with curl

Health check:

    curl http://127.0.0.1:8000/health

Expected:

    {"status":"healthy","model_loaded":true}

Single prediction:

    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

Expected:

    {"class_id":0,"species":"setosa","confidence":1.0,"probabilities":{"setosa":1.0,"versicolor":0.0,"virginica":0.0}}

Batch prediction (up to 100 flowers):

    curl -X POST http://127.0.0.1:8000/predict/batch -H "Content-Type: application/json" -d '{"samples": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}]}'

Expected: `"count":2` and the species `setosa` and `virginica`, each with confidence and probabilities.

Invalid input (negative petal width):

    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": -1}'

Expected: an error with status 422 and the message `Input should be greater than 0`.

---

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Welcome message and list of endpoints |
| GET | `/health` | `healthy` if the model is loaded; returns 503 if not |
| GET | `/model/info` | Model type, version, training date, hyperparameters and test accuracy |
| POST | `/predict` | Predicts one flower: species, class id, confidence and per-class probabilities |
| POST | `/predict/batch` | Predicts up to 100 flowers in one request |

Input rules: all four measurements are required, in centimeters, greater than 0 and at most 10. Anything else returns a 422 error explaining which field is wrong.

---

## Changes from the original lab

### API (`src/main.py`, rewritten)
- **Model loaded once at startup** using FastAPI's lifespan event. The original loaded the model file from disk on every request.
- **Input validation:** each measurement must be greater than 0 and at most 10 cm. The original accepted any number, including negative values.
- **Richer response:** returns the species name, confidence and per-class probabilities instead of only a class number (0, 1 or 2).
- **New endpoints:** `/predict/batch` (up to 100 samples, empty or oversized batches rejected), `/model/info`, and `/health`, which now reports whether the model is actually loaded and returns 503 if it is not.
- **Clear errors:** if the model file is missing, endpoints return 503 with instructions to train it, instead of a generic 500 error.
- **Endpoints changed from `async def` to `def`:** model prediction is CPU work that blocks, and FastAPI runs regular `def` endpoints in a thread pool so they do not block the server.
- API title, description and version shown on the `/docs` page.

### Model loading and prediction (`src/predict.py`, rewritten)
- New `IrisModel` class that keeps the model in memory and returns species, confidence and probabilities for one or many samples.
- Absolute file paths, so the code works from any folder. The original used `../model/...`, which only worked when run from inside `src/`.

### Training (`src/train.py`, `src/data.py`)
- The model is now evaluated on the test set (the original never measured accuracy).
- Saves `model_metadata.json` with model type, version, training time, scikit-learn version, hyperparameters, feature and class names, sample counts and test accuracy. This is served by `/model/info`.
- Stratified train/test split, so each species has the same share in both sets.
- New `src/config.py` holds all paths and names in one place.

### Testing and setup
- New `tests/test_api.py` with 11 automated tests covering every endpoint, valid predictions, and invalid inputs (negative, unrealistic and missing values; empty and oversized batches).
- Updated and pinned package versions so the lab installs on current Python.
- Fixed the original README's run command (`uvicorn app:main` does not work; the correct form here is `uvicorn src.main:app`).

---

## Project structure

    MLOps_Lab2_FastAPI/
    ├── model/
    │   ├── iris_model.pkl         # Trained model
    │   └── model_metadata.json    # Model details served by /model/info
    ├── src/
    │   ├── config.py              # File paths and dataset constants
    │   ├── data.py                # Load and split the Iris data
    │   ├── train.py               # Train, evaluate and save the model
    │   ├── predict.py             # IrisModel class used by the API
    │   └── main.py                # FastAPI app and endpoints
    ├── tests/
    │   └── test_api.py            # 11 API tests
    ├── pytest.ini
    ├── requirements.txt
    └── README.md

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run commands from the project folder (the one containing `src/`), not from inside `src/`.
- **`/health` shows `unhealthy` or predictions return 503**: the model file is missing. Run `python -m src.train`, then restart the server.
- **`Address already in use`**: another server is running on port 8000. Stop it with `Ctrl + C`, or run `uvicorn src.main:app --port 8001`.
- **`command not found: uvicorn`**: the environment is not active. Run `source fastapi_env/bin/activate` again.

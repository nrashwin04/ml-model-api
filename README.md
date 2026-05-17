# 🔢 MNIST Digit Recogniser API

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)

A production-ready REST API for handwritten digit recognition, built with FastAPI and a PyTorch CNN trained on MNIST. Upload any image of a handwritten digit and get a prediction with confidence scores — with full Swagger documentation at `/docs`.

> Built as part of my MSc CS portfolio to demonstrate end-to-end ML deployment — not just model training.

---

## Live Demo

> 🚀 Run locally and visit `http://localhost:8000/docs` for the full interactive Swagger UI

---

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message and links |
| GET | `/health` | Model status and API health check |
| POST | `/predict` | Upload a digit image, get prediction + confidence |

---

## Example

### Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "image=@your_digit.png"
```

### Response
```json
{
  "prediction": 7,
  "confidence": 0.9823,
  "probabilities": [0.001, 0.002, 0.003, 0.004, 0.001, 0.002, 0.003, 0.982, 0.001, 0.001],
  "model_version": "1.0.0"
}
```

---

## Architecture

```
POST /predict (image)
    → utils.py     — preprocessing (greyscale, threshold, crop, normalise)
    → model.py     — CNN inference (singleton, loaded once on startup)
    → schemas.py   — Pydantic response validation
    → JSON response
```

The model is loaded once on startup via `ModelManager` and held in memory — no reloading per request.

---

## Project structure

```
ml-model-api/
├── app/
│   ├── main.py           # FastAPI app, routes, startup event
│   ├── model.py          # ModelManager — loading and inference
│   ├── schemas.py        # Pydantic request/response models
│   └── utils.py          # image preprocessing pipeline
├── mnist_cnn.py          # CNN architecture (from training project)
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model

The CNN architecture consists of:
- 2× Conv2D layers with BatchNorm and Dropout2D
- MaxPooling after each conv block
- Fully connected layer with Dropout
- Trained on MNIST — 99%+ test accuracy

> **Note:** `mnist_cnn.pth` is excluded via `.gitignore` due to file size. Train the model or download weights before running.

**To train the model:**
```bash
python mnist_cnn.py
```
This will train and save `mnist_cnn.pth` automatically.

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/nrashwin04/ml-model-api.git
cd ml-model-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train or place mnist_cnn.pth in root directory
python mnist_cnn.py

# 4. Start the API
uvicorn app.main:app --reload

# 5. Open Swagger UI
# http://localhost:8000/docs
```

---

## Run with Docker

```bash
# Build
docker build -t mnist-api .

# Run (mount weights file since it's gitignored)
docker run -p 8000:8000 -v $(pwd)/mnist_cnn.pth:/app/mnist_cnn.pth mnist-api
```

---

## Tech stack

| Layer | Tech |
|---|---|
| API framework | FastAPI |
| Inference | PyTorch |
| Input validation | Pydantic |
| Image processing | Pillow |
| Server | Uvicorn |
| Containerisation | Docker |

---

## Roadmap

- [x] /predict endpoint with confidence scores
- [x] /health endpoint with model status
- [x] Swagger docs auto-generated
- [x] Pydantic input/output validation
- [x] Docker containerisation
- [ ] Deploy to Railway or Render
- [ ] Add /batch endpoint for multiple images
- [ ] Extend to EMNIST (letters + digits)

---

*Built with FastAPI + PyTorch · MSc CS portfolio project*

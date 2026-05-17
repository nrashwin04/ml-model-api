from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging

from app.schemas import PredictionResponse, HealthResponse, ErrorResponse
from app.model import model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MNIST Digit Recogniser API",
    description="REST API for handwritten digit recognition using a PyTorch CNN trained on MNIST.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    logger.info("Loading model on startup...")
    model_manager.load()
    if model_manager.is_loaded:
        logger.info("Model loaded successfully.")
    else:
        logger.error("Failed to load model.")

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Welcome to the MNIST Digit Recogniser API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    if model_manager.is_loaded:
        return HealthResponse(status="ok", model_loaded=True)
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False, "model_version": "1.0.0"}
        )

@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type", "detail": "Only image files are accepted."}
        )
    
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image", "detail": f"Failed to read image: {str(e)}"}
        )

    try:
        prediction, confidence, probabilities = model_manager.predict(pil_image)
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Inference error", "detail": str(e)}
        )

# How to run locally
# Install dependencies
# pip install -r requirements.txt

# Run the API
# uvicorn app.main:app --reload

# Visit Swagger docs
# http://localhost:8000/docs

# How to run with Docker
# docker build -t mnist-api .
# docker run -p 8000:8000 -v $(pwd)/mnist_cnn.pth:/app/mnist_cnn.pth mnist-api

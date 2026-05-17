import torch
import logging
from PIL import Image
from typing import Tuple, List
from mnist_cnn import CNN
from app.utils import preprocess_image

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.device = None
        self._loaded = False

    def load(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CNN().to(self.device)
            self.model.load_state_dict(torch.load("mnist_cnn.pth", map_location=self.device, weights_only=True))
            self.model.eval()
            self._loaded = True
            logger.info("Model loaded successfully.")
        except Exception as e:
            self._loaded = False
            logger.error(f"Failed to load model: {e}")

    def predict(self, image: Image.Image) -> Tuple[int, float, List[float]]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        
        tensor_image = preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_image)
            probs = torch.softmax(output, dim=1).squeeze().tolist()
            prediction = output.argmax(dim=1, keepdim=True).item()
            confidence = probs[prediction]
            
        return prediction, confidence, probs

    @property
    def is_loaded(self) -> bool:
        return self._loaded

model_manager = ModelManager()

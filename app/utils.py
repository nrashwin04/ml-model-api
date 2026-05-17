import torch
from torchvision import transforms
from PIL import ImageOps, Image

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    
    tensor_check = transforms.ToTensor()(image)
    if tensor_check.mean() > 0.5:
        image = ImageOps.invert(image)
        
    image = image.point(lambda p: p if p > 80 else 0)
    
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        
    transform = transforms.Compose([
        transforms.Resize((20, 20), antialias=True),
        transforms.Pad(4, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0)

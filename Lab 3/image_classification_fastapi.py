from PIL import Image
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import ViTImageProcessor, ViTForImageClassification
import logging
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class ImageRequest(BaseModel):
    """
    A simple class to encapsulate an image URL for request.
    
    Attributes:
    - url (str): The URL of the image to be classified.
    """
    url: str


app = FastAPI(title="Image Classification API", description="API for classifying images using Vision Transformer (ViT).")

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def load_image(url):
    """
    Function to load an image from a URL.
    
    Args:
    - url (str): URL of the image.
    
    Returns:
    - Image: PIL.Image object.
    """
    img = Image.open(requests.get(url, stream=True).raw)
    return img


def image_classification(picture):
    """
    Classifies an image using the pre-trained ViT model.
    
    Args:
    - picture (Image): PIL.Image object to classify.
    
    Returns:
    - str: Predicted class label.
    """
    inputs = processor(images=picture, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint that greets users.
    
    Returns:
    - dict: A welcome message.
    """
    return {"message": "Welcome to the Image Classification API!"}


@app.post("/classify-image", tags=["Image Classification"])
def classify_image(request: ImageRequest):
    """
    Endpoint to classify an image by its URL.
    
    Args:
    - request (ImageRequest): Object containing the image URL.
    
    Returns:
    - dict: Contains the classification result or an error message.
    """
    logger.info("Received classification request for URL: %s", request.url)
    try:
        loaded_image = load_image(request.url)
        result = image_classification(loaded_image)
        logger.info("Classification result: %s", result)
        return {"classification": result}
    except IOError as e:
        logger.error("Failed to load image: %s", str(e))
        return {"error": f"Failed to load image: {str(e)}"}
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        return {"error": f"Unexpected error: {str(e)}"}

def load_image(url):
    """
    Function to load an image from a URL.
    
    Args:
    - url (str): URL of the image.
    
    Returns:
    - Image: PIL.Image object.
    
    Raises:
    - RequestException: If there is an issue with the network request.
    - IOError: If there is an issue loading the image.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw)
        return img
    except RequestException as e:
        logger.error("Network error occurred: %s", str(e))
        raise
    except IOError as e:
        logger.error("IO error occurred: %s", str(e))
        raise

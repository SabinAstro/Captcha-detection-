
import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    image = np.array(image)
    image = cv2.resize(image, (105, 105))
    image = image.astype("float32") / 255.0
    return image

from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

app = Flask(__name__)

# Load model and label binarizer
model = load_model("captcha_breaker_model.h5")
CHAR_SET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
lb = LabelBinarizer()
lb.fit(list(CHAR_SET))

def decode_prediction(preds):
    return ''.join(lb.classes_[np.argmax(p)] for p in preds)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (160, 60)) / 255.0
    image = image.reshape(1, 60, 160, 1)
    preds = model.predict(image)
    prediction = decode_prediction(preds)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import os
from utils.preprocess import preprocess_image

app = Flask(__name__)
model = load_model("model/siamese_model.h5")

THRESHOLD = 0.5  # Similarity threshold

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify_signature():
    if 'ref' not in request.files or 'test' not in request.files:
        return jsonify({"error": "Upload both 'ref' and 'test' signature images."}), 400

    ref_img = preprocess_image(request.files['ref'].read())
    test_img = preprocess_image(request.files['test'].read())

    ref_img = ref_img.reshape(1, 105, 105, 1)
    test_img = test_img.reshape(1, 105, 105, 1)

    pred = model.predict([ref_img, test_img])[0][0]
    is_genuine = pred < THRESHOLD

    return jsonify({
        "similarity_score": float(1 - pred),
        "result": "Genuine" if is_genuine else "Forged"
    })

if __name__ == "__main__":
    app.run(debug=True)

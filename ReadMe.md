
# Signature Verification System

This project uses a Siamese Neural Network to verify whether a test signature is genuine or forged by comparing it to a reference signature.

## Features
- Siamese Network architecture
- Preprocessing with OpenCV and PIL
- REST API (Flask)
- Simple web interface for uploads
- Ready for cloud deployment

## Usage

### Local
```bash
pip install -r requirements.txt
python app.py
```

### Train Model
```bash
python model/siamese_network.py
```
Ensure your dataset is organized in: `dataset/{writer}/genuine/*.png` and `dataset/{writer}/forged/*.png`

### Web UI
Open `http://localhost:5000` in your browser. Upload a reference and test signature.

### API Endpoint
`POST /verify`
- Form-data fields: `ref` (reference image), `test` (test image)

### Response
```json
{
  "similarity_score": 0.8437,
  "result": "Genuine"
}
```

## Deployment
Use Render, Railway, or Fly.io. Ensure you include the `model/siamese_model.h5` file in the `model/` folder.

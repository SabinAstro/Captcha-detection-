# CAPTCHA Breaker API

This project is a deep learning-based CAPTCHA solver served as a Flask REST API. It loads a pre-trained CNN model (`captcha_breaker_model.h5`) and predicts CAPTCHA characters from uploaded images.

## Features
- Accepts grayscale CAPTCHA images
- Predicts 5-character alphanumeric CAPTCHA codes
- Fast and deployable anywhere

## How to Use

### Run Locally
```bash
pip install -r requirements.txt
python app.py
```

### API Endpoint
**POST /predict**
- Form field: `file` - your image file (.png, .jpg)

Example:
```bash
curl -X POST -F "file=@captcha.png" http://localhost:5000/predict
```

Response:
```json
{
  "prediction": "A9C2K"
}
```

## Deployment
Use [Render](https://render.com), Railway, or Fly.io for deployment.

**Procfile** ensures it works with Gunicorn in production:
```txt
web: gunicorn app:app
```

---

### captcha_breaker_model.h5
> Make sure this file exists in the same directory. It is your trained TensorFlow model. You can generate it with your training script.

If you need help training or exporting the model, let me know!

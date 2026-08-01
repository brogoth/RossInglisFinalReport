# AI-Powered Sustainability Advisor

## Requirements
- Python 3.10+
- A browser

## Setup

cd backend
pip install -r requirements.txt --break-system-packages

## Run it

Terminal 1:
cd backend
python app.py

Wait for `Running on http://127.0.0.1:5001`, then leave it running.

Open `frontend/index.html` in your browser (double click it or open with your browser).

Fill in the form and click "Get my recommendations."

Use "Reset demo data" to clear between tests.

To stop the backend, press Ctrl+C in Terminal 1.

## Retrain the model (optional)

cd backend
python train_model.py

# Urea Level Prediction System (Agriculture)

This repository contains a simple Urea Level Prediction System built as a demonstration project.
It uses a synthetic dataset to train a regression model that predicts recommended **urea application level**
based on soil, weather and crop parameters.

## What is included
- `data/sample_data.csv` — synthetic training dataset (1,000 samples)
- `data/sample_predict.csv` — example input for prediction
- `src/generate_synthetic_data.py` — script to regenerate synthetic data
- `src/train.py` — script to train the model and save it to `models/urea_model.pkl`
- `src/predict.py` — script to load the model and make predictions from a CSV
- `models/urea_model.pkl` — trained RandomForest model (pretrained on synthetic data)
- `app/streamlit_app.py` — simple Streamlit app for making single-row predictions
- `requirements.txt` — Python dependencies

## Quick start

1. Clone or download this project and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Train model (optional, model is already provided):
```bash
python src/train.py --data data/sample_data.csv --out models/urea_model.pkl
```

3. Predict from CSV:
```bash
python src/predict.py --model models/urea_model.pkl --input data/sample_predict.csv --output predictions.csv
```

4. Run Streamlit demo:
```bash
streamlit run app/streamlit_app.py
```

## Notes
- This project uses **synthetic** data for demonstration. Replace `data/sample_data.csv` with a real dataset for a production-ready solution.
- Feature engineering, hyperparameter tuning, and validation with field data are required before using in real agricultural decision-making.


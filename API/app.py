from flask import Flask, request, jsonify
import pandas as pd
import pickle

from transformers import MapColumnTransformer, TopEncoderTransformer, DatetimeFeatureTransformer, \
    FillNAAndStr, DropOriginalColumnsTransformer

app = Flask(__name__)

MODEL_PATH = '../v1/models/XGBoost.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        pred_proba = model.predict_proba(input_df)[0, 1]
        return jsonify({'prediction': float(pred_proba), 'model_name': 'XGBoost'})

    except Exception as e:
        print("❌ Ошибка:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()

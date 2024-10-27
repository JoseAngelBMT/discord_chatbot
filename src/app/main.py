from flask import Flask, request, jsonify
from typing import Dict, Optional
from dotenv import load_dotenv
import os
from src.model.GPT2_model import GPT2FineTuning


load_dotenv('config.env')

app = Flask(__name__)
model = GPT2FineTuning()

VALID_API_KEYS = os.environ.get("VALID_API_KEYS").split(',')


def generate_prediction(prompt: str) -> str:
    return model.predict(prompt)


@app.route('/predict', methods=['GET'])
def get_prediction():
    api_key: Optional[str] = request.headers.get('API-KEY')

    if api_key not in VALID_API_KEYS:
        return jsonify({"error": "Unauthorized"}), 401

    data: str = str(request.args['prompt'])
    print(data)
    if not data:
        return jsonify({"error": "prompt is necessary"}), 400

    prompt: str = data

    response: str = generate_prediction(prompt)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0', port=5000, debug=True)

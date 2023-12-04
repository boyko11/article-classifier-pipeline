from flask import Flask, request, jsonify
import torch

from pipeline_steps.step3_data_preprocessor import DataPreprocessor
from pipeline_steps.step50_smoke_test_data_repo import DataRepo
from pipeline_steps.step52_trained_model_loader import ModelLoader

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texts = data['texts']

    print(texts)
    print('------------')

    embeddings = [DataPreprocessor().text_to_embedding(text) for text in texts]
    embeddings_tensor = torch.stack(embeddings)
    with torch.no_grad():
        predictions = ModelLoader.load()(embeddings_tensor)

    predicted_category_indices = predictions.argmax(dim=1).tolist()
    predicted_category_names = DataRepo.int_labels_to_words(predicted_category_indices)

    response = {'predictions': predicted_category_names}

    print(response)
    print('------------')

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

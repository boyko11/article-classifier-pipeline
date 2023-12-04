import os
import torch

from pipeline_steps.step4_model_trainer import TextClassifier


class ModelLoader:
    _model_cache = None

    @staticmethod
    def load():
        if ModelLoader._model_cache is None:
            model = TextClassifier()
            model_path = os.path.join('model_latest', 'model_state_dict_latest.pth')
            model.load_state_dict(torch.load(model_path))
            model.eval()
            ModelLoader._model_cache = model
        return ModelLoader._model_cache

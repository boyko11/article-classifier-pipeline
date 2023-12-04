import os
import shutil
import time

import torch
import subprocess

from entities.pipeline_node_abstract import PipelineNode
from pipeline_steps.step3_data_preprocessor import DataPreprocessor
from pipeline_steps.step4_model_trainer import TextClassifier
from pipeline_steps.step50_smoke_test_data_repo import DataRepo


class ModelDeployer(PipelineNode):

    def __init__(self, image_name="ag-news-model-endpoint", container_name="ag-news-model-endpoint",
                 port_mapping="5000:5000", model_repo_dir='model_repo', model_latest_dir="model_latest"):
        self.image_name = image_name
        self.container_name = container_name
        self.port_mapping = port_mapping
        self.model_repo_dir = model_repo_dir
        self.model_latest_dir = model_latest_dir

    @staticmethod
    def pre_deploy_smoke_test(model):
        smoke_test_data_repo = DataRepo()
        model.eval()

        data_preprocessor = DataPreprocessor()
        embeddings = []
        for idx, text in enumerate(smoke_test_data_repo.get_smoke_test_data()):
            embeddings.append(data_preprocessor.text_to_embedding(text))

        embeddings_torch = torch.stack(embeddings)
        projected_labels_torch = model(embeddings_torch)

        # one-hot encoded offset for 0 indexing
        _, max_indices = torch.max(projected_labels_torch, 1)
        projected_labels = max_indices + 1

        assert smoke_test_data_repo.get_smoke_test_labels() == projected_labels.tolist()

        print("Smoke Test passed.")

    def build_docker_image(self):
        subprocess.run(["docker", "build", "-t", self.image_name, "."])
        print("-----Docker image built.")

    def stop_and_remove_container(self):
        subprocess.run(["docker", "stop", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        subprocess.run(["docker", "rm", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"-----{self.container_name} stopped and removed")
        time.sleep(2)

    def run_docker_container(self):
        self.stop_and_remove_container()
        subprocess.run(["docker", "run", "-d", "-p", self.port_mapping, "--name", self.container_name, self.image_name])
        print(f"-----{self.container_name} for image {self.image_name} started with port mapping: {self.port_mapping}")

    def run(self):
        model_versions = [os.path.join(self.model_repo_dir, file) for file in os.listdir(self.model_repo_dir)]
        latest_model_version = max(model_versions, key=os.path.getctime)

        model = TextClassifier()
        model.load_state_dict(torch.load(latest_model_version))

        ModelDeployer.pre_deploy_smoke_test(model)

        # Promote the model: copy the latest model to the model_latest_dir
        destination_file_path = os.path.join(self.model_latest_dir, 'model_state_dict_latest.pth')
        shutil.copy(latest_model_version, destination_file_path)
        print(f"Copied {latest_model_version} to {destination_file_path}")

        self.build_docker_image()

        print()

        self.run_docker_container()

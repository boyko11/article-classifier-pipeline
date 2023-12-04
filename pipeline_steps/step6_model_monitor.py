import requests
import time

from entities.pipeline_node_abstract import PipelineNode
from pipeline_steps.step50_smoke_test_data_repo import DataRepo


class ModelMonitor(PipelineNode):

    def __init__(self, endpoint_url='http://localhost:5000/predict', sleep_interval=10):
        self.smoke_test_data_repo = DataRepo()
        self.endpoint_url = endpoint_url
        self.sleep_interval = sleep_interval

    def run(self):
        print("Firing up Monitor...")
        time.sleep(self.sleep_interval)

        test_data = self.smoke_test_data_repo.get_smoke_test_data()
        test_labels = DataRepo.int_labels_to_words(self.smoke_test_data_repo.get_smoke_test_labels(), index_start=1)

        while True:
            response = requests.post(self.endpoint_url, json={"texts": test_data})
            if response.status_code == 200:
                predicted_labels = response.json()['predictions']
                assert predicted_labels == test_labels, f"Test failed. Expected {test_labels}, got {predicted_labels}"
                print("Model monitor: Test passed.")
            else:
                print(f"Model monitor: Failed to get a response, status code {response.status_code}")

            time.sleep(self.sleep_interval)
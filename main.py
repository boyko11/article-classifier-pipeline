from step1_dataset_retriever import DatasetRetrieverAGNews
from step2_eda import Eda
from step3_data_preprocessor import DataPreprocessor
from step4_model_trainer import ModelTrainer
from step5_model_deployer import ModelDeployer
from step6_model_monitor import ModelMonitor


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self):
        data = None
        for step in self.steps:
            args = data if isinstance(data, tuple) else (data,) if data is not None else ()
            data = step.run(*args)


pipeline_ag_news = Pipeline([
    DatasetRetrieverAGNews(),
    Eda(),
    DataPreprocessor(),
    ModelTrainer(),
    ModelDeployer(),
    ModelMonitor()
])

pipeline_ag_news.run()

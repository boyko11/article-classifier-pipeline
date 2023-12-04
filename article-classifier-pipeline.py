from pipeline_steps.step1_dataset_retriever import DatasetRetrieverAGNews
from pipeline_steps.step2_eda import Eda
from pipeline_steps.step3_data_preprocessor import DataPreprocessor
from pipeline_steps.step4_model_trainer import ModelTrainer
from pipeline_steps.step5_model_deployer import ModelDeployer
from pipeline_steps.step6_model_monitor import ModelMonitor


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

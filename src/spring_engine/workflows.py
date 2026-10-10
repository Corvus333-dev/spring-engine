from spring_engine.config import DataConfig, DirConfig
from spring_engine.pipeline import data

def build_data():
    data_cfg = DataConfig()
    dir_cfg = DirConfig(data_cfg)

    data_pipeline = data.DataPipeline(data_cfg, dir_cfg)
    data_pipeline.run()
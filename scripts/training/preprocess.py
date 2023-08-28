import logging
from pathlib import Path
import pandas as pd
import yaml
from src.processing import features, target, file_handling


def preprocess(CONFIG):
    file_handling.combine_datasets(CONFIG)  # this will get replaced when there is a larget datastore
    df = pd.read_csv(
        '/Users/tonderaimadamba/Documents/Client Work/Ryan/ryan-MLops/rpm-mvp/data/processed___main/_processed_input_data.csv') #TODO: Tonderai How it saved on my local
    df = df[~df['Category Code'].isna()].sample(100).reset_index()
    df = features.combine_feature_columns(CONFIG, df)
    df = target.encode_target(CONFIG, df)
    df.to_csv('/Users/tonderaimadamba/Documents/Client Work/Ryan/ryan-MLops/rpm-mvp/data/processed/processed.csv') #TODO: Tonderai How it saved on my local


if __name__ == '__main__':
    # Setup config and logging
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    logging.basicConfig(
        level=CONFIG["log_level"],
        format=CONFIG["log_format"],
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info("---Preprocessing data---")
    preprocess(CONFIG)

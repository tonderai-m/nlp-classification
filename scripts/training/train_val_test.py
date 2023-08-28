import logging
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

import src.model.pytorch
from src.model.utils import fetch_logged_data
from src.evaluation.metrics import multi_class_metrics
from src.model.utils import locate_model
from src.processing import tokenize
from transformers import DistilBertTokenizer, TFDistilBertModel


def train_val_test(CONFIG, experiment, model):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model) as run:
        # Log model type
        mlflow.set_tag("model_type", model)

        # Load data splits
        load_dir = Path("/".join([CONFIG["data_storage"]["local_data_path"], "processed"]))
        file_name = 'processed'
        extension = CONFIG["data_storage"]["save_extension"]
        train_df = pd.read_csv(str(load_dir / file_name) + "_train" + extension, converters={'target': pd.eval})
        val_df = pd.read_csv(str(load_dir / file_name) + "_val" + extension, converters={'target': pd.eval})
        test_df = pd.read_csv(str(load_dir / file_name) + "_test" + extension, converters={'target': pd.eval})
        logging.info(f"Loaded training datasets from {load_dir}")

        model = locate_model(model)(
            seed=CONFIG["seed"],
            hidden_dim=CONFIG["model"]["hidden_dim"],
            learning_rate=CONFIG["model"]["learning_rate"],
            #Todo: Tonderai length should equal one because we combined all the columns. Will fix it to automatically get the column length
            #input_dim=len(CONFIG["model"]["feature_cols"]),
            input_dim=1,

            total_steps=CONFIG["model"]["batch_size"] * train_df['feature'].shape[0] //
                        CONFIG["model"]["batch_size"],
            output_dim=len(train_df['target'][0])
        )

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Create data loaders
        train_tensor = tokenize.TextDataset(train_df, tokenizer, CONFIG['model']['max_len'])
        # val_tensor = tokenize.TextDataset(val_df, tokenizer, CONFIG['model']['max_len'])
        # test_tensor = tokenize.TextDataset(test_df, tokenizer, CONFIG['model']['max_len'])
        # train_loader = model.tensor_to_loader(train_tensor, CONFIG["model"]["batch_size"],
        #                                       CONFIG["model"]["num_workers"], shuffle=True)
        # val_loader = model.tensor_to_loader(val_tensor, CONFIG["model"]["batch_size"], CONFIG["model"]["num_workers"],
        #                                     shuffle=False)
        # test_loader = model.tensor_to_loader(test_tensor, CONFIG["model"]["batch_size"], CONFIG["model"]["num_workers"],
        #                                      shuffle=False)

        # # Setup and run trainer
        # trainer = model.setup_trainer(experiment, run.info.run_id, CONFIG["model"]["max_epochs"])
        # trainer.fit(model, train_loader, val_loader)
        # # trainer.test(test_dataloader=test_loader)
        # mlflow.pytorch.log_model(
        #     model,
        #     "pytorch_model",
        #     signature=mlflow.models.signature.infer_signature(
        #         train_df['feature'],
        #         train_df['target'],
        #     ),
        # )
        return train_tensor

        # Calculate and log metrics
        # mlflow.log_metrics(multi_class_metrics(train_loader,
        #                                        model.predict(train_tensor['encodings']), "train"))
        # mlflow.log_metrics(multi_class_metrics(val_tensor,
        #                                        model.predict(val_tensor['encodings']), "val"))
        # mlflow.log_metrics(multi_class_metrics(test_tensor,
        #                                        model.predict(test_tensor['encodings']), "test"))

    # fetch logged data
    # params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    # for key, val in metrics.items():
    #     logging.info(f"{key}:{round(val, 3)}")


if __name__ == "__main__":
    # Grab command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model type", default="src.model.pytorch.MultiClassClassifier")
    parser.add_argument("--experiment", help="experiment id", default="None")
    args = parser.parse_args()

    # Setup config and logging
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    logging.basicConfig(
        level=CONFIG["log_level"],
        format=CONFIG["log_format"],
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info("---Training and validating model---")
    train_val_test(CONFIG, args.experiment, args.model)

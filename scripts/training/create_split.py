import logging
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split 


def create_split(CONFIG):
    # Load raw data
    load_dir_processed_binarized = Path("/".join([CONFIG["data_storage"]["local_data_path"], "processed___main"]))
    to_split_this =  pd.read_csv(load_dir_processed_binarized / (CONFIG["data_storage"]["binarized_file_name"]+ CONFIG["data_storage"]["save_extension"]))

    train_size = CONFIG["preprocessing"]["train_ratio"]
    val_size = CONFIG["preprocessing"]["val_ratio"]
    train_idx, validation_test_idx = train_test_split(to_split_this.index, train_size = train_size, random_state=CONFIG['seed'])
    val_idx,test_idx = train_test_split(validation_test_idx, train_size = val_size/(1-train_size), random_state=CONFIG['seed'])

    train_df, val_df, test_df = (
        to_split_this.iloc[train_idx],
        to_split_this.iloc[val_idx],
        to_split_this.iloc[test_idx],
    )

    # binarized_target[CONFIG["model"]["target_col_binarized"]] = encode_target(binarized_target[CONFIG["data_storage"]["target_col_input"]])
    # binarized_target_final = binarized_target[[CONFIG["model"]["feature_combined_col_names"],CONFIG["model"]["target_col_binarized"]]]
    # Save data splits

    save_dir = Path("/".join([CONFIG["data_storage"]["local_data_path"], "Processed_split"]))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        save_dir / (CONFIG["data_storage"]["processed_split_file_names"][0]+ CONFIG["data_storage"]["target_extension"]),
    )
    val_df.to_csv(
        save_dir / (CONFIG["data_storage"]["processed_split_file_names"][1]+ CONFIG["data_storage"]["target_extension"]),
    )
    test_df.to_csv(
        save_dir / (CONFIG["data_storage"]["processed_split_file_names"][2]+ CONFIG["data_storage"]["target_extension"]),
    )
    logging.info("Split data into train, val, test sets")
    # # TODO tonderai needs fixing the UPC code needs to be a string import everything as a string


    return  train_df, val_df, test_df


if __name__ == "__main__":
    # Setup config and logging
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    logging.basicConfig(
        level=CONFIG["log_level"],
        format=CONFIG["log_format"],
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info("---Splitting processed data---")
    create_split(CONFIG)

import logging
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split(CONFIG):
    # Load processed data
    load_dir = Path("/".join([CONFIG["data_storage"]["local_data_path"], "processed"]))
    file_name = CONFIG["data_storage"]["intermediate_file_name"]
    extension = CONFIG["data_storage"]["save_extension"]
    df = pd.read_csv(str(load_dir / file_name) + extension)
    logging.info(f"Loaded {df.shape[0]} rows of data")

    # Check split configs
    assert CONFIG["preprocessing"]["train_ratio"] + CONFIG["preprocessing"]["val_ratio"] + CONFIG["preprocessing"]["test_ratio"] == 1
    assert CONFIG["preprocessing"]["train_ratio"] > 0
    assert CONFIG["preprocessing"]["val_ratio"] > 0
    assert CONFIG["preprocessing"]["test_ratio"] > 0

    # check for the correct number of each index

    category_count_df = df.groupby('Category Code')['Category Code'].count().rename_axis('Category')
    category_list = category_count_df.pipe(lambda  dfx: dfx.loc[dfx>=10]).reset_index(name='Count')['Category'].tolist()
    df = df[df['Category Code'].isin(category_list)]
    df = df.reset_index()

    # Split data
    idx = list(range(df.shape[0]))
    train_idx, val_idx = train_test_split(
        idx,
        train_size=CONFIG["preprocessing"]["train_ratio"],
        stratify=df[CONFIG["model"]["target_col_name"]],
        random_state=CONFIG["seed"]
    )
    val_test_ratio = CONFIG["preprocessing"]["val_ratio"] / \
                     (CONFIG["preprocessing"]["val_ratio"] + CONFIG["preprocessing"]["test_ratio"])
    val_idx, test_idx = train_test_split(
        val_idx,
        train_size=val_test_ratio,
        stratify=df.loc[val_idx,
                        CONFIG["model"]["target_col_name"]],
        random_state=CONFIG["seed"]
    )

    train_df, val_df, test_df = (
        df.iloc[train_idx],
        df.iloc[val_idx],
        df.iloc[test_idx],
    )
    logging.info("Split data into train, val, test sets")

    # Save data splits
    save_dir = Path("/".join([CONFIG["data_storage"]["local_data_path"], "processed"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(
        str(save_dir / file_name) + "_train" + extension,
        index=False,
    )
    val_df.to_csv(
        str(save_dir / file_name) + "_val" + extension,
        index=False,
    )
    test_df.to_csv(
        str(save_dir / file_name) + "_test" + extension,
        index=False,
    )
    logging.info(f"Saved datasets to {save_dir}")
    #Todo: Tonderai dont forget to fix train test split or split.py the len(train + test + val) does not equal len(processed)


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
    split(CONFIG)

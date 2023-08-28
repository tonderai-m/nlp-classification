import logging
# import yaml
import glob
from pathlib import Path
import pandas as pd

def combine_datasets(CONFIG):

    load_dir = Path("/".join([CONFIG["data_storage"]["local_data_path"], "input"]))
    csv_files = glob.glob(str(load_dir) + CONFIG["data_storage"]["input_extension"])
    df_list = []
    for filename in csv_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    # df = df.sample(100) #TODO Tonderai needs to be removed when running the whole model we are just sampling 100 data points to make sure it works
    # df = df.fillna('nan') #TODO Tonderai not needed added replaced with df = df[~df['Category Code'].isna()].reset_index() needs to move up the tree 
    df = df[~df['Category Code'].isna()].reset_index()

    process_file_name = CONFIG["data_storage"]["processed_main_file_names"] + CONFIG["data_storage"]["save_extension"]
    save_dir = Path("/".join([CONFIG["data_storage"]["local_data_path"], "processed___main"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / process_file_name)

    logging.info(f"Saved processed_target data to {save_dir / process_file_name}")
    return df #TODO Tonderai needs to be removed when running this has no purpose at all will get file from directory instead


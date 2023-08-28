from pathlib import Path
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer
import logging
import yaml
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from pickle import dump


def encode_target(CONFIG, df):
    """
    using the target column of the dataframe, onehotencode the target column
    :param CONFIG:
    :param df:
    """
    # label binarizer
    lb = LabelBinarizer()
    labels = lb.fit_transform(df['Category Code']).tolist()
    df[CONFIG['model']['target_col_name']] = pd.Series(labels)
    # add some kind of naming convention to save the binaraizer

    target_name = CONFIG["execution"]["target_col_binarizer_name"] + CONFIG["execution"]["pickle_extension"]
    save_dir = Path("/".join([CONFIG["execution"]["local_execution_path"], "binarizer"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    dump(lb, open(save_dir / target_name, 'wb'))
    logging.info(f"Saved binarizer data to {save_dir / target_name}")
    return df


def decode_target(CONFIG, df):
    pass
    # TODO do something here not sure yet Tonderai will handle it
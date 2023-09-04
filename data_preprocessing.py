import logging
import glob
from pathlib import Path
import pandas as pd
from sklearn.utils import resample
from random import sample

def injestDataset():

    load_dir = Path("data/Womens Clothing E-Commerce Reviews.csv")
    data = pd.read_csv(load_dir, index_col=None, header=0)
    df = data[['Review Text', 'Rating','Recommended IND']]
    df = df.dropna(how="any")

    logging.info(f"Saved processed_target data to {load_dir}")
    return df

def downSample(df,columnName,n):
    """
    The purpose of this down sample is to have equal ratings of [1,2,3,4,5] or good v bad classes
    """
    good = df[df[columnName] == 1]
    bad = df[df[columnName] == 0].sample(n=round(n/2), random_state=42)
    df_downsampled = resample(good,
             replace=False,
             n_samples=len(bad),
             random_state=42)
    df = pd.concat([df_downsampled,bad])
    df = df.reset_index()
    df.rename(columns={'index': 'trueIndex'}, inplace=True)
    return df

    


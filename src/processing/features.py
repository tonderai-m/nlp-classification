# TODO tonderai, this is where you can combine the text

def combine_feature_columns(CONFIG, df):
    """
    we take all the column names from config.yaml and create a new column. we return only 2 columns from that

    :param CONFIG: config file in main directory
    :param df: df with features and target raw
    """
    df[CONFIG["model"]["feature_name"]] = df[CONFIG["model"]["feature_cols"]].astype(str).agg(', '.join, axis=1)

    return df[[CONFIG["model"]["feature_name"], "Category Code"]]


########################################################################################################################

# You can add more feature transformations here

########################################################################################################################

if __name__ == "__main__":
    print('works')
    # add unit tests here

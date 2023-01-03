from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

import lightgbm as lgbm 
import pandas as pd
import numpy as np
import argparse
import pickle
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to input data")
    args = parser.parse_args()

    # Get a handle to the workspace
    ml_client = MLClient.from_config(credential=credential)

    # all versions will have the date encoded in the version number
    version_num = datetime.datetime.now().strftime("%Y%m%d")

    df = pd.read_csv(args.train_data)
    df.head()

    X = np.array(df.drop(["Unnamed: 0", "close", "close_shifted"], axis=1))
    y = np.array(df["close_shifted"])

    # Train model on whole dataset
    lgbm_model = lgbm.LGBMRegressor(max_depth=100, reg_alpha=0.05, reg_lambda=0.05).fit(X, y)

    with open('ibm_model.pkl', 'wb') as f:
        pickle.dump(lgbm_model, f)

    file_model = Model(
        path="ibm_model.pkl",
        type=AssetTypes.CUSTOM_MODEL,
        name="IBM-Model",
        description="Model created from local file.",
        version=version_num
    )
    ml_client.models.create_or_update(file_model)

if __name__ == "__main__":
    main()

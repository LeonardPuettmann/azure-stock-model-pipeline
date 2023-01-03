
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path or URL to input data")
    parser.add_argument("--train_data", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input_data)
    df = df.drop("adjusted_close", axis=1)

    df = prep_data(df)
    df = df.fillna(method="ffill")

    path = os.path.join(args.train_data, "stock-data.csv")
    df.to_csv(path)

    stock_data = Data(
        name="stock-data",
        path=path,
        type=AssetTypes.URI_FILE,
        description="Dataset to train a model on the IBM stock data.",
        tags={"source_type": "web", "source": "AlphaVantage"},
    )

    credit_data = ml_client.data.create_or_update(stock_data)

def prep_data(dataframe):
    # get rolling mean an exponential moving average
    dataframe["rolling_3_mean"] = dataframe["close"].shift(1).rolling(3).mean()
    dataframe["rolling_7_mean"] = dataframe["close"].shift(1).rolling(3).mean()
    dataframe["ewma"] = dataframe["close"].shift(1).ewm(alpha=0.5).mean()

    # convert timestamp to unix timecode 
    dataframe['unix_timestamp'] = pd.to_datetime(dataframe['timestamp']).values.astype(int)/ 10**9

    # get day of week and month
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe["weekday"] = dataframe['timestamp'].dt.dayofweek
    dataframe["month"] = dataframe['timestamp'].dt.month
    dataframe = dataframe.drop("timestamp", axis=1)

    # shift the target column 
    dataframe["close_shifted"] = dataframe["close"].shift(-1)
    return dataframe

if __name__ == "__main__": 
    main()

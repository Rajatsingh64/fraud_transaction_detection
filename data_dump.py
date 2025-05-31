from src.logger import logging
from src.exception import SrcException
from src.config import mongo_client , database_name
import pandas as pd
import os, sys

# Define file path and MongoDB details
file_path = "dataset/card_fraud.csv"
collection_name = "transactions"

try:
    # Read the dataset as a pandas DataFrame
    df = pd.read_csv(file_path)

    # Convert the DataFrame to a list of dictionaries (records) for MongoDB insertion
    df_dict = df.to_dict(orient="records")

    # Insert the data into the specified MongoDB collection
    mongo_client[database_name][collection_name].insert_many(df_dict)

    print("Data successfully inserted into MongoDB Atlas database.")

except Exception as e:
    raise SrcException(e, sys)

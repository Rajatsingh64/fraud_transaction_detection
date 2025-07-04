import pandas as pd
import numpy as np
import os  , sys
import pickle
from src.exception import SrcException
from src.logger import logging
from src.config import mongo_client
import yaml
import dill

def get_collection_as_dataframe(database_name, collection_name):
    """
    This function extracts data from a MongoDB collection and returns it as a pandas DataFrame.

    Parameters:
    - mongo_client (MongoClient): Instance of a MongoDB client connected to MongoDB Atlas.
    - database_name (str): The name of the MongoDB database.
    - collection_name (str): The name of the MongoDB collection.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the collection.
    """
    try:
        # Extract data from the specified collection
        collection = mongo_client[database_name][collection_name]
        
        # Find all documents in the collection and convert to DataFrame
        cursor = collection.find()  # using find() instead of find_all()
        df = pd.DataFrame(list(cursor))  # Convert the cursor to a list and then to a DataFrame
        
        # Removing irrelevant features (e.g., MongoDB's _id)
        if "_id" in df.columns:
            df.drop("_id", axis=1, inplace=True)
        
        return df  

    except Exception as e:
        print(f"Error: {e}")  
        return pd.DataFrame()  

# Load a set of pickle files, put them together in a single DataFrame, and order them by time
# It takes as input the folder DIR_INPUT where the files are stored, and the BEGIN_DATE and END_DATE
def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    #  Note: -1 are missing values for real world data 
    df_final=df_final.replace([-1],0)
    
    return df_final


#############################
# Data Extractor
##############################

def get_relevant_past_df(query, database_name, collection_name):
    """
    Fetch historical transactions from MongoDB based on a given query.

    Args:
        query (dict): MongoDB query to filter documents (e.g., by CUSTOMER_ID or TERMINAL_ID).
        database_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection within the database.
        limit (int, optional): Maximum number of documents to fetch. Defaults to 1000.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved documents (excluding MongoDB _id field).

    Example Query:
        query = {
            "$or": [
                {"CUSTOMER_ID": 12345},
                {"TERMINAL_ID": 67890}
            ]
        }
    """
    try:
        # Access the specified MongoDB collection
        collection = mongo_client[database_name][collection_name]

        # Exclude MongoDB's default _id field
        projection = {"_id": 0}

        # Fetch documents using the query and projection
        cursor = collection.find(query, projection)
        results = list(cursor)

        # Convert the results to a DataFrame
        return pd.DataFrame(results)

    except Exception as e:
        raise SrcException(e,sys)

def store_prediction_records_to_database(mongo_client, database_name, collection_name, data):
    try:
        mongo_client[database_name][collection_name].insert_one(data)
        print("Prediction data successfully dumped into database")
    except Exception as e:
        raise SrcException(e, sys)
    

def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise SrcException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using dill for serialization.

    Parameters:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to be serialized and saved.

    Process:
        - Logs entry into the method.
        - Ensures the directory exists.
        - Serializes and saves the object to the file.
        - Logs exit from the method.
    """
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise SrcException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Loads and returns a Python object from a file using dill.

    Parameters:
        file_path (str): The file path from where the object will be loaded.

    Returns:
        object: The deserialized Python object.

    Process:
        - Checks if the file exists.
        - Opens the file and loads the object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise SrcException(e, sys) 
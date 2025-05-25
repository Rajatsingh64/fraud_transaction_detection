import pandas as pd
import numpy as np
import os  , sys
import pickle


#for data ingestion
def get_collection_as_dataframe(mongo_client, database_name, collection_name):
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
        
        return df  # Return the dataframe

    except Exception as e:
        print(f"Error: {e}")  # Print the exception if anything goes wrong
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

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
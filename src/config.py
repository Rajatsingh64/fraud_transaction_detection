import pymongo as pm 
from dotenv import load_dotenv
from src.logger import logging
from src.exception import SrcException
from dataclasses import dataclass
import os, sys

try:
    # Log and print the process of loading the .env file
    logging.info('Loading .env file...')
    print('Loading .env file...')
    load_dotenv()

    @dataclass
    # Define a dataclass to store environment variables
    class EnvironmentVariables:
        mongo_url: str = os.getenv("MONGO_URL")

    # Create an instance of the environment variables class
    env = EnvironmentVariables()

    # Establish a connection to MongoDB Atlas
    logging.info("Connecting to MongoDB Atlas database...")
    print("Connecting to MongoDB Atlas database...")

    mongo_client = pm.MongoClient(env.mongo_url)

    logging.info("Successfully connected to MongoDB Atlas database.")
    print("Successfully connected to MongoDB Atlas database.")


except Exception as e:
    # Raise a custom exception with detailed error info
    raise SrcException(e, sys)

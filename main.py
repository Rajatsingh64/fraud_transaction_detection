from src.logger import logging
from src.exception import SrcException
from src.pipelines.training_pipeline import run_training_pipeline
import os,sys


if __name__=="__main__":

    try:
        run_training_pipeline()

    except Exception as e:
        raise SrcException(e,sys)
import warnings
warnings.filterwarnings('ignore')
import fasttext
from fasttext import train_unsupervised
import gensim
from gensim.models import FastText
import ast
########################################
#library for parsing and extracting wiki revision data
import os
import re
import io
import sys
import math
import json
import html
import pickle
import difflib
import unicodedata
import bs4
import bz2
import py7zr
import numpy
import mwparserfromhell
import libarchive.public
#import dataset
import wikitextparser as wtp
from pprint import pprint
from wikitextparser import remove_markup, parse
import datetime
from pandas import DataFrame
import random
import pickle
import pandas as pd
import logging
from datasets import *
from models import *
from aggregation import *

########################################
# This libraray is for edit distance model
from collections import Counter
from aion.util.spell_check import SpellCorrector
from fuzzywuzzy import fuzz

#############################################
# Library for cross validation
#scikit learn library
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
#############################################
class Corrections():
    """
    The main class.
    """
    def __init__(self,dataset):
        """
        The constructor.
        """
        self.dataset_name=dataset
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        
        #################loading datasets###########
        self.clean_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", self.dataset_name, "clean.csv"))
        self.dirty_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", self.dataset_name, "dirty.csv"))
        self.clean_df=read_csv_dataset( self.clean_path)
        self.clean_data=self.clean_df
        self.dirty_df=read_csv_dataset(self.dirty_path)
        self.dirty_data=self.dirty_df
        self.total_error=calculate_total_error_realworld(self.clean_df, self.dirty_df)
        self.dirty_org_df=pd.read_csv(self.dirty_path)
        experiment_general(self)
def experiment_general(self):
        print("experiment_data_type_tgeneral is running now")
        perfromance_table = pd.DataFrame(columns = ['data_type', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        if self.dataset_name=="wiki":
            pass
        else:
            dataset_dictionary = {
                "name": self.dataset_name,
                "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", self.dataset_name, "dirty.csv")),
                "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", self.dataset_name, "clean.csv"))}
            
            test_data_path = os.path.join("datasets", self.dataset_name,"general_arg.csv")
            if not os.path.exists(test_data_path):
                print("test data path not exits!") #prepare_testing_datasets_real_world_general_general_agg #prepare_testing_datasets_real_world_general_general
                error_correction=prepare_testing_datasets_real_world_general_general_agg(self.dirty_data,self.clean_data,"general_arg",self.dataset_name)
            else:
                print("test data path exits!")
                error_correction=read_csv_dataset(test_data_path)
            test_data_path = os.path.join("datasets", self.dataset_name,"general_domain.csv") # domain based testing datasets
            
            if not error_correction.empty:
                retrain_data=prepare_dataset_for_retrain_realworld(self.clean_data,self.dirty_data) #error_correction_fasttext_supervised_without_constraints_threshold
                #p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"general")
                model_aggregation_final(self.dirty_org_df,error_correction,retrain_data,self.dataset_name,self.total_error,self.dirty_data,self.clean_data)
if __name__ == "__main__":
    app = Corrections()
    app.experiment_general()
    
    
                

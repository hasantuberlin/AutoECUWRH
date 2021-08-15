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
        self.MODELS = ["Edit_Distance_G","Edit_Distance_G_F", "Edit_Distance_D","Edit_Distance_D_F","Fasttext_G","Fasttext_G_F","Fasttext_D","Fasttext_D_F","Fasttext_Supervised_DC","Fasttext_Supervised_WDC"]
        self.DATASETS=["wiki","tax","hospital","flights","beers"]
        self.result_path=os.path.join("performance_final_dictionary","new")
        #################loading datasets###########
        self.hos_clean_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "hospital", "clean.csv"))
        self.hos_dirty_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "hospital", "dirty.csv"))
        self.hos_clean_df=read_csv_dataset( self.hos_clean_path)
        self.hos_dirty_df=read_csv_dataset(self.hos_dirty_path)
        self.hos_total_error=calculate_total_error_realworld(self.hos_clean_df, self.hos_dirty_df)
        self.hos_dirty_org_df=pd.read_csv(self.hos_dirty_path)

        self.tax_clean_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "tax", "clean.csv"))
        self.tax_dirty_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "tax", "dirty.csv"))
        self.tax_clean_df=read_csv_dataset( self.tax_clean_path)
        self.tax_dirty_df=read_csv_dataset(self.tax_dirty_path)
        self.tax_dirty_org_df=pd.read_csv(self.tax_dirty_path)
        self.tax_total_error=calculate_total_error_realworld(self.tax_clean_df, self.tax_dirty_df)

        self.flights_clean_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "flights", "clean.csv"))
        self.flights_dirty_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "flights", "dirty.csv"))
        self.flights_clean_df=read_csv_dataset( self.flights_clean_path)
        self.flights_dirty_df=read_csv_dataset(self.flights_dirty_path)
        self.flights_total_error=calculate_total_error_realworld(self.flights_clean_df, self.flights_dirty_df)
        self.flights_dirty_org_df=pd.read_csv(self.flights_dirty_path)


        self.beer_clean_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "beers", "clean.csv"))
        self.beer_dirty_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "beers", "dirty.csv"))
        self.beer_clean_df=read_csv_dataset( self.beer_clean_path)
        self.beer_dirty_df=read_csv_dataset(self.beer_dirty_path)
        self.beer_total_error=calculate_total_error_realworld(self.beer_clean_df, self.beer_dirty_df)
        self.beer_dirty_org_df=pd.read_csv(self.beer_dirty_path)



        self.wiki_clean_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "wiki", "clean.csv"))
        self.wiki_dirty_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "wiki", "dirty.csv"))
        self.wiki_clean_df=read_csv_dataset( self.wiki_clean_path)
        self.wiki_dirty_df=read_csv_dataset(self.wiki_dirty_path)
        self.wiki_total_error=calculate_total_error_realworld(self.wiki_clean_df, self.wiki_dirty_df)

        #########################
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.Null_P="Invalid"
        self.Null_R="Invalid"
        self.Null_F="Invalid"
        self.domain_dirty_col=['address_1','city','state','county','state'] #consider only location related column
        self.domain_clean_col=['Address1','City','State','CountyName','city','state'] #consider only location related column
        self.hos_fds=[['city','zip'],['city','county'],['zip','state'],['zip','county'],['county','state']]
        self.hos_fds_att=['city','zip','county','state']
        self.flight_fds=[['flight','act_dep_time'],['flight','sched_dep_time'],['flight','sched_arr_time'],['flight','act_arr_time']]
        self.flight_fds_att=['flight','act_dep_time','sched_dep_time','sched_arr_time','act_arr_time']
        self.tax_fds=[['zip','city'],['zip','state'],['area_code','state']]
        self.tax_fds_att=['zip','city','state','area_code']
        #############Result##################
        self.Data_Types=["general","textual", "numeric", "alphanumeric"]
        self.Data_Errors=["Outlier","Pattern Violation", "Rule Violation", "Knowledge Base Violation"]
        self.Data_Uniqueness=["0_30","31_60","61_100"]
        self.results_data_type={model: {dataset: {data_type: [] for data_type in self.Data_Types} for dataset in self.DATASETS} for model in self.MODELS}
        self.results_data_error={model: {dataset: {data_error: [] for data_error in self.Data_Errors} for dataset in self.DATASETS} for model in self.MODELS}
        self.results_uniqueness={model: {dataset: {data_unique: [] for data_unique in self.Data_Uniqueness} for dataset in self.DATASETS} for model in self.MODELS}
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
            if self.dataset_name=="tax":
                #continue
                fds_col_set=self.tax_fds
                fds_col=self.tax_fds_att
                dirty_data=self.tax_dirty_df
                dirty_data_col= dirty_data.columns.values  
                clean_data= self.tax_clean_df
                clean_data_col=clean_data.columns.values   
                total_error= self.tax_total_error 
                dirty_org_df= self.tax_dirty_org_df      
            elif self.dataset_name=="hospital":
                #continue
                fds_col_set=self.hos_fds
                fds_col=self.hos_fds_att
                dirty_data=self.hos_dirty_df
                dirty_data_col= dirty_data.columns.values  
                clean_data= self.hos_clean_df
                clean_data_col=clean_data.columns.values   
                total_error= self.hos_total_error 
                dirty_org_df =self.hos_dirty_org_df  
            elif self.dataset_name=="flights":
                #continue
                fds_col_set=self.flight_fds
                fds_col=self.flight_fds_att
                dirty_data=self.flights_dirty_df
                dirty_data_col= dirty_data.columns.values  
                clean_data= self.flights_clean_df
                clean_data_col=clean_data.columns.values   
                total_error= self.flights_total_error 
                dirty_org_df= self.flights_dirty_org_df  
            elif self.dataset_name=="beers":
                #continue
                #fds_col_set=self.flight_fds
                #fds_col=self.flight_fds_att
                dirty_data=self.beer_dirty_df
                dirty_data_col= dirty_data.columns.values  
                clean_data= self.beer_clean_df
                clean_data_col=clean_data.columns.values   
                total_error= self.beer_total_error 
                dirty_org_df= self.beer_dirty_org_df  
            test_data_path = os.path.join("datasets", self.dataset_name,"general_arg.csv")
            if not os.path.exists(test_data_path):
                print("test data path not exits!") #prepare_testing_datasets_real_world_general_general_agg #prepare_testing_datasets_real_world_general_general
                error_correction=prepare_testing_datasets_real_world_general_general_agg(dirty_data,clean_data,"general_arg",self.dataset_name)
            else:
                print("test data path exits!")
                error_correction=read_csv_dataset(test_data_path)
            test_data_path = os.path.join("datasets", self.dataset_name,"general_domain.csv") # domain based testing datasets
            
            if not error_correction.empty:
                retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data) #error_correction_fasttext_supervised_without_constraints_threshold
                #p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"general")
                model_aggregation_final(dirty_org_df,error_correction,retrain_data,self.dataset_name,total_error,dirty_data,clean_data)
if __name__ == "__main__":
    app = Corrections()
    app.experiment_general()
    
    
                

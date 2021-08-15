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
class Experiments:
    """
    The main class.
    """
    def __init__(self):
        """
        The constructor.
        """
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.MODELS = ["Edit_Distance_G","Edit_Distance_G_F", "Edit_Distance_D","Edit_Distance_D_F","Fasttext_G","Fasttext_G_F","Fasttext_D","Fasttext_D_F","Fasttext_Supervised_DC","Fasttext_Supervised_WDC"]
        self.DATASETS=["wiki","tax","hospital","flights","beers"]
        self.result_path=os.path.join("performance_final_dictionary","new")
        #################loaing datasets###########
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
    def experiment_general(self):
        print("experiment_data_type_tgeneral is running now")
        perfromance_table = pd.DataFrame(columns = ['data_type', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    continue
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error 
                    dirty_org_df= self.tax_dirty_org_df      
                elif dataset_name=="hospital":
                    #continue
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error 
                    dirty_org_df =self.hos_dirty_org_df  
                elif dataset_name=="flights":
                    continue
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error 
                    dirty_org_df= self.flights_dirty_org_df  
                elif dataset_name=="beers":
                    continue
                    #fds_col_set=self.flight_fds
                    #fds_col=self.flight_fds_att
                    dirty_data=self.beer_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.beer_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.beer_total_error 
                    dirty_org_df= self.beer_dirty_org_df  
                test_data_path = os.path.join("datasets", dataset_name,"general_arg.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!") #prepare_testing_datasets_real_world_general_general_agg #prepare_testing_datasets_real_world_general_general
                    error_correction=prepare_testing_datasets_real_world_general_general_agg(dirty_data,clean_data,"general_arg",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                test_data_path = os.path.join("datasets", dataset_name,"general_domain.csv") # domain based testing datasets
                #if not os.path.exists(test_data_path):
                  #  error_correction_d=prepare_domain_testing_datasets_real_world_general(dirty_data,clean_data,"general",dataset_name,self.domain_dirty_col,self.domain_clean_col)
               # else:
                   # print("test data path exits!")
                   # error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"general")
                            #p,r,f,cfe=error_correction_fasttext_supervised_with_constraints_threshold(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"general")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data) #error_correction_fasttext_supervised_without_constraints_threshold
                            #p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"general")
                            p,r,f,cfe=model_aggregation_final(dirty_org_df,model,error_correction,retrain_data,dataset_name,total_error,dirty_data,clean_data,"general")
                        #error_correction_fasttext_supervised_without_constraints
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_type[model][dataset_name]["general"].append(output_result)
                    perfromance_table.loc[-1] = ["general",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_type, open(os.path.join(self.result_path, "general.dictionary"), "wb"))
        write_csv_dataset("performance/new/general.csv",perfromance_table)   

    def experiment_data_type_textual(self):
        print("experiment_data_type_textual is running now")
        perfromance_table = pd.DataFrame(columns = ['data_type', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="dummy":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    continue
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    continue
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    continue
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error 
                elif dataset_name=="wiki":
                    dirty_data=self.wiki_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.wiki_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.wiki_total_error
                    print(total_error)

                test_data_path = os.path.join("datasets", dataset_name,"textual.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_types(dirty_data,clean_data,"textual",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                #test_data_path = os.path.join("datasets", dataset_name,"textual_domain.csv") # domain based testing datasets
                #if not os.path.exists(test_data_path):
                 #   error_correction_d=prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,"textual",dataset_name,self.domain_dirty_col,self.domain_clean_col)
               # else:
                   # print("test data path exits!")
                   # error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"textual")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"textual")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_type[model][dataset_name]["textual"].append(output_result)
                    perfromance_table.loc[-1] = ["textual",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_type, open(os.path.join(self.result_path, "data_type_score_text_f.dictionary"), "wb"))
        write_csv_dataset("performance/new/datatype_textual.csv",perfromance_table)     
    def experiment_data_type_numeric(self):
        print("experiment_data_type_numeric is running now")
        perfromance_table = pd.DataFrame(columns = ['data_type', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wikii":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                elif dataset_name=="wiki":
                    dirty_data=self.wiki_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.wiki_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.wiki_total_error
                    print(total_error)

                test_data_path = os.path.join("datasets", dataset_name,"numeric.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_types(dirty_data,clean_data,"numeric",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                #test_data_path = os.path.join("datasets", dataset_name,"numeric_domain.csv") # domain based testing datasets
                #if not os.path.exists(test_data_path):
                    #error_correction_d=prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,"numeric",dataset_name,self.domain_dirty_col,self.domain_clean_col)
                #else:
                   # print("test data path exits!")
                    #error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"numeric")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"numeric")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_type[model][dataset_name]["numeric"].append(output_result)    
                    perfromance_table.loc[-1] = ["numeric",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_type, open(os.path.join(self.result_path, "data_type_score_nm_s.dictionary"), "wb"))
        write_csv_dataset("performance/new/datatype_numeric.csv",perfromance_table)
    def experiment_data_type_alphanumeric(self):
        print("experiment_data_type_alphanumeric is running now")
        perfromance_table = pd.DataFrame(columns = ['data_type', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wikii":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    continue
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    continue
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    continue
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error
                elif dataset_name=="wiki":
                    dirty_data=self.wiki_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.wiki_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.wiki_total_error
                    print(total_error)
 
                test_data_path = os.path.join("datasets", dataset_name,"alphanumeric.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_types(dirty_data,clean_data,"alphanumeric",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                test_data_path = os.path.join("datasets", dataset_name,"alphanumeric_domain.csv") # domain based testing datasets
                if not os.path.exists(test_data_path):
                    error_correction_d=prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,"alphanumeric",dataset_name,self.domain_dirty_col,self.domain_clean_col)
                else:
                    print("test data path exits!")
                    error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f)
                    elif model=="Edit_Distance_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"alphanumeric")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"alphanumeric")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_type[model][dataset_name]["alphanumeric"].append(output_result)    
                    perfromance_table.loc[-1] = ["alphanumeric",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_type, open(os.path.join(self.result_path, "data_type_score_l.dictionary"), "wb"))
        write_csv_dataset("performance/new/datatype_alphanumeric.csv",perfromance_table)
    def experiment_data_error_rv(self):
        print("experiment_data_error_rule_violations is running now")
        perfromance_table = pd.DataFrame(columns = ['data_error', 'dataset','model','precision','recall','f_score']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_set= self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_set= self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_set= self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error
                test_data_path = os.path.join("datasets", dataset_name,"fds.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_error(dirty_data,clean_data,"fds",dataset_name,fds_col)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                test_data_path = os.path.join("datasets", dataset_name,"fds_domain.csv") # domain based testing datasets
                if not os.path.exists(test_data_path):
                    #pass
                    error_correction_d=prepare_domain_testing_datasets_real_world_data_error(dirty_data,clean_data,"fds",dataset_name,self.domain_dirty_col,self.domain_clean_col,fds_col)
                else:
                    print("test data path exits!")
                    error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            p,r,f=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)
                    elif model=="Edit_Distance_D":
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            p,r,f=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f)
                    elif model=="Edit_Distance_D_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)             
                    elif model=="Fasttext_G":
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f=error_correction_fasttext_fds(model,error_correction,fds_col,dirt_data,total_error)
                        print(p,r,f)
                    elif model=="Fasttext_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f=error_correction_fasttext_with_retrain_realworld_fds_new(model,data_for_retrain,dataset_name,fds_set,total_error, dirty_data, clean_data,self.domain_dirty_col,self.domain_clean_col)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)
                    elif model=="Fasttext_D":
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f=error_correction_fasttext_fds(model,error_correction_d,fds_col,dirt_data,total_error)
                        print(p,r,f)
                    elif model=="Fasttext_D_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f=error_correction_fasttext_with_retrain_realworld_fds_new(model,data_for_retrain,dataset_name,fds_set,total_error, dirty_data, clean_data,self.domain_dirty_col,self.domain_clean_col)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)
                    output_result=(p,r,f)
                    self.results_data_error[model][dataset_name]["Rule Violation"].append(output_result)      
                    perfromance_table.loc[-1] = ["Rule Violation",dataset_name, model,p,r,f]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_error, open(os.path.join(self.result_path, "results_data_error_R_modified.dictionary"), "wb"))
        write_csv_dataset("performance/new/rule_violation_modified.csv",perfromance_table)  
    def experiment_data_error_pv(self):
        print("experiment_data_error_pattern_violation is running now")
        perfromance_table = pd.DataFrame(columns = ['data_error', 'dataset','model','precision','recall','f_score']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_set= self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_set= self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_set= self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error
                test_data_path = os.path.join("datasets", dataset_name,"pvs.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_error(dirty_data,clean_data,"pvs",dataset_name,fds_col)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                test_data_path = os.path.join("datasets", dataset_name,"pvs_domain.csv") # domain based testing datasets
                if not os.path.exists(test_data_path):
                    error_correction_d=prepare_domain_testing_datasets_real_world_data_error(dirty_data,clean_data,"pvs",dataset_name,self.domain_dirty_col,self.domain_clean_col,fds_col)
                else:
                    print("test data path exits!")
                    error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            p,r,f=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)
                    elif model=="Edit_Distance_D":
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            p,r,f=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f)
                    elif model=="Edit_Distance_D_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)             
                    elif model=="Fasttext_G":
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            p,r,f=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f)
                    elif model=="Fasttext_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)
                    elif model=="Fasttext_D":
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F                          
                        else:
                            p,r,f=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f)
                    elif model=="Fasttext_D_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f=self.Null_P,self.Null_R,self.Null_F
                        print(p,r,f)
                    output_result=(p,r,f)
                    self.results_data_error[model][dataset_name]["pattern_violations"].append(output_result)  
                    perfromance_table.loc[-1] = ["pattern_violations",dataset_name, model,p,r,f]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_error, open(os.path.join(self.result_path, "data_error_score.dictionary"), "wb"))
        write_csv_dataset("performance/data_error_pattern_violations.csv",perfromance_table) 
    def experiment_uniqueness_0_30(self):
        print("experiment_data_uniqueness_0_30 is running now")
        print("This experiment runs on 0-30 percent unique values")
        perfromance_table = pd.DataFrame(columns = ['uniqueness', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wikii":
                continue
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    continue
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    continue
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    continue
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error 
                elif dataset_name=="wiki":
                    dirty_data=self.wiki_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.wiki_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.wiki_total_error
                    print(total_error)
                test_data_path = os.path.join("datasets", dataset_name,"uniqueness_0_30.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_types(dirty_data,clean_data,"uniqueness_0_30",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                #test_data_path = os.path.join("datasets", dataset_name,"uniqueness_0_30_domain.csv") # domain based testing datasets
                #if not os.path.exists(test_data_path):
                    #error_correction_d=prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,"uniqueness_0_30",dataset_name,self.domain_dirty_col,self.domain_clean_col)
                #else:
                   # print("test data path exits!")
                   # error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"uniqueness_0_30")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"uniqueness_0_30")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_uniqueness[model][dataset_name]["0_30"].append(output_result)       
                    perfromance_table.loc[-1] = ["uniqueness_0_30",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_uniqueness, open(os.path.join(self.result_path, "data_uniqueness_score_f.dictionary"), "wb"))
        write_csv_dataset("performance/new/uniqueness_0_30.csv",perfromance_table)     
    def experiment_uniqueness_31_60(self):
        print("experiment_uniqueness_31_60 is running now")
        print("This experiment runs on 31-60 percent unique values")
        perfromance_table = pd.DataFrame(columns = ['uniqueness', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                continue
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                test_data_path = os.path.join("datasets", dataset_name,"uniqueness_31_60.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_types(dirty_data,clean_data,"uniqueness_31_60",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                test_data_path = os.path.join("datasets", dataset_name,"uniqueness_31_60_domain.csv") # domain based testing datasets
                if not os.path.exists(test_data_path):
                    error_correction_d=prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,"uniqueness_31_60",dataset_name,self.domain_dirty_col,self.domain_clean_col)
                else:
                    print("test data path exits!")
                    error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                       # continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                       # continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                       # continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        continue
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                       # continue
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        #continue
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                       # continue
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"uniqueness_31_60")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"uniqueness_31_60")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_uniqueness[model][dataset_name]["31_60"].append(output_result)     
                    perfromance_table.loc[-1] = ["uniqueness_31_60",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_uniqueness, open(os.path.join(self.result_path, "data_uniqueness_score_s.dictionary"), "wb"))
        write_csv_dataset("performance/new/uniqueness_31_60.csv",perfromance_table)     
    def experiment_uniqueness_61_100(self):
        print("experiment_data_uniqueness_61_100 is running now")
        print("This experiment runs on 61-100 percent unique values")
        perfromance_table = pd.DataFrame(columns = ['uniqueness', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                pass
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                test_data_path = os.path.join("datasets", dataset_name,"uniqueness_61_100.csv")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    error_correction=prepare_testing_datasets_real_world_data_types(dirty_data,clean_data,"uniqueness_61_100",dataset_name)
                else:
                    print("test data path exits!")
                    error_correction=read_csv_dataset(test_data_path)
                test_data_path = os.path.join("datasets", dataset_name,"uniqueness_61_100_domain.csv") # domain based testing datasets
                if not os.path.exists(test_data_path):
                    error_correction_d=prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,"uniqueness_61_100",dataset_name,self.domain_dirty_col,self.domain_clean_col)
                else:
                    print("test data path exits!")
                    error_correction_d=read_csv_dataset(test_data_path)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,cfe                        
                        else:
                            p,r,f,cfe=error_correction_edit_distance(model,error_correction_d,total_error)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        print(model," is running with ", dataset_name)
                        if error_correction.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction,total_error)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        print(model," is running with ", dataset_name)
                        if error_correction_d.empty:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_fasttext(model,error_correction_d,total_error)
                        print(p,r,f)
                    elif model=="Fasttext_D_F":
                        print(model," is running with ", dataset_name)
                        if not error_correction_d.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld(model,error_correction_d,retrain_data,dataset_name,total_error)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,error_correction,"uniqueness_61_100")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if not error_correction.empty:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints(dirty_data,clean_data,dataset_name,retrain_data,total_error,error_correction,"uniqueness_61_100")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_uniqueness[model][dataset_name]["61_100"].append(output_result)      
                    perfromance_table.loc[-1] = ["uniqueness_61_100",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_uniqueness, open(os.path.join(self.result_path, "data_uniqueness_score_l.dictionary"), "wb"))
        write_csv_dataset("performance/new/uniqueness_61_100.csv",perfromance_table)  
    def experiment_data_error_outlier(self):
        print("experiment_data_error_outlier is running now")
        perfromance_table = pd.DataFrame(columns = ['data_error', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                continue
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                test_data_path = os.path.join("datasets", dataset_name, dataset_name+"_OD.pkl")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    detected_error=[]
                else:
                    print("test data path exits!")
                    detected_error=pickle.load(open(os.path.join(test_data_path), "rb"))
                test_data_path_domain = os.path.join("datasets", dataset_name,dataset_name+"_d_OD.pkl") # domain based testing datasets
                if not os.path.exists(test_data_path_domain):
                    detected_error_d=[]
                else:
                    print("test data path exits!")
                    detected_error_d=pickle.load(open(os.path.join(test_data_path_domain), "rb"))
                actual_error_correction_dict=get_dataframes_difference(dirty_data,clean_data)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints_pkl(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints_pkl(dirty_data,clean_data,dataset_name,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_error[model][dataset_name]["Outlier"].append(output_result)      
                    perfromance_table.loc[-1] = ["Outlier",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_error, open(os.path.join(self.result_path, "results_data_error_O.dictionary"), "wb"))
        write_csv_dataset("performance/new/Outlier.csv",perfromance_table)  
    def experiment_data_error_pattern_violation(self):
        print("experiment_data_Pattern Violation is running now")
        perfromance_table = pd.DataFrame(columns = ['data_error', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                continue
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                test_data_path = os.path.join("datasets", dataset_name, dataset_name+"_PVD.pkl")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    detected_error=[]
                else:
                    print("test data path exits!")
                    detected_error=pickle.load(open(os.path.join(test_data_path), "rb"))
                test_data_path_domain = os.path.join("datasets", dataset_name,dataset_name+"_d_PVD.pkl") # domain based testing datasets
                if not os.path.exists(test_data_path_domain):
                    detected_error_d=[]
                else:
                    print("test data path exits!")
                    detected_error_d=pickle.load(open(os.path.join(test_data_path_domain), "rb"))
                actual_error_correction_dict=get_dataframes_difference(dirty_data,clean_data)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                        
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints_pkl(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints_pkl(dirty_data,clean_data,dataset_name,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_error[model][dataset_name]["Pattern Violation"].append(output_result)      
                    perfromance_table.loc[-1] = ["Pattern Violation",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_error, open(os.path.join(self.result_path, "results_data_error_P.dictionary"), "wb"))
        write_csv_dataset("performance/new/pattern_violation.csv",perfromance_table)  
    def experiment_data_error_rule_violation(self):
        print("experiment_data_Rule Violation is running now")
        perfromance_table = pd.DataFrame(columns = ['data_error', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                continue
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                test_data_path = os.path.join("datasets", dataset_name, dataset_name+"_RVD.pkl")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    detected_error=[]
                else:
                    print("test data path exits!")
                    detected_error=pickle.load(open(os.path.join(test_data_path), "rb"))
                test_data_path_domain = os.path.join("datasets", dataset_name,dataset_name+"_d_RVD.pkl") # domain based testing datasets
                if not os.path.exists(test_data_path_domain):
                    detected_error_d=[]
                else:
                    print("test data path exits!")
                    detected_error_d=pickle.load(open(os.path.join(test_data_path_domain), "rb"))
                actual_error_correction_dict=get_dataframes_difference(dirty_data,clean_data)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        #continue
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                       #continue
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                          
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        #continue
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        #continue
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        #continue
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints_pkl(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints_pkl(dirty_data,clean_data,dataset_name,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_error[model][dataset_name]["Rule Violation"].append(output_result)      
                    perfromance_table.loc[-1] = ["Rule Violation",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_error, open(os.path.join(self.result_path, "results_data_error_R.dictionary"), "wb"))
        write_csv_dataset("performance/new/rule_violation.csv",perfromance_table)
    def experiment_data_error_knowledge_base_violation(self):
        print("experiment_data_Knowledge Base Violation is running now")
        perfromance_table = pd.DataFrame(columns = ['data_error', 'dataset','model','precision','recall','f_score','cfe']) #want_to_clean 1 or 0
        for dataset_name in self.DATASETS:
            if dataset_name=="wiki":
                continue
            else:
                dataset_dictionary = {
                    "name": dataset_name,
                    "dirty_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "clean.csv"))}
                if dataset_name=="tax":
                    fds_col_set=self.tax_fds
                    fds_col=self.tax_fds_att
                    dirty_data=self.tax_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.tax_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.tax_total_error     
                elif dataset_name=="hospital":
                    fds_col_set=self.hos_fds
                    fds_col=self.hos_fds_att
                    dirty_data=self.hos_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.hos_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.hos_total_error    
                elif dataset_name=="flights":
                    fds_col_set=self.flight_fds
                    fds_col=self.flight_fds_att
                    dirty_data=self.flights_dirty_df
                    dirty_data_col= dirty_data.columns.values  
                    clean_data= self.flights_clean_df
                    clean_data_col=clean_data.columns.values   
                    total_error= self.flights_total_error  
                test_data_path = os.path.join("datasets", dataset_name, dataset_name+"_KBVD.pkl")
                if not os.path.exists(test_data_path):
                    print("test data path not exits!")
                    detected_error=[]
                else:
                    print("test data path exits!")
                    detected_error=pickle.load(open(os.path.join(test_data_path), "rb"))
                test_data_path_domain = os.path.join("datasets", dataset_name,dataset_name+"_d_KBVD.pkl") # domain based testing datasets
                if not os.path.exists(test_data_path_domain):
                    detected_error_d=[]
                else:
                    print("test data path exits!")
                    detected_error_d=pickle.load(open(os.path.join(test_data_path_domain), "rb"))
                actual_error_correction_dict=get_dataframes_difference(dirty_data,clean_data)
                for model in self.MODELS:
                    if model=="Edit_Distance_G":
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_G_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D":
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            p,r,f,cfe=error_correction_edit_distance_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Edit_Distance_D_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_edit_distance_retrain_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)             
                    elif model=="Fasttext_G":
                        print(model," is running with ", dataset_name)
                        if not detected_error:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_G_F":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D":
                        print(model," is running with ", dataset_name)
                        if not detected_error_d:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0                         
                        else:
                            dirt_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_pkl(model,detected_error_d,total_error,actual_error_correction_dict,dirty_data)
                        print(p,r,f,cfe)
                    elif model=="Fasttext_D_F":
                        print(model," is running with ", dataset_name)
                        if  detected_error_d:
                            retrain_data=prepare_dataset_for_retrain_realworld_domain(clean_data,dirty_data, self.domain_dirty_col)
                            p,r,f,cfe=error_correction_fasttext_with_retrain_realworld_pkl(model,detected_error_d,retrain_data,dataset_name,total_error,actual_error_correction_dict,dirty_data)
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_DC":
                        print(model," is running with ", dataset_name)
                        if detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_with_constraints_pkl(dirty_data,clean_data,dataset_name, fds_col_set,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)
                    elif model=="Fasttext_Supervised_WDC":
                        print(model," is running with ", dataset_name)
                        if  detected_error:
                            retrain_data=prepare_dataset_for_retrain_realworld(clean_data,dirty_data)
                            p,r,f,cfe=error_correction_fasttext_supervised_without_constraints_pkl(dirty_data,clean_data,dataset_name,retrain_data,total_error,detected_error,"null")
                        else:
                            p,r,f,cfe=self.Null_P,self.Null_R,self.Null_F,0
                        print(p,r,f,cfe)               
                    output_result=(p,r,f,cfe)
                    self.results_data_error[model][dataset_name]["Knowledge Base Violation"].append(output_result)      
                    perfromance_table.loc[-1] = ["Knowledge Base Violation",dataset_name, model,p,r,f,cfe]
                    perfromance_table.index = perfromance_table.index + 1  # shifting index
                    perfromance_table = perfromance_table.sort_index()
        pickle.dump(self.results_data_error, open(os.path.join(self.result_path, "results_data_error_K_L.dictionary"), "wb"))
        write_csv_dataset("performance/new/kb_violation.csv",perfromance_table)
if __name__ == "__main__":
    app = Experiments()
    #app.experiment_data_type_textual()
    #app.experiment_data_type_numeric()
    #app.experiment_data_type_alphanumeric()
    #print("###################data uniqueness ##############################")
    #app.experiment_uniqueness_0_30()
    #app.experiment_uniqueness_31_60()
    #app.experiment_uniqueness_61_100()
    #print("###################uniqueness##############################")
    #print("Start Data Error types")
    print("YEs, ...........................")
   # app.experiment_data_error_outlier()
    #app.experiment_data_error_pattern_violation()
    #app.experiment_data_error_rule_violation()
    #app.experiment_data_error_knowledge_base_violation()
    #app.experiment_data_error_rv()
    app.experiment_general()
    print("##################### Finished data error##############")
   
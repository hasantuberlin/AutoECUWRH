# library for fasttext
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
########################################
# This libraray is for edit distance model
from collections import Counter
from aion.util.spell_check import SpellCorrector
from fuzzywuzzy import fuzz
aion_dir = 'aion/'
sys.path.insert(0, aion_dir)
def add_aion(curr_path=None):
    if curr_path is None:
        dir_path = os.getcwd()
        target_path = os.path.dirname(os.path.dirname(dir_path))
        if target_path not in sys.path:
            #print('Added %s into sys.path.' % (target_path))
            sys.path.insert(0, target_path)
            
add_aion()
#############################################
# Library for cross validation
#scikit learn library
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
#############################################
def error_correction_edit_distance(model_type, error_correction,total_error):
        #total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Edit_Distance_G":
                with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
            if model_type=="Edit_Distance_D":
                with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file: #edit_distance_domain_location
                    model_edit_distance = pickle.load(pickle_file)
        except Exception as e:
            print('Exception: ',str(e))   
        for error_value, actual_value,want_to_clean in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean']):
            try:    
                #total_error=total_error+1
                error_value=str(error_value)
                want_to_clean=str(want_to_clean)
                if want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1:
                    total_error_to_repaired=total_error_to_repaired+1 
                    error_value=str(error_value)
                    error_value=error_value.strip()   
                    first=model_edit_distance.correction(error_value)
                    first=str(first)
                    actual_value=str(actual_value)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print("Exception : ", str(e))
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def error_correction_edit_distance_retrain(model_type,error_correction,data_for_retrain,dataset_name,total_error):
        #total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Edit_Distance_G_F":
                with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
            if model_type=="Edit_Distance_D_F":
                with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
        except Exception as e:
            print('Model Error: ',str(e))
        if dataset_name=="wikii":
            train_data_rows=data_for_retrain           
        else:
            train_data_rows=[]   
            data_for_retrain=data_for_retrain.values.tolist()
            for row in data_for_retrain:
                row = list(map(str, row))
                row=list(filter(None, row))
                train_data_rows.extend(row)
        if train_data_rows:
            dict1=model_edit_distance.dictionary
            general_corpus = [str(s) for s in train_data_rows]
            corpus = Counter(general_corpus)
            corpus.update(dict1)
            model_edit_distance = SpellCorrector(dictionary=corpus, verbose=1)
        total_p=0
        total_error_to_repaired=0
        for error_value, actual_value, want_to_clean in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean']):
            total_p=total_p+1
            try:
                #total_error=total_error+1
                error_value=str(error_value)
                want_to_clean=str(want_to_clean)
                if  want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1 and error_value!="":
                    total_error_to_repaired=total_error_to_repaired+1
                    error_value=str(error_value)
                    error_value=error_value.strip()  
                    first,prob=model_edit_distance.correction(error_value)
                    print("###################")
                    first=str(first)
                    actual_value=str(actual_value)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        total_repaired=total_repaired+1
                    elif first==error_value:
                        print(prob) #updated edit distnce 4_12
                        total_error_to_repaired=total_error_to_repaired-1

            except Exception as e:
                print('Exception: ', str(e))
                continue
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def error_correction_fasttext(model_type,error_correction,total_error): #model=edit distance, fasttext, dataset_type=wiki_realword
        #total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_G":
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_D":
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        for error_value, actual_value,want_to_clean in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean']):
            try:    
                want_to_clean=str(want_to_clean)
                error_value=str(error_value)
                if want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1:
                    error_value=str(error_value)
                    error_value=error_value.strip()
                    total_error_to_repaired=total_error_to_repaired+1    
                    similar_value=model_fasttext.most_similar(error_value)
                    first,b=similar_value[0]
                    actual_value=str(actual_value)
                    first=str(first)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def error_correction_fasttext_with_retrain_realworld(model_type,error_correction,data_for_retrain,dataset_name,total_error):
       # total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_G_F":
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_D_F":
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))     
        train_data_rows=[]
        try:         
            data_for_retrain=data_for_retrain.values.tolist()
            for row in data_for_retrain:
                row = list(map(str, row))
                row=list(filter(None, row))
                train_data_rows.append(row)
            #train_data_rows=list(filter(None, train_data_rows)) # each value as word not each tuple
            if train_data_rows:
                model_fasttext.build_vocab(train_data_rows, update=True)
                model_fasttext.train(sentences=train_data_rows, total_examples = len(train_data_rows), epochs=5)

               # model = fasttext.train_unsupervised(train_data_rows, epoch=1, lr=0.5)
                #print("###########################################") #updated 28.11.2020
                    
        except Exception as e:
            print("Exception from spell model : ", str(e))
        for error_value, actual_value,want_to_clean in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean']):
            try:    
                #total_error=total_error+1
                error_value=str(error_value)
                want_to_clean=str(want_to_clean)
                if want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000:
                    error_value=str(error_value)
                    error_value=error_value.strip()
                    total_error_to_repaired=total_error_to_repaired+1
                    similar_value=model_fasttext.most_similar(error_value)  
                    #similar_value=model_fasttext.most_similar(error_value)
                    #print(actual_value,similar_value)
                    first,b=similar_value[0]
                    actual_value=str(actual_value)  
                    first=str(first)               
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def error_correction_fasttext_fds(model_type,error_correction,fds_set,dirty_data,total_error): #model=edit distance, fasttext, dataset_type=wiki_realword
        #total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_G":
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_D":
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        for error_value, actual_value,want_to_clean,index in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean'],error_correction['index']):
            try:
                dirty_row=[]
                if fds_set:
                    for fds in fds_set:
                        dirty_row.append(str(dirty_data.at[index, fds]))
                if dirty_row:
                    dirty_row = list(map(str, dirty_row))
                    dirty_row=list(filter(None, dirty_row))  
                #total_error=total_error+1
                want_to_clean=str(want_to_clean)
                if want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000:
                    total_error_to_repaired=total_error_to_repaired+1
                    if fds_set and dirty_row:
                        similar_value=model_fasttext.most_similar(positive=dirty_row, negative=[error_value])
                    else:
                        similar_value=model_fasttext.most_similar(error_value)   
                    #similar_value=model_fasttext.most_similar(error_value)
                    first,b=similar_value[0]
                    #first=first.lower()
                    #actual_value=actual_value.lower()
                    first=first.strip()
                    actual_value=actual_value.strip()
                    #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def error_correction_fasttext_with_retrain_realworld_fds(model_type,error_correction,data_for_retrain,dataset_name,fds_set,total_error):
        #total_error=0
        total_error_to_repaired=0
        total_repaired=0
        dirty_data=data_for_retrain
        try:
            if model_type=="Fasttext_G_F":
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_D_F":
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))     
        train_data_rows=[]
        try:         
            data_for_retrain=data_for_retrain.values.tolist()
            for row in data_for_retrain:
                row = list(map(str, row))
                row=list(filter(None, row))
                train_data_rows.append(row)
            if train_data_rows:
                if train_data_rows:
                    model_fasttext.build_vocab(train_data_rows, update=True)
                    model_fasttext.train(sentences=train_data_rows, total_examples = len(train_data_rows), epochs=5)
        except Exception as e:
            print("Exception from spell model : ", str(e))
        for error_value, actual_value,want_to_clean,index in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean'],error_correction['index']):
            try:
                dirty_row=[]
                if fds_set:
                    for fds in fds_set:
                        dirty_row.append(str(dirty_data.at[index, fds]))
                if dirty_row:
                    dirty_row = list(map(str, dirty_row))
                    dirty_row=list(filter(None, dirty_row))  
                #total_error=total_error+1
                want_to_clean=str(want_to_clean)
                if want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000:
                    total_error_to_repaired=total_error_to_repaired+1
                    if fds_set and dirty_row:
                        similar_value=model_fasttext.most_similar(positive=dirty_row, negative=[error_value])
                    else:
                        similar_value=model_fasttext.most_similar(error_value)
                    first,b=similar_value[0]
                    actual_value=str(actual_value)               
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def evaluate_model (total_error, total_error_to_repair, total_correction):
        if total_error_to_repair==0:
            precision=0.00
        else:
            precision=total_correction/total_error_to_repair
            precision=round(precision,2)
        if total_error==0:
            recall=0.00
        else:
            recall=total_correction/total_error
            recall=round(recall,2)
        if (precision+recall)==0:
            f_score=0.00
        else:
            f_score=(2 * precision * recall) / (precision + recall)
            f_score=int((f_score * 100)) / 100.0
            #f_score=round(f_score,2)     
        return precision, recall,f_score
def error_correction_edit_distance_pkl(model_type, error_detection,total_error,actual_error_correction_dict,dirty_data):
    #total_error=0
    total_error_to_repaired=0
    total_repaired=0
    try:
        if model_type=="Edit_Distance_G":
            with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                model_edit_distance = pickle.load(pickle_file)
        if model_type=="Edit_Distance_D":
            with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file: #edit_distance_domain_location
                model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Exception: ',str(e))   
    for tuple_pair in error_detection:
        i,j=tuple_pair
        error_value=dirty_data.iloc[i][j]
        actual_value=actual_error_correction_dict[tuple_pair]

        try:    
            #total_error=total_error+1
            error_value=str(error_value)
            if len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1:
                total_error_to_repaired=total_error_to_repaired+1 
                error_value=str(error_value)
                error_value=error_value.strip()   
                first=model_edit_distance.correction(error_value)
                first=str(first)
                actual_value=str(actual_value)
                first=first.strip()
                actual_value=actual_value.strip()
                if first==actual_value:
                    total_repaired=total_repaired+1
        except Exception as e:
            print("Exception : ", str(e))
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_edit_distance_retrain_pkl(model_type,error_detection,data_for_retrain,dataset_name,total_error,actual_error_correction_dict,dirty_data):
    #total_error=0
    total_error_to_repaired=0
    total_repaired=0
    try:
        if model_type=="Edit_Distance_G_F":
            with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                model_edit_distance = pickle.load(pickle_file)
        if model_type=="Edit_Distance_D_F":
            with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file:
                model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))
    if dataset_name=="wiki":
        train_data_rows=data_for_retrain           
    else:
        train_data_rows=[]   
        data_for_retrain=data_for_retrain.values.tolist()
        for row in data_for_retrain:
            row = list(map(str, row))
            row=list(filter(None, row))
            train_data_rows.extend(row)
    if train_data_rows:
        dict1=model_edit_distance.dictionary
        general_corpus = [str(s) for s in train_data_rows]
        corpus = Counter(general_corpus)
        corpus.update(dict1)
        model_edit_distance = SpellCorrector(dictionary=corpus, verbose=1)
    total_p=0
    total_error_to_repaired=0
    for tuple_pair in error_detection:
        i,j=tuple_pair
        error_value=dirty_data.iloc[i][j]
        actual_value=actual_error_correction_dict[tuple_pair]
        try:
            if len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1 and  error_value!="":
                total_error_to_repaired=total_error_to_repaired+1
                error_value=str(error_value)
                error_value=error_value.strip()  
                first=model_edit_distance.correction(error_value)
                first=str(first)
                actual_value=str(actual_value)
                first=first.strip()
                actual_value=actual_value.strip()
                if first==actual_value:
                    total_repaired=total_repaired+1
                elif first==error_value: #updated edit distnce 4_12
                    total_error_to_repaired=total_error_to_repaired-1

        except Exception as e:
            print('Exception: ', str(e))
            continue
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_fasttext_pkl(model_type,error_detection,total_error,actual_error_correction_dict,dirty_data): #model=edit distance, fasttext, dataset_type=wiki_realword
    #total_error=0
    total_error_to_repaired=0
    total_repaired=0
    try:
        if model_type=="Fasttext_G":
            model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
        if model_type=="Fasttext_D":
            model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
    except Exception as e:
        print('Model Error: ',str(e))
    for tuple_pair in error_detection:
        i,j=tuple_pair
        error_value=dirty_data.iloc[i][j]
        actual_value=actual_error_correction_dict[tuple_pair]
        try:    
            if len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1:
                error_value=str(error_value)
                error_value=error_value.strip()
                total_error_to_repaired=total_error_to_repaired+1    
                similar_value=model_fasttext.most_similar(error_value)
                first,b=similar_value[0]
                actual_value=str(actual_value)
                first=str(first)
                first=first.strip()
                actual_value=actual_value.strip()
                if first==actual_value:
                    total_repaired=total_repaired+1
        except Exception as e:
            print('Error correction model: ',str(e))
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_fasttext_with_retrain_realworld_pkl(model_type,error_detection,data_for_retrain,dataset_name,total_error,actual_error_correction_dict,dirty_data):
    # total_error=0
    total_error_to_repaired=0
    total_repaired=0
    try:
        if model_type=="Fasttext_G_F":
            model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
        if model_type=="Fasttext_D_F":
            model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
    except Exception as e:
        print('Model Error: ',str(e))     
    train_data_rows=[]
    try:         
        data_for_retrain=data_for_retrain.values.tolist()
        for row in data_for_retrain:
            row = list(map(str, row))
            row=list(filter(None, row))
            train_data_rows.append(row)
        if train_data_rows:
            if train_data_rows:
                model_fasttext.build_vocab(train_data_rows, update=True)
                model_fasttext.train(sentences=train_data_rows, total_examples = len(train_data_rows), epochs=5)
    except Exception as e:
        print("Exception from spell model : ", str(e))
    for tuple_pair in error_detection:
        i,j=tuple_pair
        error_value=dirty_data.iloc[i][j]
        actual_value=actual_error_correction_dict[tuple_pair]

        try:    
            if len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1:
                error_value=str(error_value)
                error_value=error_value.strip()
                total_error_to_repaired=total_error_to_repaired+1
                similar_value=model_fasttext.most_similar(error_value) 
                first,b=similar_value[0]
                actual_value=str(actual_value)  
                first=str(first)               
                first=first.strip()
                actual_value=actual_value.strip()
                if first==actual_value:
                    total_repaired=total_repaired+1
        except Exception as e:
            print('Error correction model: ',str(e))
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_fasttext_with_retrain_realworld_fds_new(model_type,data_for_retrain,dataset_name,fds_set,total_error, dirty_data, clean_data,domain_dirty_col,domain_clean_col):
        #total_error=0
        total_error_to_repaired=0
        total_repaired=0
        dirty_data=data_for_retrain
        try:
            if model_type=="Fasttext_G_F":
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_D_F":
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        err_fds_con=[]
        for fds in  fds_set:
            data_for_retrain=data_for_retrain[fds]
            train_data_rows=[]
            try:         
                data_for_retrain=data_for_retrain.values.tolist()
                for row in data_for_retrain:
                    row = list(map(str, row))
                    row=list(filter(None, row))
                    train_data_rows.append(row)
                if train_data_rows:
                    if train_data_rows:
                        model_fasttext = FastText(train_data_rows, min_count=1, workers=8, iter=500, window=len(train_data_rows[0]), sg=1)
            except Exception as e:
                print("Exception from spell model : ", str(e))
            err_fds=[]
            for col_fds in fds:
                if col_fds in err_fds_con:
                    continue
                else:
                    err_fds_con.append(col_fds)
                    err_fds.append(col_fds)
            if model_type=="Fasttext_G_F":
                error_correction=prepare_testing_datasets_real_world_data_error(dirty_data,clean_data,"fds",dataset_name,err_fds)
            else:
                error_correction=prepare_domain_testing_datasets_real_world_data_error(dirty_data,clean_data,"fds",dataset_name,domain_dirty_col,domain_clean_col,err_fds)
            for error_value, actual_value,want_to_clean,index in zip(error_correction['error'],error_correction['actual'],error_correction['want_to_clean'],error_correction['index']):
                try:
                    dirty_row=[]
                    if fds:
                        for fds_col in fds:
                            dirty_row.append(str(dirty_data.at[index, fds_col]))
                    if dirty_row:
                        dirty_row = list(map(str, dirty_row))
                        dirty_row=list(filter(None, dirty_row))
                        if error_value:
                            dirty_row.remove(error_value)
                    want_to_clean=str(want_to_clean)
                    if want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000:
                        total_error_to_repaired=total_error_to_repaired+1
                        if fds_set and dirty_row:
                            similar_value=model_fasttext.most_similar(positive=dirty_row, negative=[error_value])
                        else:
                            similar_value=model_fasttext.most_similar(error_value)
                        first,b=similar_value[0]
                        actual_value=str(actual_value)               
                        first=first.strip()
                        actual_value=actual_value.strip()
                        if first==actual_value:
                            total_repaired=total_repaired+1
                except Exception as e:
                    print('Error correction model: ',str(e))
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
        return p,r,f,total_repaired
def error_correction_fasttext_supervised_with_constraints(dataframe,clean_dataframe,dataset_name, fds_col_set,retrain_data,total_error,error_correction_specific,data_type):
    total_error_to_repaired=0
    total_repaired=0
    dirty_data=dataframe
    dirty_data_col=dirty_data.columns.values
    attribute=[]
    if data_type=="general":
        attribute=list(dirty_data_col)
    if data_type=="textual":
        attribute=textual_attributes(dirty_data,dirty_data_col)
    elif data_type=="alphanumeric":
        attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
    elif data_type=="numeric":
        attribute=numeric_attributes(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_0_30":
        attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)   
    elif data_type=="uniqueness_31_60":
        attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_61_100":
        attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
    corrected_col=[]
    for fds_col in fds_col_set:
        feature_data=fds_col
        for fds in fds_col:
            if fds==fds_col[0]:# it will predict the second value
                continue
            if attribute and fds not in attribute:
                continue
            if corrected_col:
                if fds in corrected_col:
                    continue
                else:
                    corrected_col.append(fds)
            data_for_retrain=retrain_data[feature_data]
            predict_data_rows=retrain_data[feature_data]
            data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
            data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
            df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
            df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
            model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
            error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
            index_list=error_correction_specific['index'].to_list()
            retrain_data_p=retrain_data[feature_data]
            for  actual_value,want_to_clean,index, specific_index in zip(error_correction['actual'],error_correction_specific['want_to_clean'],error_correction['index'],error_correction_specific['index']):
                index1=str(index)
                if index1 in index_list: #total_error_to_repaired<5000
                    predict_error=retrain_data_p.iloc[index].tolist()
                    row = list(map(str, predict_error))
                    row=list(filter(None, row))
                    retrain_data_col=retrain_data_p.columns.values
                    if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col :
                        row=row[1:]
                    if row:
                        if len(row)==1:
                            row=predict_error
                        else:                   
                            row = ' '.join([str(r) for r in row])
                        repaired=model_flight.predict(row,k=1,threshold=0.90)
                        try:
                            if repaired[0][0]:
                                total_error_to_repaired=total_error_to_repaired+1   
                                k=repaired[0][0].replace('__label__','')
                                repaired_k=k.replace('_',' ')
                                repaired_k=repaired_k.strip()
                                actual_value=actual_value.strip()
                                if repaired_k==actual_value:
                                    total_repaired=total_repaired+1
                        except:
                            continue
                else:
                    continue
        print(total_error,total_error_to_repaired,total_repaired )
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_fasttext_supervised_without_constraints(dataframe,clean_dataframe,dataset_name,retrain_data,total_error,error_correction_specific,data_type):
    total_error_to_repaired=0
    total_repaired=0
    dirty_data=dataframe
    dirty_data_col=dirty_data.columns.values
    attribute=[]
    if data_type=="general":
        attribute=list(dirty_data_col)
    if data_type=="textual":
        attribute=textual_attributes(dirty_data,dirty_data_col)
    elif data_type=="alphanumeric":
        attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
    elif data_type=="numeric":
        attribute=numeric_attributes(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_0_30":
        attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)   
    elif data_type=="uniqueness_31_60":
        attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_61_100":
        attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
    columns=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index" :
        columns.remove(columns[0])
    for fds in columns:
        if attribute and fds not in attribute:
            continue
        feature_data=columns
        data_for_retrain=retrain_data[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
        index_list=error_correction_specific['index'].to_list()
        #print(index_list)
        error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
        for actual_value,want_to_clean,index, specific_index in zip(error_correction['actual'],error_correction_specific['want_to_clean'],error_correction['index'],error_correction_specific['index']):
            index1=str(index)
            if index1 in index_list: #total_error_to_repaired<5000
                predict_error=retrain_data.iloc[index]
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=retrain_data.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col :
                    row=row[1:]
                if row:
                    row = ' '.join([str(r) for r in row])            
                try:
                    repaired=model_flight.predict(row,k=1,threshold=0.90)
                    if repaired[0][0]:
                        total_error_to_repaired=total_error_to_repaired+1   
                        k=repaired[0][0].replace('__label__','')
                        repaired_k=k.replace('_',' ')
                        repaired_k=repaired_k.strip()
                        actual_value=actual_value.strip()
                        if repaired_k==actual_value:
                            total_repaired=total_repaired+1
                except:
                    continue
            else:
                continue
        print(total_error,total_error_to_repaired,total_repaired )
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired

def error_correction_fasttext_supervised_without_constraints_pkl(dataframe,clean_dataframe,dataset_name,retrain_data,total_error,error_correction_specific,data_type):
    index_list=[]
    for tuple_pair in error_correction_specific:
        i,j=tuple_pair
        index_list.append(i)
    total_error_to_repaired=0
    total_repaired=0
    dirty_data=dataframe
    dirty_data_col=dirty_data.columns.values
    attribute=[]
    if data_type=="textual":
        attribute=textual_attributes(dirty_data,dirty_data_col)
    elif data_type=="alphanumeric":
        attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
    elif data_type=="numeric":
        attribute=numeric_attributes(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_0_30":
        attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)   
    elif data_type=="uniqueness_31_60":
        attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_61_100":
        attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
    columns=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index" :
        columns.remove(columns[0])
    for fds in columns:
        if attribute and fds not in attribute:
            continue
        feature_data=columns
        data_for_retrain=retrain_data[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
        #index_list=error_correction_specific['index'].to_list()
        #print(index_list)
        error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
        for actual_value,index in zip(error_correction['actual'],error_correction['index']):
            int_index=index
            index1=str(index)
            if index1 in index_list or int_index in index_list and total_error_to_repaired<5000 :
                predict_error=retrain_data.iloc[index]
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=retrain_data.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col :
                    row=row[1:]
                if row:
                    row = ' '.join([str(r) for r in row])
                repaired=model_flight.predict(row,k=1,threshold=0.90)
                #print(row,repaired)
                try:
                    if repaired[0][0]:
                        total_error_to_repaired=total_error_to_repaired+1   
                        k=repaired[0][0].replace('__label__','')
                        repaired_k=k.replace('_',' ')
                        repaired_k=repaired_k.strip()
                        actual_value=actual_value.strip()
                        if repaired_k==actual_value:
                            total_repaired=total_repaired+1
                except:
                    continue
            else:
                continue
        print(total_error,total_error_to_repaired,total_repaired )
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_fasttext_supervised_with_constraints_pkl(dataframe,clean_dataframe,dataset_name, fds_col_set,retrain_data,total_error,error_correction_specific,data_type):
    index_list=[]
    for tuple_pair in error_correction_specific:
        i,j=tuple_pair
        index_list.append(i)
    total_error_to_repaired=0
    total_repaired=0
    dirty_data=dataframe
    dirty_data_col=dirty_data.columns.values
    attribute=[]
    if data_type=="textual":
        attribute=textual_attributes(dirty_data,dirty_data_col)
    elif data_type=="alphanumeric":
        attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
    elif data_type=="numeric":
        attribute=numeric_attributes(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_0_30":
        attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)   
    elif data_type=="uniqueness_31_60":
        attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_61_100":
        attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
    corrected_col=[]
    for fds_col in fds_col_set:
        feature_data=fds_col
        for fds in fds_col:
            if attribute and fds not in attribute:
                continue
            if corrected_col:
                if fds in corrected_col:
                    continue
                else:
                    corrected_col.append(fds)
            data_for_retrain=retrain_data[feature_data]
            predict_data_rows=retrain_data[feature_data]
            data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
            data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
            df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
            df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
            model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
            error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
            #index_list=error_correction_specific['index'].to_list()
            #print(index_list)
            retrain_data_p=retrain_data[feature_data]
            for  actual_value,index in zip(error_correction['actual'],error_correction['index']):
                int_index=index
                index1=str(index)
                if index1 in index_list or int_index in index_list and total_error_to_repaired<5000:
                    predict_error=retrain_data_p.iloc[index].tolist()
                    row = list(map(str, predict_error))
                    row=list(filter(None, row))
                    retrain_data_col=retrain_data_p.columns.values
                    if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col :
                        row=row[1:]
                    if row:
                        if len(row)==1:
                            row=predict_error
                        else:                   
                            row = ' '.join([str(r) for r in row])
                        repaired=model_flight.predict(row,k=1,threshold=0.90)
                        try:
                            if repaired[0][0]:
                                total_error_to_repaired=total_error_to_repaired+1   
                                k=repaired[0][0].replace('__label__','')
                                repaired_k=k.replace('_',' ')
                                repaired_k=repaired_k.strip()
                                actual_value=actual_value.strip()
                                if repaired_k==actual_value:
                                    total_repaired=total_repaired+1
                        except:
                            continue
                else:
                    continue
        print(total_error,total_error_to_repaired,total_repaired )
    print(total_error,total_error_to_repaired,total_repaired )
    if total_error_to_repaired>0:
        p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_fasttext_supervised_without_constraints_threshold(dataframe,clean_dataframe,dataset_name,retrain_data,total_error,error_correction_specific,data_type):
    total_error_to_repaired=0
    total_repaired=0
    threshold_analysis = pd.DataFrame(columns = ['precision', 'recall','TP','FP','threshold'])
    threshold_range=[x * 0.1 for x in range(1, 10)]
    threshold_range.append(0.99)
    dirty_data=dataframe
    dirty_data_col=dirty_data.columns.values
    attribute=[]
    if data_type=="general":
        attribute=list(dirty_data_col)
    if data_type=="textual":
        attribute=textual_attributes(dirty_data,dirty_data_col)
    elif data_type=="alphanumeric":
        attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
    elif data_type=="numeric":
        attribute=numeric_attributes(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_0_30":
        attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)   
    elif data_type=="uniqueness_31_60":
        attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_61_100":
        attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
    columns=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index" :
        columns.remove(columns[0])
    for threshold_v in threshold_range:
        print(dataset_name, threshold_v)
        total_error_to_repaired=0
        total_repaired=0
        corrected_col=[]
        for fds in columns:
            if attribute and fds not in attribute:
                continue
            feature_data=columns
            data_for_retrain=retrain_data[feature_data]
            data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
            data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
            df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
            df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
            model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
            index_list=error_correction_specific['index'].to_list()
            #print(index_list)
            error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
            for actual_value,want_to_clean,index, specific_index in zip(error_correction['actual'],error_correction_specific['want_to_clean'],error_correction['index'],error_correction_specific['index']):
                index1=str(index)
                if index1 in index_list and total_error_to_repaired<5000:
                    predict_error=retrain_data.iloc[index]
                    row = list(map(str, predict_error))
                    row=list(filter(None, row))
                    retrain_data_col=retrain_data.columns.values
                    if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col :
                        row=row[1:]
                    if row:
                        row = ' '.join([str(r) for r in row])            
                    try:
                        repaired=model_flight.predict(row,k=1,threshold=threshold_v)
                        if repaired[0][0]:
                            total_error_to_repaired=total_error_to_repaired+1   
                            k=repaired[0][0].replace('__label__','')
                            repaired_k=k.replace('_',' ')
                            repaired_k=repaired_k.strip()
                            actual_value=actual_value.strip()
                            if repaired_k==actual_value:
                                total_repaired=total_repaired+1
                    except:
                        continue
                else:
                    continue
            #print(total_error,total_error_to_repaired,total_repaired )
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
            fp=total_error_to_repaired-total_repaired
            threshold_analysis.loc[-1] = [p, r,total_repaired,fp,threshold_v]
            threshold_analysis.index = threshold_analysis.index + 1  # shifting index
            threshold_analysis = threshold_analysis.sort_index()
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
    path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "WDC_threshold.csv"))
    write_csv_dataset(path,threshold_analysis)
    return p,r,f,total_repaired
def error_correction_fasttext_supervised_with_constraints_threshold(dataframe,clean_dataframe,dataset_name, fds_col_set,retrain_data,total_error,error_correction_specific,data_type):
    dirty_data=dataframe
    threshold_analysis = pd.DataFrame(columns = ['precision', 'recall','TP','FP','threshold'])
    threshold_range=[x * 0.1 for x in range(1, 10)]
    threshold_range.append(0.99)
    dirty_data_col=dirty_data.columns.values
    attribute=[]
    if data_type=="general":
        attribute=list(dirty_data_col)
    if data_type=="textual":
        attribute=textual_attributes(dirty_data,dirty_data_col)
    elif data_type=="alphanumeric":
        attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
    elif data_type=="numeric":
        attribute=numeric_attributes(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_0_30":
        attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)   
    elif data_type=="uniqueness_31_60":
        attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
    elif data_type=="uniqueness_61_100":
        attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
    
    for threshold_v in threshold_range:
        print(dataset_name, threshold_v)
        total_error_to_repaired=0
        total_repaired=0
        corrected_col=[]
        for fds_col in fds_col_set:
            feature_data=fds_col
            for fds in fds_col:
                if fds==fds_col[0]:# it will predict the second value
                    continue
                if attribute and fds not in attribute:
                    continue
                if corrected_col:
                    if fds in corrected_col:
                        continue
                    else:
                        corrected_col.append(fds)
                data_for_retrain=retrain_data[feature_data]
                predict_data_rows=retrain_data[feature_data]
                data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
                data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
                df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
                df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
                model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
                error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
                index_list=error_correction_specific['index'].to_list()
                retrain_data_p=retrain_data[feature_data]
                for  actual_value,want_to_clean,index, specific_index in zip(error_correction['actual'],error_correction_specific['want_to_clean'],error_correction['index'],error_correction_specific['index']):
                    index1=str(index)
                    if index1 in index_list and total_error_to_repaired<5000:
                        predict_error=retrain_data_p.iloc[index].tolist()
                        row = list(map(str, predict_error))
                        row=list(filter(None, row))
                        retrain_data_col=retrain_data_p.columns.values
                        if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col :
                            row=row[1:]
                        if row:
                            if len(row)==1:
                                row=predict_error
                            else:                   
                                row = ' '.join([str(r) for r in row])
                            repaired=model_flight.predict(row,k=1,threshold=threshold_v)
                            try:
                                if repaired[0][0]:
                                    total_error_to_repaired=total_error_to_repaired+1   
                                    k=repaired[0][0].replace('__label__','')
                                    repaired_k=k.replace('_',' ')
                                    repaired_k=repaired_k.strip()
                                    actual_value=actual_value.strip()
                                    if repaired_k==actual_value:
                                        total_repaired=total_repaired+1
                            except:
                                continue
                    else:
                        continue
                    

            #print(total_error,total_error_to_repaired,total_repaired )
        print(total_error,total_error_to_repaired,total_repaired )
        if total_error_to_repaired>0:
            p,r,f=evaluate_model(total_error,total_error_to_repaired,total_repaired)
            fp=total_error_to_repaired-total_repaired
            threshold_analysis.loc[-1] = [p, r,total_repaired,fp,threshold_v]
            threshold_analysis.index = threshold_analysis.index + 1  # shifting index
            threshold_analysis = threshold_analysis.sort_index()
        else:
            p,r,f="Invalid", "Invalid", "Invalid"
    path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "DC_threshold.csv"))
    write_csv_dataset(path,threshold_analysis)
       
    return p,r,f,total_repaired

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

def evaluate_model_2 (total_error, total_error_to_repair, total_correction):
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

def error_correction_edit_supervised_fasttext(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    repaired_dataset=data_for_retrain
    error_correction_edit=prepare_testing_datasets_real_world_data_types(dataframe,clean_dataframe,"textual",dataset_name)
     #total_error=0
    repaired_index_list=[]
    total_error_to_repaired=0
    total_repaired=0
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
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
    for error_value, actual_value, want_to_clean,index,col in zip(error_correction_edit['error'],error_correction_edit['actual'],error_correction_edit['want_to_clean'],error_correction_edit['index'],error_correction_edit['col']):
        total_p=total_p+1
        try:
            #total_error=total_error+1
            error_value=str(error_value)
            want_to_clean=str(want_to_clean)
            if  want_to_clean=="1" and len(error_value)<20 and total_error_to_repaired<5000 and len(error_value)>1 and error_value!="":
                error_value=str(error_value)
                error_value=error_value.strip()  
                first=model_edit_distance.correction(error_value)
                first=str(first)
                actual_value=str(actual_value)
                first=first.strip()
                actual_value=actual_value.strip()
                if first==error_value or first=="None":
                    continue
                else:
                    total_error_to_repaired=total_error_to_repaired+1
                    repaired_dataset.at[index,col]=first
                    repaired_index_list.append(tuple((index, col)))
                    if first==actual_value:
                        total_repaired=total_repaired+1
        except Exception as e:
            print('Exception: ', str(e))
            continue
    #total_error1=calculate_total_error_realworld(clean_dataframe, repaired_dataset)
    #print(repaired_index_list)
    #print(repaired_dataset)
   # print(total_error,total_error_to_repaired,total_repaired )
    print("Supervised Fasttext Start")
    #total_error_to_repaired=0
    #total_repaired=0
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
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    for fds in columns:
        if attribute and fds not in attribute:
            continue
        feature_data=columns
        data_for_retrain=repaired_dataset[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_flight_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_flight_full.txt",epoch=100,loss='hs')
        index_list=error_correction['index'].to_list()
        #print(index_list)
        error_correction_1=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
        for actual_value,want_to_clean,index, col in zip(error_correction['actual'],error_correction['want_to_clean'],error_correction['index'],error_correction['col']):
            index=int(index)
            index1=str(index)
            if (index1,col)  in repaired_index_list:
                #print('yes')
                continue #fds==col and (repaired_dataset.at[index,col] =="None" or  not repaired_dataset.at[index,col])
                #print('yes first')
            elif fds==col and (repaired_dataset.at[index,col] =="None" or  not repaired_dataset.at[index,col]):
                index1=str(index)
                if index1 in index_list: #total_error_to_repaired<5000
                    #print('yes second')
                    predict_error=repaired_dataset.iloc[index]
                    row = list(map(str, predict_error))
                    row=list(filter(None, row))
                    retrain_data_col=repaired_dataset.columns.values
                    if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                        row=row[1:]
                  #  if row:
                       # row = ' '.join([str(r) for r in row])            
                    try:
                       # print('yes')
                        if row or row != "":
                            row = ' '.join([str(r) for r in row]) 
                            #row=[1:]
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
        p,r,f=evaluate_model_2(total_error,total_error_to_repaired,total_repaired)
    else:
        p,r,f="Invalid", "Invalid", "Invalid"
    return p,r,f,total_repaired
def error_correction_supervised_fasttext_edit_distance_01(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    numeric_attr=numeric_attributes(dataframe,dataframe.columns.values)
    total_repair=0
    total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    column_name=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    track_repaired=[]
    for fds in columns:
        if attribute and fds not in attribute:
            continue
        feature_data=columns
        data_for_retrain=detected_dataset[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_full.txt",epoch=200,loss='hs')
        fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
        count=0
        for i, j in fds_dict:
            if count <1000: 
                count=count+1
                error_value=dataframe.iloc[i,j]
                error_value=str(error_value)
                error_value=error_value.strip()
                actual_value=str(diff_dict[i,j])
                actual_value=actual_value.strip()
                predict_error=detected_dataset.iloc[i]
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=detected_dataset.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                    row=row[1:]
                try:   
                    if row or row != "":
                        row = ' '.join([str(r) for r in row]) 
                        repaired=model_flight.predict(row,k=1,threshold=0.99)
                        if repaired[0][0]:
                            k=repaired[0][0].replace('__label__','')
                            repaired_f=k.replace('_',' ')
                            repaired_f=repaired_f.strip()
                            result_dictionary_fasttext_edit[i,j]=repaired_f
                except Exception as e:
                    print("Error correction exception: ", str(e))
                if error_value and len(error_value)<20: #((textual_attr and fds in textual_attr) or (numeric_attr and fds in numeric_attr))
                    repaired_e=model_edit_distance.correction(error_value)
                    repaired_e=str(repaired_e)
                    repaired_e=repaired_e.strip()
                    if repaired_e==error_value or repaired_e=="None":
                        continue
                    else:
                        result_dictionary_fasttext_edit[i,j]=repaired_e
            else:
                break
    total_error=0
    total_repair=0
    for i, j in   result_dictionary_fasttext_edit:
        actual_value=str(diff_dict[i,j])
        actual_value=actual_value.strip()
        repaired_value=str(result_dictionary_fasttext_edit[i,j])
        repaired_value=repaired_value.strip()
        total_error=total_error+1
        if repaired_value==actual_value:
            total_repair=total_repair+1
    print(total_error,total_repair)
    print(total_repair/total_error)
    with open('Fasttext_90_edit_textual_numeric_tax.pickle', 'wb') as handle:
        pickle.dump(result_dictionary_fasttext_edit, handle, protocol=pickle.HIGHEST_PROTOCOL)
def error_correction_supervised__edit_distance_01_fasttext(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    numeric_attr=numeric_attributes(dataframe,dataframe.columns.values)
    total_repair=0
    total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    column_name=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    track_repaired=[]
    for fds in columns:
        if attribute and fds not in attribute:
            continue
        feature_data=columns
        data_for_retrain=detected_dataset[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_full.txt",epoch=200,loss='hs')
        fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
        count=0
        for i, j in fds_dict:
            if count<1000: 
                count=count+1
                error_value=dataframe.iloc[i,j]
                error_value=str(error_value)
                error_value=error_value.strip()
                actual_value=str(diff_dict[i,j])
                actual_value=actual_value.strip()
                predict_error=detected_dataset.iloc[i]
                if error_value and len(error_value)<20:
                    repaired_e, prob=model_edit_distance.correction(error_value)
                    repaired_e=str(repaired_e)
                    repaired_e=repaired_e.strip()
                    if repaired_e==error_value or repaired_e=="None":
                        pass
                    else:
                        result_dictionary_fasttext_edit[i,j]=repaired_e
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=detected_dataset.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                    row=row[1:]
                try:   
                    if row or row != "":
                        row = ' '.join([str(r) for r in row]) 
                        repaired=model_flight.predict(row,k=1,threshold=0.90)
                        if repaired[0][0]:
                            k=repaired[0][0].replace('__label__','')
                            repaired_f=k.replace('_',' ')
                            repaired_f=repaired_f.strip()
                            result_dictionary_fasttext_edit[i,j]=repaired_f
                except Exception as e:
                    print("Error correction exception: ", str(e))
            else:
                break
                
    for i, j in   result_dictionary_fasttext_edit:
        actual_value=str(diff_dict[i,j])
        actual_value=actual_value.strip()
        repaired_value=str(result_dictionary_fasttext_edit[i,j])
        repaired_value=repaired_value.strip()
        total_error=total_error+1
        if repaired_value==actual_value:
            total_repair=total_repair+1
    print(total_error,total_repair)
    print(total_repair/total_error)         
def error_correction_supervised_fasttext_edit_distance_02_with_confidence(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    total_repair=0
    total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    column_name=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    track_repaired=[]
    for fds in columns:
        if attribute and fds not in attribute:
            continue
        feature_data=columns
        data_for_retrain=detected_dataset[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_full.txt", epoch=200, loss='hs')
        fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
        count=0
        for i, j in fds_dict:
            if column_name[j]==fds and count <1000:
                count=count+1
                track_repaired.append([i,j])
                error_value=dataframe.iloc[i,j]
                error_value=str(error_value)
                error_value=error_value.strip()
                actual_value=str(diff_dict[i,j])
                actual_value=actual_value.strip()
                predict_error=detected_dataset.iloc[i]
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=detected_dataset.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                    row=row[1:]
                repaired_e=None
                repaired_f=None
                if error_value  and len(error_value)<20 and textual_attr and fds in textual_attr:
                    repaired_e=model_edit_distance.correction(error_value)
                    repaired_e=str(repaired_e)
                    repaired_e=repaired_e.strip()
                try:
                    if row or row != "":
                        row = ' '.join([str(r) for r in row]) 
                        repaired=model_flight.predict(row,k=1,threshold=0.90)
                        if repaired:
                            k=repaired[0][0].replace('__label__','')
                            repaired_f=k.replace('_',' ')
                            repaired_f=repaired_f.strip()
                            model_confidence=repaired[1][0]
                except Exception as e:
                    print("Error correction exception: ", str(e))
                    #if repaired_e:
                        #if repaired_e==error_value or repaired_e=="None":
                           # continue
                       # else:
                            #result_dictionary_fasttext_edit[i,j]=repaired_e

                if repaired_e and  repaired_f and repaired_e==repaired_f:
                    result_dictionary_fasttext_edit[i,j]=repaired_e
                elif repaired_e and repaired_f and model_confidence>=0.99:
                    result_dictionary_fasttext_edit[i,j]=repaired_f
                elif (repaired_e==error_value or repaired_e=="None" or not repaired_e ) and repaired_f:
                    if model_confidence>=0.99:
                        result_dictionary_fasttext_edit[i,j]=repaired_f
                elif repaired_e:
                    if repaired_e==error_value or repaired_e=="None":
                        continue
                    else:
                        result_dictionary_fasttext_edit[i,j]=repaired_e
           
    for i, j in   result_dictionary_fasttext_edit:
        actual_value=str(diff_dict[i,j])
        actual_value=actual_value.strip()
        repaired_value=str(result_dictionary_fasttext_edit[i,j])
        repaired_value=repaired_value.strip()
        total_error=total_error+1
        if repaired_value==actual_value:
            total_repair=total_repair+1
    print(total_error,total_repair)
    print(total_repair/total_error)
def error_correction_supervised_fasttext_edit_distance_02_with_confidence_add_uniqueness(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    uniqueness_attr=uniqueness_percentage_30(dataframe,dataframe.columns.values)
    print(uniqueness_attr)
    total_repair=0
    #total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    column_name=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    track_repaired=[]
    for fds in columns:
        if attribute and fds not in uniqueness_attr:
            continue
        feature_data=columns
        data_for_retrain=detected_dataset[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_full.txt", epoch=200, loss='hs')
        fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
        count=0
        for i, j in fds_dict:
            if column_name[j]==fds and count <1000:
                count=count+1
                track_repaired.append([i,j])
                error_value=dataframe.iloc[i,j]
                error_value=str(error_value)
                error_value=error_value.strip()
                actual_value=str(diff_dict[i,j])
                actual_value=actual_value.strip()
                predict_error=detected_dataset.iloc[i]
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=detected_dataset.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                    row=row[1:]
                repaired_e=None
                repaired_f=None
                if error_value  and len(error_value)<20 :  #and textual_attr and fds in textual_attr
                    repaired_e=model_edit_distance.correction(error_value)
                    repaired_e=str(repaired_e)
                    repaired_e=repaired_e.strip()
                try:
                    if row or row != "":
                        row = ' '.join([str(r) for r in row]) 
                        repaired=model_flight.predict(row,k=1,threshold=0.90)
                        if repaired:
                            k=repaired[0][0].replace('__label__','')
                            repaired_f=k.replace('_',' ')
                            repaired_f=repaired_f.strip()
                            model_confidence=repaired[1][0]
                except Exception as e:
                    print("Error correction exception: ", str(e))
                    #if repaired_e:
                        #if repaired_e==error_value or repaired_e=="None":
                           # continue
                       # else:
                            #result_dictionary_fasttext_edit[i,j]=repaired_e

                if repaired_e and  repaired_f and repaired_e==repaired_f:
                    result_dictionary_fasttext_edit[i,j]=repaired_e
                
    total_error_to_repair=0    
    for i, j in   result_dictionary_fasttext_edit:
        actual_value=str(diff_dict[i,j])
        actual_value=actual_value.strip()
        repaired_value=str(result_dictionary_fasttext_edit[i,j])
        repaired_value=repaired_value.strip()
        total_error_to_repair=total_error_to_repair+1
        if repaired_value==actual_value:
            total_repair=total_repair+1
    print(total_error,total_repair)
    p,r,f=evaluate_model_2 (total_error, total_error_to_repair, total_repair)
    print(p,r,f,total_repair)
def error_correction_supervised_fasttext_edit_distance_02_equality_check(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    uniqueness_attr_10=uniqueness_percentage_10(dataframe,dataframe.columns.values)
    uniqueness_attr_20=uniqueness_percentage_20(dataframe,dataframe.columns.values)
    uniqueness_attr_30=uniqueness_percentage_30(dataframe,dataframe.columns.values)
    numeric_attr=numeric_attributes(dataframe,dataframe.columns.values)
    #print(uniqueness_attr)
    total_repair=0
    #total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    column_name=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    track_repaired=[]
    for fds in columns:
       # if attribute : #and fds not in uniqueness_attr
            #continue
        feature_data=columns
        data_for_retrain=detected_dataset[feature_data]
        data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
        data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
        df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
        df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
        model_flight=fasttext.train_supervised(input="datasets/training_full.txt", epoch=200, loss='hs')
        fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
        count=0
        for i, j in fds_dict:
            if column_name[j]==fds and count <1000:
                count=count+1
                track_repaired.append([i,j])
                error_value=dataframe.iloc[i,j]
                error_value=str(error_value)
                error_value=error_value.strip()
                actual_value=str(diff_dict[i,j])
                actual_value=actual_value.strip()
                predict_error=detected_dataset.iloc[i]
                row = list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=detected_dataset.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                    row=row[1:]
                repaired_e=None
                repaired_f=None
                if error_value  and len(error_value)<20 :  #and textual_attr and fds in textual_attr
                    repaired_e=model_edit_distance.correction(error_value)
                    repaired_e=str(repaired_e)
                    repaired_e=repaired_e.strip()
                try:
                    if row or row != "":
                        row = ' '.join([str(r) for r in row]) 
                        repaired=model_flight.predict(row,k=1,threshold=0.90)
                        if repaired:
                            k=repaired[0][0].replace('__label__','')
                            repaired_f=k.replace('_',' ')
                            repaired_f=repaired_f.strip()
                            model_confidence=repaired[1][0]
                except Exception as e:
                    print("Error correction exception: ", str(e))
                    #if repaired_e:
                        #if repaired_e==error_value or repaired_e=="None":
                           # continue
                       # else:
                            #result_dictionary_fasttext_edit[i,j]=repaired_e

                if repaired_e and  repaired_f and repaired_e==repaired_f:
                    result_dictionary_fasttext_edit[i,j]=repaired_e
                elif repaired_e and repaired_f and model_confidence>=0.995: #without repired e
                    result_dictionary_fasttext_edit[i,j]=repaired_f
                elif repaired_f and (uniqueness_attr_10 and fds in uniqueness_attr_10): # with repaired e
                    result_dictionary_fasttext_edit[i,j]=repaired_f
               # elif repaired_f and (uniqueness_attr_20 and fds in uniqueness_attr_20):
                    #result_dictionary_fasttext_edit[i,j]=repaired_f
                elif repaired_e and ((textual_attr and fds in textual_attr) ) : # or (numeric_attr and fds in numeric_attr)
                    if repaired_e==error_value:
                        continue
                    else:
                        result_dictionary_fasttext_edit[i,j]=repaired_e

                
    total_error_to_repair=0    
    for i, j in   result_dictionary_fasttext_edit:
        actual_value=str(diff_dict[i,j])
        actual_value=actual_value.strip()
        repaired_value=str(result_dictionary_fasttext_edit[i,j])
        repaired_value=repaired_value.strip()
        total_error_to_repair=total_error_to_repair+1
        if repaired_value==actual_value:
            total_repair=total_repair+1
    print(total_error,total_repair)
    p,r,f=evaluate_model_2 (total_error, total_error_to_repair, total_repair)
    print(p,r,f,total_repair)
def model_aggregation_first(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    error_correction_prob = pd.DataFrame(columns = ['actual', 'repair','prob'])
    aggregate_prob=0
    actual_value=None
    repaired_value=None
    prob_e=0
    prob_f=0
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    numeric_attr=numeric_attributes(dataframe,dataframe.columns.values)
    uniqueness_attr_10=uniqueness_percentage_10(dataframe,dataframe.columns.values)
    total_repair=0
    total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    for c in [0.00,0.25,0.50,0.75,1.00]:
        error_correction_prob = pd.DataFrame(columns = ['actual', 'repair','prob'])
        aggregate_prob=0
        actual_value=None
        repaired_value=None
        prob_e=0
        prob_f=0
        columns=list(dataframe.columns.values)
        column_name=list(dataframe.columns.values)
        if columns[0]=="tuple_id" or columns[0]=="index":
            columns.remove(columns[0])
        track_repaired=[]
        for fds in columns:
            if attribute and fds not in attribute:
                continue
            feature_data=columns
            data_for_retrain=detected_dataset[feature_data]
            data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
            data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
            df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
            df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
            model_flight=fasttext.train_supervised(input="datasets/training_full.txt",epoch=200,loss='hs')
            fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
            count=0
            for i, j in fds_dict:
                aggregate_prob=0
                #actual_value=None
                repaired_value=None
                prob_e=0
                prob_f=0
                repaired_e=None
                repaired_f=None
                if count<2000:
                    count=count+1
                    error_value=dataframe.iloc[i,j]
                    error_value=str(error_value)
                    error_value=error_value.strip()
                    actual_value=str(diff_dict[i,j])
                    actual_value=actual_value.strip()
                    predict_error=detected_dataset.iloc[i]
                    if error_value and len(error_value)<20:
                        repaired_e, prob_e=model_edit_distance.correction(error_value)
                        repaired_e=str(repaired_e)
                        repaired_e=repaired_e.strip()
                        if repaired_e==error_value or repaired_e=="None" or prob_e==0 or repaired_e=="":
                            repaired_e=None
                    row = list(map(str, predict_error))
                    row=list(filter(None, row))
                    retrain_data_col=detected_dataset.columns.values
                    if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                        row=row[1:]
                    try:   
                        if row or row != "":
                            row = ' '.join([str(r) for r in row]) 
                            repaired=model_flight.predict(row,k=1,threshold=0.90)
                            if repaired[0][0]:
                                k=repaired[0][0].replace('__label__','')
                                repaired_f=k.replace('_',' ')
                                repaired_f=repaired_f.strip()
                                prob_f=repaired[1][0]
                                #result_dictionary_fasttext_edit[i,j]=repaired_f
                    except Exception as e:
                        print("Error correction exception: ", str(e))
                    if repaired_e or repaired_f:
                        aggregate_prob_e=0
                        aggregate_prob_f=0
                        if repaired_e and repaired_f  and repaired_e==repaired_f:
                            aggregate_prob=c*prob_e+(1-c)*prob_f
                            repaired_value= repaired_e
                        else:
                            if repaired_e:
                                aggregate_prob_e=c*prob_e+(1-c)*prob_f
                            if repaired_f:
                                aggregate_prob_f=c*prob_e+(1-c)*prob_f
                            if repaired_f and repaired_e and prob_f>=0.99:  
                                repaired_value=repaired_f
                                aggregate_prob=aggregate_prob_f
                            elif repaired_e and (textual_attr and (fds in textual_attr)):
                                prob_e=1.0
                                aggregate_prob=c*prob_e+(1-c)*prob_f
                                repaired_value=repaired_e
                            elif repaired_f and (uniqueness_attr_10 and fds in uniqueness_attr_10):
                                repaired_value=repaired_f
                                aggregate_prob=aggregate_prob_f
                        if repaired_value:
                            error_correction_prob.loc[-1] = [actual_value, repaired_value,aggregate_prob]
                            error_correction_prob.index = error_correction_prob.index + 1  # shifting index
                            error_correction_prob = error_correction_prob.sort_index()
                else:
                    break
                
        k=str(c*100)
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, k+".csv"))
        #print("Finishing preparing data set: ", name)
        write_csv_dataset(path,error_correction_prob)
def model_aggregation_last(model_type,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe,data_type):
    diff_dict=get_dataframes_difference(dataframe, clean_dataframe)
    error_correction_prob = pd.DataFrame(columns = ['actual', 'repair','prob'])
    aggregate_prob=0
    actual_value=None
    repaired_value=None
    prob_e=0
    prob_f=0
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    numeric_attr=numeric_attributes(dataframe,dataframe.columns.values)
    uniqueness_attr_10=uniqueness_percentage_10(dataframe,dataframe.columns.values)
    total_repair=0
    total_error=0
    result_dictionary_edit={}
    result_dictionary_fasttext={}
    result_dictionary_fasttext_edit={}
    result_list=[]
    detected_dataset=data_for_retrain
    try:
        with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
            model_edit_distance = pickle.load(pickle_file)
    except Exception as e:
        print('Model Error: ',str(e))  
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
    print("Supervised Fasttext Start")
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
    for c in [0.00,0.25,0.50,0.75,1.00]:
        error_correction_prob = pd.DataFrame(columns = ['actual', 'repair','prob'])
        aggregate_prob=0
        actual_value=None
        repaired_value=None
        prob_e=0
        prob_f=0
        columns=list(dataframe.columns.values)
        column_name=list(dataframe.columns.values)
        if columns[0]=="tuple_id" or columns[0]=="index":
            columns.remove(columns[0])
        track_repaired=[]
        for fds in columns:
            if attribute and fds not in attribute:
                continue
            feature_data=columns
            data_for_retrain=detected_dataset[feature_data]
            data_for_retrain[fds]='__label__' + data_for_retrain[fds].astype(str)
            data_for_retrain[fds]=[s.replace(" ", "_") for s in data_for_retrain[fds]]
            df3 = data_for_retrain[data_for_retrain[fds] != "__label__None"]
            df3.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
            model_flight=fasttext.train_supervised(input="datasets/training_full.txt",epoch=200,loss='hs')
            fds_dict = {k: v for k, v in diff_dict.items() if column_name[k[1]]==fds}
            count=0
            for i, j in fds_dict:
                aggregate_prob=0
                #actual_value=None
                repaired_value=None
                prob_e=0
                prob_f=0
                repaired_e=None
                repaired_f=None
                if count<2000:
                    count=count+1
                    error_value=dataframe.iloc[i,j]
                    error_value=str(error_value)
                    error_value=error_value.strip()
                    actual_value=str(diff_dict[i,j])
                    actual_value=actual_value.strip()
                    predict_error=detected_dataset.iloc[i]
                    if error_value and len(error_value)<20:
                        repaired_e, prob_e=model_edit_distance.correction(error_value)
                        repaired_e=str(repaired_e)
                        repaired_e=repaired_e.strip()
                        if repaired_e==error_value or repaired_e=="None" or prob_e==0 or repaired_e=="":
                            repaired_e=None
                    row = list(map(str, predict_error))
                    row=list(filter(None, row))
                    retrain_data_col=detected_dataset.columns.values
                    if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                        row=row[1:]
                    try:   
                        if row or row != "":
                            row = ' '.join([str(r) for r in row]) 
                            repaired=model_flight.predict(row,k=1,threshold=0.90)
                            if repaired[0][0]:
                                k=repaired[0][0].replace('__label__','')
                                repaired_f=k.replace('_',' ')
                                repaired_f=repaired_f.strip()
                                prob_f=repaired[1][0]
                                #result_dictionary_fasttext_edit[i,j]=repaired_f
                    except Exception as e:
                        print("Error correction exception: ", str(e))
                    aggregate_prob=c*prob_e+(1-c)*prob_f
                    if repaired_e and repaired_f:
                        if repaired_e==repaired_f:
                            repaired_value=repaired_f
                        elif c<0.50:
                            repaired_value=repaired_f
                        elif c==0.50:
                            if (textual_attr and (fds in textual_attr)):
                                repaired_value=repaired_e
                            else:
                                repaired_value=repaired_f
                        else:
                            repaired_value=repaired_e
                    elif repaired_e or repaired_f:
                        if repaired_e and  c>0:
                            repaired_value=repaired_e
                        elif repaired_f and c<1:
                            repaired_value=repaired_f

                    if repaired_value:
                        error_correction_prob.loc[-1] = [actual_value, repaired_value,aggregate_prob]
                        error_correction_prob.index = error_correction_prob.index + 1  # shifting index
                        error_correction_prob = error_correction_prob.sort_index()
                else:
                    break              
        k=str(c*100)
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, k+"new.csv"))
        #print("Finishing preparing data set: ", name)
        write_csv_dataset(path,error_correction_prob)
                    
def model_aggregation_final(dirty_org_df,error_correction,data_for_retrain,dataset_name,total_error,dataframe,clean_dataframe):
    repaired_dataset = pd.DataFrame(columns = ['actual','error', 'edit','prob_e','fasttext','prob_f','textual','uniqueness','eflag','fflag'])
    #print(dirty_org_df.dtypes)
    textual_attr=textual_attributes(dataframe,dataframe.columns.values)
    alphanum_attr=alphanumeric_attribute(dataframe,dataframe.columns.values)
    uniqueness_attr_10=uniqueness_percentage_10(dataframe,dataframe.columns.values)
    uniqueness_attr_20=uniqueness_percentage_20(dataframe,dataframe.columns.values)
    uniqueness_attr_30=uniqueness_percentage_30(dataframe,dataframe.columns.values)
    numeric_attr=numeric_attributes(dataframe,dataframe.columns.values)
    detected_dataset=data_for_retrain
    dirty_data=dataframe
    columns=list(dataframe.columns.values)
    column_name=list(dataframe.columns.values)
    if columns[0]=="tuple_id" or columns[0]=="index":
        columns.remove(columns[0])
    for fds in columns:
        if dirty_org_df[fds].dtypes=='float64':
        #    print('yes')
             pass
             #continue
        count=0
        data_for_retrain_edit=[]
        train_data_rows_edit=[]
        textual_flag=False
        uniqueness_attr_30_flag=False
        model_edit_distance=None
        feature_data=columns
        ######################EDit distance##############
        data_for_retrain_edit=detected_dataset[fds]
        data_for_retrain_edit=data_for_retrain_edit.values.tolist()
        #for row in data_for_retrain_edit:
        data_for_retrain_edit = [i for i in data_for_retrain_edit if i] 
        data_for_retrain_edit = list(map(str, data_for_retrain_edit))
        data_for_retrain_edit=list(filter(None, data_for_retrain_edit))
        for item in data_for_retrain_edit:
            k=str(item)
            if k=='None'or k=="None" or not item:
                data_for_retrain_edit.remove(item)
        #print(data_for_retrain_edit)

        #train_data_rows_edit.extend(row)
        if data_for_retrain_edit:
            general_corpus = [str(s) for s in data_for_retrain_edit]
            corpus = Counter(general_corpus)
            model_edit_distance = SpellCorrector(dictionary=corpus, verbose=1)   
        ####################################################
        #########################Fasttext###################
        #delete=__label__None
        #data_for_retrain_fast=
        data_for_retrain_fast=detected_dataset[feature_data]
        data_for_retrain_fast[fds]="__label__" + data_for_retrain_fast[fds].astype(str)
        data_for_retrain_fast[fds]=[s.replace(" ", "_") for s in data_for_retrain_fast[fds]]
        data_for_retrain_fast = data_for_retrain_fast[data_for_retrain_fast[fds] != "__label__None"]
        #df3 = df3[df3.fds != "__label__None"]
        #df3 = data_for_retrain_fast[data_for_retrain_fast[fds] != __label__None]
        #df3 = data_for_retrain_fast[data_for_retrain_fast[fds] != "__label__empty"]
        #print(data_for_retrain_fast)
        if not data_for_retrain_fast.empty:
            data_for_retrain_fast.to_csv(r'datasets/training_full.txt', index=False, sep=' ', header=False)
        #print('yes')
            model_fasttext=fasttext.train_supervised(input="datasets/training_full.txt", epoch=100, loss='hs')
        else:
            model_fasttext=None

        ########################################################
        error_correction=prepare_testing_datasets_real_world_data_error(dataframe,clean_dataframe,"sf",dataset_name,fds)
        if (uniqueness_attr_30 and fds in uniqueness_attr_30): # with repaired e
            uniqueness_attr_30_flag=True
        if ((textual_attr and fds in textual_attr)):
            textual_flag=True
        fasttext_retrain=detected_dataset[feature_data]
        for actual_value,error_value,want_to_clean,index, col in zip(error_correction['actual'],error_correction['error'],error_correction['want_to_clean'],error_correction['index'],error_correction['col']):
            repaired_e="NAE"
            repaired_f="NAE"
            prob_e=0
            prob_f=0
            eflag=False
            fflag=False
            row=None
            count=count+1
            error_value=str(error_value)
            error_value=error_value.strip()
            actual_value=str(actual_value)
            actual_value=actual_value.strip()
            if error_value ==actual_value:
                continue
            if model_edit_distance and error_value  and len(error_value)<20:  #and textual_attr and fds in textual_attr
                #print(model_edit_distance.correction(error_value))
                repaired_e,prob_e=model_edit_distance.correction(error_value)
                #print(repaired_e, prob_e)
                repaired_e=str(repaired_e)
                repaired_e=repaired_e.strip()
                if repaired_e==error_value:
                    repaired_e="NAE"
            index=int(index)
            if fds==col and  len(error_value)<20 and (fasttext_retrain.at[index,col] =="None" or  not fasttext_retrain.at[index,col]):
                predict_error=fasttext_retrain.iloc[index]
                predict_error=list(predict_error)
                for item in predict_error:
                    k=str(item)
                    if k=='None'or k=="None" or not item:
                        predict_error.remove(item)
                row =list(map(str, predict_error))
                row=list(filter(None, row))
                retrain_data_col=detected_dataset.columns.values
                if retrain_data_col[0]=="tuple_id" or retrain_data_col[0]=="index" or "tuple_id" in retrain_data_col or "index" in retrain_data_col:
                    row=row[1:]     
                try:
                    if row or row != "":
                        row = ' '.join([str(r) for r in row])
                        if model_fasttext:
                            repaired=model_fasttext.predict(row,k=1,threshold=0.10)
                            if repaired[0][0]:
                                k=repaired[0][0].replace('__label__','')
                                repaired_f=k.replace('_',' ')
                                repaired_f=str(repaired_f)
                                repaired_f=repaired_f.strip()
                                prob_f=repaired[1][0]
                except Exception as e:
                    print("Error correction exception: ", str(e))

                if (repaired_e or repaired_f) and (prob_e>0 or prob_f>0):
                    if repaired_e or repaired_e!= "NAE":
                        if repaired_e==actual_value:
                            eflag=True
                    else:
                        eflag="NAE"
                    if repaired_e or repaired_e!= "NAE":
                        if repaired_f==actual_value:
                            fflag=True
                    else:
                        fflag="NAE"
                    repaired_dataset.loc[-1] = [actual_value,error_value,repaired_e,prob_e,repaired_f,prob_f,textual_flag,uniqueness_attr_30_flag,eflag,fflag]
                    repaired_dataset.index = repaired_dataset.index + 1  # shifting index
                    repaired_dataset = repaired_dataset.sort_index()
#total_error=4920
    total_error_to_reapir=0
    total_repair=0
    coeff_e=0
    prob_e=0
    prob_f=0
    threshold=0.75
    repaired=None
    aggre_prob=coeff_e*prob_e+(1-coeff_e)*prob_f
    for coeff in [0.50]:
        total_error_to_reapir=0
        total_repair=0
        for actual, repair_e, prob_e, repair_f,prob_f in zip(repaired_dataset['actual'],repaired_dataset['edit'],repaired_dataset['prob_e'],repaired_dataset['fasttext'],repaired_dataset['prob_f']):
            aggre_prob=coeff*prob_e+(1-coeff)*prob_f
            if repair_e !="NAE" and repair_f!="NAE" and repair_e==repair_f and prob_e>=0.50:
                total_error_to_reapir=total_error_to_reapir+1
                repaired=repair_e    
                if actual==repaired:
                    total_repair=total_repair+1 
            elif coeff==0:
                if prob_f>=0.99 and repair_f!="NAE" and aggre_prob>=threshold:
                    total_error_to_reapir=total_error_to_reapir+1
                    repaired=repair_f
                    if actual==repaired:
                        total_repair=total_repair+1 
            elif coeff==1:
                if prob_e>=0.50 and repair_e !="NAE" and repair_e and aggre_prob>=threshold:
                    total_error_to_reapir=total_error_to_reapir+1
                    repaired=repair_f
                    if actual==repaired:
                        total_repair=total_repair+1    
            elif prob_f>=0.90 and repair_f!="NAE" and aggre_prob>=threshold:
                total_error_to_reapir=total_error_to_reapir+1
                repaired=repair_f
                if actual==repaired:
                    total_repair=total_repair+1
            elif prob_e>=0.50 and repair_e !="NAE" and repair_e and aggre_prob>=threshold:
                total_error_to_reapir=total_error_to_reapir+1
                repaired=repair_e
                if actual==repaired:
                    total_repair=total_repair+1   

        p,r,f=evaluate_model(total_error,total_error_to_reapir,total_repair)
        #print(coeff,p,total_repair,total_error_to_reapir)
        print("Performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(dataset_name, p, r, f))

    path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "repaired_updated.csv"))
    write_csv_dataset(path,repaired_dataset)     
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
        f_score=round(f_score,2)     
    return precision, recall,f_score

########## Loading Library##############
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
import wikitextparser as wtp
from pprint import pprint
from wikitextparser import remove_markup, parse
import datetime
from pandas import DataFrame
import random
import pickle
import pandas as pd
import logging
###########
def get_dataframes_difference(dataframe_1, dataframe_2): #return detected error cell with correction value
    if dataframe_1.shape != dataframe_2.shape:
        sys.stderr.write("Two compared datasets do not have equal sizes!\n")
    difference_dictionary = {}
    difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
    for j in range(dataframe_1.shape[1]):
        for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
            difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
    return difference_dictionary
def value_normalizer(value):
        """
        This method takes a value and minimally normalizes it.
        """
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
        return value
def read_csv_dataset(dataset_path):
        """
        This method reads a dataset from a csv file path.
        """
        dataframe = pd.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).applymap(value_normalizer)
        return dataframe
def write_csv_dataset(dataset_path, dataframe):
        """
        This method writes a dataset to a csv file path.
        """
        dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")
def textual_value(value):
        """
        This method will return True if the value contains only alphabet
        """
        Flag=False
        if not any(x1.isdigit() for x1 in value):
                Flag=True
        return Flag
def numeric_value(value):
        """
        This method will return True if the value contains only digit
        """
        Flag=False
        if all(x1.isdigit() for x1 in value):
                Flag=True
        return Flag
def alphanumeric_value(value):
        """
        This method will return True if the value contains mixure of digit and alphabet
        """
        Flag=False
        check_flag_text=textual_value(value)
        if not check_flag_text:
                check_flag_num=textual_value(value)
                if not check_flag_num:
                        Flag=True
        return Flag
def textual_attributes(df,columns):
        """
        This method will return all columns which contains only textual value
        """
        column = []

        for col in columns:
                count=0
                for value in df[col]:
                        check_text=textual_value(value)
                        if check_text:
                                count=count+1
                percent=(100*count)/(len(df[col]))
                if percent>50:
                        column.append(col)
        return column

              
        #return diversity
def numeric_attributes(df,columns):
        """
        This method will return all columns which contains only numeric value
        """
        column = []

        for col in columns:
                count=0
                for value in df[col]:
                        check_text=numeric_value(value)
                        if check_text:
                                count=count+1
                percent=(100*count)/(len(df[col]))
                if percent>50:
                       column.append(col)
        return column
def alphanumeric_attribute(df,columns):
        """
        This method will return all columns which contains  textual value and numeric value
        """
        column = []

        for col in columns:
                count=0
                for value in df[col]:
                        check_text=alphanumeric_value(value)
                        if check_text:
                                count=count+1
                percent=(100*count)/(len(df[col]))
                if percent>50:
                        column.append(col)
        return column
def uniqueness_percentage_30(df, columns):
    """
    Return all attributes which have the 0-30% unique values
    """
    diversity = []

    for col in columns:
        divert = len(df[col].unique())
        percen=(100*divert)/(len(df[col]))
        if percen>=0 and percen<=30:
          diversity.append(col)

    #diversity_series = pd.Series(diversity)
    return diversity
def uniqueness_percentage_20(df, columns):
    """
    Return all attributes which have the 0-30% unique values
    """
    diversity = []

    for col in columns:
        divert = len(df[col].unique())
        percen=(100*divert)/(len(df[col]))
        if percen>=0 and percen<=20:
          diversity.append(col)

    #diversity_series = pd.Series(diversity)
    return diversity
def uniqueness_percentage_10(df, columns):
    """
    Return all attributes which have the 0-30% unique values
    """
    diversity = []

    for col in columns:
        divert = len(df[col].unique())
        percen=(100*divert)/(len(df[col]))
        if percen>=0 and percen<=10:
          diversity.append(col)

    #diversity_series = pd.Series(diversity)
    return diversity
def uniqueness_percentage_60(df, columns):
    """
    Return all attributes which have the 31-60% unique values
    """
    diversity = []

    for col in columns:
        divert = len(df[col].unique())
        percen=(100*divert)/(len(df[col]))
        if percen>=31 and percen<=60:
          diversity.append(col)

    #diversity_series = pd.Series(diversity)
    return diversity
def uniqueness_percentage_90(df, columns):

    """
    Return all attributes which have the 61-100% unique values
    """
    diversity = []

    for col in columns:
        divert = len(df[col].unique())
        percen=(100*divert)/(len(df[col]))
        if percen>=61 and percen<=100:
          diversity.append(col)

    #diversity_series = pd.Series(diversity)
    return diversity
def prepare_testing_datasets_real_world_data_types(dirty_data,clean_data, data_type,name):
        #actual_error = pd.DataFrame(columns = ['actual', 'error','index','want_to_clean'])
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','col','want_to_clean']) #want_to_clean 1 or 0
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        if data_type=="textual":
                attribute=textual_attributes(dirty_data,dirty_data_col)
                print(attribute)
        elif data_type=="alphanumeric":
                attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
                print(attribute) #alphanumeric
        elif data_type=="numeric":
                attribute=numeric_attributes(dirty_data,dirty_data_col)
                print(attribute)  #numeric
        elif data_type=="uniqueness_0_30":
                attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)
                print(attribute)
        elif data_type=="uniqueness_31_60":
                attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
                print(attribute)
        elif data_type=="uniqueness_61_100":
                attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
                print(attribute)
        if attribute:
                for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                        want_to_clean=0
                        if dir_col in attribute:
                                want_to_clean=1
                                for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                                        value1=str(value1)
                                        value1=value_normalizer(value1)
                                        value2=str(value2)
                                        value2=value_normalizer(value2)
                                        value1=value1.strip()
                                        value2=value2.strip()
                                        if value1==value2:
                                                continue
                                        else:
                                                actual_error.loc[-1] = [value1, value2,indx,dir_col,want_to_clean]
                                                actual_error.index = actual_error.index + 1  # shifting index
                                                actual_error = actual_error.sort_index()
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_type+".csv"))
        #print("Finishing preparing data set: ", name)
        write_csv_dataset(path,actual_error)
        
        return actual_error
def prepare_dataset_for_retrain_realworld(clean_data_df, dirty_data_df): #send two datasets
        clean_data = clean_data_df.applymap(str)
        dirty_data = dirty_data_df.applymap(str)
        if clean_data.shape != dirty_data.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        else:
            clean_data_col=clean_data.columns.values
            dirty_data_col=dirty_data.columns.values
            for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = None #replace error value with NaN
        return dirty_data
def prepare_dataset_for_retrain_realworld_domain(clean_data_df,dirty_data_df, domain_dirty_col):
        clean_data = clean_data_df.applymap(str)
        dirty_data = dirty_data_df.applymap(str)
        domain_col=[]
        if clean_data.shape != dirty_data.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        else:
            clean_data_col=clean_data.columns.values
            dirty_data_col=dirty_data.columns.values
            for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                if dir_col in domain_dirty_col:
                   domain_col.append(dir_col)
                   dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = None #replace error value with NaN
        
        dirty_data=dirty_data[domain_col]
        #print(dirty_data)
        return dirty_data

def prepare_domain_testing_datasets_real_world_data_types(dirty_data,clean_data,data_type,name,domain_dirty_col,domain_clean_col):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','want_to_clean']) #want_to_clean 1 or 0
        #clean_data=pd.read_csv(clean_data_path)
        #dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        if data_type=="textual":
                attribute=textual_attributes(dirty_data,dirty_data_col)
                print(attribute)
        if data_type=="alphanumeric":
                attribute=alphanumeric_attribute(dirty_data,dirty_data_col)
                print(attribute) #alphanumeric
        if data_type=="numeric":
                attribute=numeric_attributes(dirty_data,dirty_data_col)
                print(attribute)  #numeric
        if data_type=="uniqueness_0_30":
                attribute=uniqueness_percentage_30(dirty_data,dirty_data_col)
                print(attribute)
        if data_type=="uniqueness_31_60":
                attribute=uniqueness_percentage_60(dirty_data,dirty_data_col)
                print(attribute)
        if data_type=="uniqueness_61_100":
                attribute=uniqueness_percentage_90(dirty_data,dirty_data_col)
                print(attribute)
        if attribute:
                for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                        want_to_clean=0
                        if dir_col in attribute and dir_col in domain_dirty_col:
                                want_to_clean=1
                                for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                                        value1=str(value1)
                                        value1=value_normalizer(value1)
                                        value2=str(value2)
                                        value2=value_normalizer(value2)
                                        value1=value1.strip()
                                        value2=value2.strip()
                                        if value1==value2:
                                                continue
                                        else:
                                                actual_error.loc[-1] = [value1, value2,indx,want_to_clean]
                                                actual_error.index = actual_error.index + 1  # shifting index
                                                actual_error = actual_error.sort_index()
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_type+"_domain.csv"))
        #print("Finishing preparing data set: ", name)
        write_csv_dataset(path,actual_error)
        return actual_error

def prepare_testing_datasets_real_world_data_error(dirty_data,clean_data, data_error,name,fds_col):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','want_to_clean','col']) #want_to_clean 1 or 0
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        if data_error=="fds" or data_error=="sf":
                attribute=fds_col
        elif data_error=="pvs":
                attribute=list(set(dirty_data_col) - set(fds_col))
        if attribute:
                for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                        want_to_clean=0
                        if dir_col in attribute:
                                want_to_clean=1
                                for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                                        if value1==value2:
                                                continue
                                        value1=str(value1)
                                        value1=value_normalizer(value1)
                                        value2=str(value2)
                                        value2=value_normalizer(value2)
                                        value1=value1.strip()
                                        value2=value2.strip()
                                        if value1==value2:
                                                continue
                                        else:
                                                actual_error.loc[-1] = [value1, value2,indx,want_to_clean,dir_col]
                                                actual_error.index = actual_error.index + 1  # shifting index
                                                actual_error = actual_error.sort_index()
        if data_error=="fds":
                path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_error+".csv"))
                #print("Finishing preparing data set: ", name)
                write_csv_dataset(path,actual_error)
        
        return actual_error
def prepare_domain_testing_datasets_real_world_data_error(dirty_data,clean_data,data_error,name,domain_dirty_col,domain_clean_col,fds_col):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','want_to_clean']) #want_to_clean 1 or 0
        #clean_data=pd.read_csv(clean_data_path)
        #dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        if data_error=="fds":
                attribute=fds_col
        elif data_error=="pvs":
                attribute=list(set(dirty_data_col) - set(fds_col))
        if attribute:
                for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                        want_to_clean=0
                        if dir_col in attribute and dir_col in domain_dirty_col:
                                want_to_clean=1
                                for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                                        value1=str(value1)
                                        value1=value_normalizer(value1)
                                        value2=str(value2)
                                        value2=value_normalizer(value2)
                                        value1=value1.strip()
                                        value2=value2.strip()
                                        if value1==value2:
                                                continue
                                        else:
                                                actual_error.loc[-1] = [value1, value2,indx,want_to_clean]
                                                actual_error.index = actual_error.index + 1  # shifting index
                                                actual_error = actual_error.sort_index()
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_error+"_domain.csv"))
        #print("Finishing preparing data set: ", name)
        write_csv_dataset(path,actual_error)
        return actual_error
def calculate_total_error_realworld(clean_data, dirty_data):
        total_error=0
        #clean_data=pd.read_csv(clean_data_path)
        #dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                value1=str(value1)
                value2=str(value2)
                if value1==value2:
                    continue
                else:
                    total_error=total_error+1
                    
        return total_error
def get_dataframes_difference(dataframe_1, dataframe_2):
        """
        This method compares two dataframes and returns the different cells.
        """
        if dataframe_1.shape != dataframe_2.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        difference_dictionary = {}
        difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
        for j in range(dataframe_1.shape[1]):
            for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
                difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
        return difference_dictionary

def prepare_testing_datasets_real_world_general_general(dirty_data,clean_data,data_error,name):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','want_to_clean']) #want_to_clean 1 or 0
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                want_to_clean=0
                for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                        value1=str(value1)
                        value1=value_normalizer(value1)
                        value2=str(value2)
                        value2=value_normalizer(value2)
                        value1=value1.strip()
                        value2=value2.strip()
                        if value1==value2:
                                continue
                        else:
                                want_to_clean=1
                                actual_error.loc[-1] = [value1, value2,indx,want_to_clean]
                                actual_error.index = actual_error.index + 1  # shifting index
                                actual_error = actual_error.sort_index()
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_error+".csv"))
        write_csv_dataset(path,actual_error)
        return actual_error
def prepare_testing_datasets_real_world_general_general_agg(dirty_data,clean_data,data_error,name):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','col','want_to_clean']) #want_to_clean 1 or 0
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                want_to_clean=0
                for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                        value1=str(value1)
                        value1=value_normalizer(value1)
                        value2=str(value2)
                        value2=value_normalizer(value2)
                        value1=value1.strip()
                        value2=value2.strip()
                        if value1==value2:
                                continue
                        else:
                                want_to_clean=1
                                actual_error.loc[-1] = [value1, value2,indx,dir_col,want_to_clean]
                                actual_error.index = actual_error.index + 1  # shifting index
                                actual_error = actual_error.sort_index()
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_error+".csv"))
        write_csv_dataset(path,actual_error)
        return actual_error
def prepare_domain_testing_datasets_real_world_general(dirty_data,clean_data,data_error,name,domain_dirty_col,domain_clean_col):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index','want_to_clean']) #want_to_clean 1 or 0
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                want_to_clean=0
                if  dir_col in domain_dirty_col:
                        for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                                value1=str(value1)
                                value1=value_normalizer(value1)
                                value2=str(value2)
                                value2=value_normalizer(value2)
                                value1=value1.strip()
                                value2=value2.strip()
                                if value1==value2:
                                        continue
                                else:
                                        want_to_clean=1
                                        actual_error.loc[-1] = [value1, value2,indx,want_to_clean]
                                        actual_error.index = actual_error.index + 1  # shifting index
                                        actual_error = actual_error.sort_index()
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", name, data_error+"_domain.csv"))
        write_csv_dataset(path,actual_error)
        return actual_error
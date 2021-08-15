########################################
# Jharu: The Error Correction Using Wikipedia Revision History
# Md Kamrul Hasan
# kamrulhasancuetcse10@gmail.com
# June 2020-Present
# Master thesis, Big Data Management Group , TU Berlin
# Special thanks to:  Mohammad Mahdavi, moh.mahdavi.l@gmail.com, code repository :  https://github.com/BigDaMa/raha.git


########################################
# This module will extract table and infobox from wiki dump file. Then it will extract the revision data and extract the error and clean value by comparing two revision.
# This module then train edit distance, fasttext, elmo like wrod embedding method 
# We can retrained the pre trained model
# we can cross validate our testing and traning data set of wiki 
# we can correction a  dirty dataset in which error detected alreay have perfromed
# we can evaluate a model based on wiki model
# we can retrain on real world dirty dataset 
# we can  train model based on domain . for  example localtion, date etc. and apply on real world datasets
# This is the whole cleaning pipeline for our system
########################################
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
class Jharu:
    """
    The main class.
    """
    def __init__(self):
        """
        The constructor.
        """
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = True
        self.SAVE_RESULTS = True
        self.ONLINE_PHASE = False
        self.REFINE_PREDICTIONS = True
        self.LABELING_BUDGET = 20
        self.MAX_VALUE_LENGTH = 50
        self.REVISION_WINDOW_SIZE = 5
             
    def extract_revisions(self, wikipedia_dumps_folder,parsing_type):
        """
        This method takes the folder path of Wikipedia page revision history dumps and extracts infobox/table revision data.
        """
        self.rd_folder_path = os.path.join(wikipedia_dumps_folder, "revision-data")
        if not os.path.exists(self.rd_folder_path):
            os.mkdir(self.rd_folder_path)
        if parsing_type=="table":
            self.rd_folder_path_table= os.path.join(self.rd_folder_path, "table")
            if not os.path.exists(self.rd_folder_path_table):
                os.mkdir(self.rd_folder_path_table)
        elif parsing_type=="infobox":
            self.rd_folder_path_infobox = os.path.join(self.rd_folder_path, "infobox")
            if not os.path.exists(self.rd_folder_path_infobox):
                os.mkdir(self.rd_folder_path_infobox)
        elif parsing_type=="both":
            self.rd_folder_path_table = os.path.join(self.rd_folder_path, "table")
            if not os.path.exists(self.rd_folder_path_table):
                os.mkdir(self.rd_folder_path_table)
            self.rd_folder_path_infobox = os.path.join(self.rd_folder_path, "infobox")
            if not os.path.exists(self.rd_folder_path_infobox):
                os.mkdir(self.rd_folder_path_infobox)
            
        
        compressed_dumps_list = [df for df in os.listdir(wikipedia_dumps_folder) if df.endswith(".7z")]
        for file_name in compressed_dumps_list:
            compressed_dump_file_path = os.path.join(wikipedia_dumps_folder, file_name)
            dump_file_name, _ = os.path.splitext(os.path.basename(compressed_dump_file_path))
            self.rdd_folder_path = os.path.join(self.rd_folder_path, dump_file_name)
            if not os.path.exists(self.rdd_folder_path):
                os.mkdir(self.rdd_folder_path)
            else:
                continue
            archive = py7zr.SevenZipFile(compressed_dump_file_path, mode="r")
            archive.extractall(path=wikipedia_dumps_folder)
            archive.close()
            decompressed_dump_file_path = os.path.join(wikipedia_dumps_folder, dump_file_name)
            decompressed_dump_file = io.open(decompressed_dump_file_path, "r", encoding="utf-8")
            logfile_name=file_name+".log"
            logging.basicConfig(filename=logfile_name, level=logging.INFO)
            page_text = ""
            for i,line in enumerate(decompressed_dump_file):
                line = line.strip()
                if line == "<page>":
                    page_text = ""
                page_text += "\n" + line
                if line == "</page>":
                    page_tree = bs4.BeautifulSoup(page_text, "html.parser")
                    self.page_folder=str(page_tree.id.text)
                    logging.info(self.page_folder)
                    #int(self.page_folder)<=54095879 or 
                    if sys.getsizeof(page_text)> 5000000000:
                        print('Page size: ', sys.getsizeof(page_text), ' byte')
                        print(self.page_folder, ':Page already parsed or the size of the page is big')
                        continue
                    else:
                        print(self.page_folder, 'is processing now')
                        print('Page size: ', sys.getsizeof(page_text), ' byte')
                        print('Start Time', datetime.datetime.now())
                        total_infobox_count=0
                        total_table_count=0
                        for revision_tag in page_tree.find_all("revision"):
                            self.revision_id_parent="root"
                            self.revision_id_current=revision_tag.find("id").text
                            try:
                                self.revision_id_parent=revision_tag.find("parentid").text
                            except Exception as e:
                                print('Exception: Parent Id: ', str(e))
                            revision_text = revision_tag.find("text").text
                            self.code =mwparserfromhell.parse(revision_text)
                            self.table=self.code.filter_tags(matches=lambda node: node.tag=="table")
                            if parsing_type=="table":
                                revision_table_count=self.table_parsing()
                                total_table_count=total_table_count+revision_table_count
                            elif parsing_type=="infobox":
                                revision_infobox_count=self.infobox_parsing()
                                total_infobox_count=total_infobox_count+revision_infobox_count
                            elif parsing_type=="both":
                                revision_table_count=self.table_parsing()
                                total_table_count=total_table_count+revision_table_count                               
                                revision_infobox_count=self.infobox_parsing()
                                total_infobox_count=total_infobox_count+revision_infobox_count                         
                        print('The processing of ', self.page_folder,' is finished')
                        print('End time', datetime.datetime.now())
                        if total_table_count>0:
                            print("The total number of table in this page(with revision): {}".format(total_table_count))
                        if total_infobox_count>0:
                            print("The total number of infobox in this page(with revision): {}".format(total_infobox_count))
            decompressed_dump_file.close()
            os.remove(decompressed_dump_file_path)
    def infobox_parsing(self):
        """
        This method will extract all infobox templates with revision
        """
        infobox_count=0
        templates = self.code.filter_templates()
        for temp in templates:
            json_list=[]
            if "Infobox" in temp.name:
                try:
                    self.revision_page_folder_path=os.path.join(self.rd_folder_path_infobox,self.page_folder)
                    if not os.path.exists(self.revision_page_folder_path):
                        os.mkdir(self.revision_page_folder_path)
                    infobox_folder=remove_markup(str(temp.name))
                    infobox_folder=infobox_folder.strip()
                    infobox_folder= re.sub('[^a-zA-Z0-9\n\.]', ' ', (str(infobox_folder)).lower())
                    revision_infobox_folder_path=os.path.join(self.revision_page_folder_path,infobox_folder)
                    if not os.path.exists(revision_infobox_folder_path):
                        os.mkdir(revision_infobox_folder_path)
                    json_list.append(str(temp))
                    json.dump(json_list, open(os.path.join(revision_infobox_folder_path, self.revision_id_parent + '_' + self.revision_id_current + ".json"), "w"))
                    print('Infobox caption: ', infobox_folder)
                    infobox_count=infobox_count+1
                except Exception as e:
                    print('Infobox Exception: ', str(e))
        return infobox_count
    def table_parsing(self):
        """
        This method will extract all table templates with revision
        """
        table_count=0
        if self.table:                       
            for tebil in self.table:
                json_list=[]
                try:
                    table_caption = wtp.parse(str(tebil)).tables[0].caption
                    table_folder_name=remove_markup(str(table_caption))
                    table_folder_name=table_folder_name.lower()
                    table_folder_name=table_folder_name.strip()
                except Exception as e:
                    print('Exception: table folder name or out of list in table', str(e))
                    continue                 
                if table_caption:
                  try:
                      self.revision_page_folder_path=os.path.join(self.rd_folder_path_table,self.page_folder)
                      if not os.path.exists(self.revision_page_folder_path):
                          os.mkdir(self.revision_page_folder_path)
                      table_folder_name=table_folder_name.strip('\n')
                      revision_table_folder_path=os.path.join(self.revision_page_folder_path,table_folder_name)
                      revision_table_folder_path=revision_table_folder_path.strip()
                      if not os.path.exists(revision_table_folder_path):
                          os.mkdir(revision_table_folder_path)
                  except Exception as e:
                      print('Exception: revision table folder', str(e))
                      continue
                  table_count=table_count+1
                  json_list.append(str(tebil))
                  json.dump(json_list, open(os.path.join(revision_table_folder_path, self.revision_id_parent + '_' + self.revision_id_current + ".json"), "w"))
                  print('Table caption: ', table_folder_name)
                  table_count=table_count+1                                    
        return table_count
    def extract_old_new_value(self, revision_data_folder, extract_type):
        table_folder_count=0
        infobox_folder_count=0
        unique_table_error_found=0
        unique_infobox_error_found=0
        self.table_count_with_error=0
        self.infobox_count_with_error=0
        table_count_with_revision=0
        infobox_count_with_revision=0
        unique_infobox_error_found=0
        rd_folder_path = revision_data_folder
        if extract_type=="table":
            modified_data_path=os.path.join('datasets','Table_for_creating_model')
        if extract_type=="infobox":
            modified_data_path=os.path.join('datasets','Infobox_for_creating_model')
        if not os.path.isdir(modified_data_path):
            os.mkdir(modified_data_path)
        for folder in os.listdir(rd_folder_path):
            print(folder)
            revision_list_table=[]
            revision_list_infobox=[]
            page_folder=os.path.join(rd_folder_path,folder) #datasets/revision-data/archieve-foldername/page_folder
            table_No=1
            infobox_No=1
            if os.path.isdir(os.path.join(rd_folder_path, folder)):
                for nested_folder in os.listdir(os.path.join(rd_folder_path,folder)):
                    if extract_type=="table":
                        table_folder_count=table_folder_count+1
                    if extract_type=="infobox":
                        infobox_folder_count=infobox_folder_count+1
                    if os.path.isdir(os.path.join(page_folder, nested_folder)):
                        filelist = os.listdir(os.path.join(page_folder, nested_folder))
                        if extract_type=="table:":
                            table_count_with_revision=table_count_with_revision+ len(filelist)
                        if extract_type=="infobox":
                            infobox_count_with_revision=infobox_count_with_revision+ len(filelist)
                        filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))#sort the json file list with revision id
                        self.previous_revision_file=None
                        self.previous_json_name=None
                        for rf in filelist:
                            self.current_json_name=rf
                            current_parent_id=rf.split('.')[0].split('_')[0]
                            if rf.endswith(".json"):                                      
                                self.current_revision_file=json.load(io.open(os.path.join(page_folder, nested_folder, rf), encoding="utf-8"))
                                if self.previous_revision_file and self.previous_json_name:                                              
                                    previous_revision_id= self.previous_json_name.split('.')[0].split('_')[1]                                               
                                    if previous_revision_id == current_parent_id:
                                        revision_list_from_fun_table=[]
                                        revision_list_from_fun_infobox=[]
                                        if extract_type=="table":
                                            revision_list_from_fun_table=self.diff_check_revision_table()
                                        if extract_type=="infobox":
                                            revision_list_from_fun_infobox=self.diff_check_revision_infobox()
                                        if revision_list_from_fun_table:
                                            revision_list_table.append(revision_list_from_fun_table)
                                        if revision_list_from_fun_infobox:
                                            revision_list_infobox.append(revision_list_from_fun_infobox)
                                        self.previous_revision_file=self.current_revision_file
                                        self.previous_json_name=self.current_json_name
                                else:
                                    self.previous_revision_file= self.current_revision_file
                                    self.previous_json_name=self.current_json_name                                   
                        if revision_list_table:
                            unique_table_error_found=unique_table_error_found+1
                            json.dump(revision_list_table, open(os.path.join(modified_data_path, folder + '_' + str(table_No) + ".json"), "w", encoding='utf8'))
                            table_No=table_No+1
                        if revision_list_infobox:
                            unique_infobox_error_found=unique_infobox_error_found+1
                            json.dump(revision_list_infobox, open(os.path.join(modified_data_path, folder + '_' + str(infobox_No) + ".json"), "w", encoding='utf8'))
                            infobox_No=infobox_No+1
        if extract_type=="table":                                           
            print('Total table: ', table_folder_count, ' Error Table: ', self.table_count_with_error)
            txt_file="table_folder_count: "+ str(table_folder_count) + " Error Table with revision: " + str(self.table_count_with_error) + " Unique table error found: " + str(unique_table_error_found)
            with open("table_count_summary.txt", "w") as text_file:
                text_file.write(txt_file)
        if extract_type=="infobox":
            print('Total infobox: ', infobox_folder_count,  ' Error Infobox: ', self.infobox_count_with_error)
            txt_file="Infobox_folder_count: "+ str(infobox_folder_count)+  " Error Infobox with revision: " + str(self.infobox_count_with_error) + " Unique infobox error found: " + str(unique_infobox_error_found)
            with open("infobox_count_summary.txt", "w") as text_file:
                text_file.write(txt_file)

    def diff_check_revision_table(self):
        create_revision_list=[]
        table_column_current=None
        table_column_previous=None
        code_current =mwparserfromhell.parse(self.current_revision_file[0], skip_style_tags=True)
        code_previous=mwparserfromhell.parse(self.previous_revision_file[0], skip_style_tags=True)
        try:
            # Current revision table  data extraction
            table1=code_current.filter_tags(matches=lambda node: node.tag=="table")
            table_code_current = wtp.parse(str(table1[0])).tables[0]
            table_data_current=table_code_current.data()
            table_column_current=table_data_current[0]
            # Previous revision table data extraction
            table2=code_previous.filter_tags(matches=lambda node: node.tag=="table")
            table_code_previous = wtp.parse(str(table2[0])).tables[0]
            table_data_previous=table_code_previous.data()
            table_column_previous=table_data_previous[0]
            df_data=DataFrame(table_data_previous)
            header=df_data.iloc[0]
            new_column_list=header.tolist()
            df_data=df_data[1:]
            df_data.columns=header
        except Exception as e:
            print('Exception from table data: ', str(e))
        if table_column_current and table_column_previous and len(table_column_previous) == len(set(table_column_previous)):
            self.table_count_with_error=self.table_count_with_error+1
            if len(table_column_current)==len(table_column_previous):
                text1=table_data_previous
                text2=table_data_current
                if text1 and text2:
                    for index1, (txt1, txt2) in enumerate(zip(text1,text2)): #row parsing
                        if index1==0:
                            continue
                        d = difflib.Differ()
                        for index, (cell1,cell2) in enumerate(zip(txt1,txt2)): # values of row parsing
                            create_revision_dict={}
                            old_value=None
                            new_value=None
                            try:
                                cell1=remove_markup(str(cell1))
                                cell2=remove_markup(str(cell2))
                            except Exception as e:
                                print('Exception from cell remove mark up: ', str(e))
                            cell1=cell1.strip()
                            cell2=cell2.strip()
                            if cell1 and cell2 :
                                diff1 = d.compare([''.join(cell1)], [cell2])
                                try:
                                    if diff1:
                                        for line in diff1:
                                            if not line.startswith(' '):
                                                if line.startswith('-'):
                                                    old_value=line[1:]
                                                if line.startswith('+'):
                                                    new_value=line[1:]
                                        if old_value and new_value:
                                            txt1=remove_markup(str(txt1))
                                            old_value=remove_markup(str(old_value))
                                            new_value=remove_markup(str(new_value))
                                            column_name=new_column_list[index]
                                            column_name=str(column_name)
                                            column_values=df_data[column_name].tolist()
                                            column_values=remove_markup(str(column_values))
                                            cleanr = re.compile('<.*?>')           
                                            all_column=list(df_data.columns)
                                            all_column = re.sub(cleanr, ' ', str(all_column))
                                            all_column=remove_markup(all_column)
                                            column_name=re.sub(cleanr, ' ', str(column_name))
                                            column_name=remove_markup(column_name)
                                            table_data_previous.pop(index1)
                                            if len(old_value)<50 and len(new_value)<50:
                                                create_revision_dict={ "columns": all_column,"dirty_table": table_data_previous, "domain": column_values, "vicinity": txt1, "errored_column": column_name,"old_value": old_value, "new_value": new_value}
                                                create_revision_list.append(create_revision_dict)
                                                print('column: ',column_name,'old_cell: ',old_value,  'new_cell: ', new_value)
                                except Exception as e:
                                    print('Exception from revised value: ', str(e))                 
        return create_revision_list

    def diff_check_revision_infobox(self):
        d = difflib.Differ()
        create_revision_list=[]
        value_current=None
        value_previous=None
        old_value=None
        new_value=None
        try:
            code_current =mwparserfromhell.parse(self.current_revision_file[0], skip_style_tags=True)
            code_previous=mwparserfromhell.parse(self.previous_revision_file[0], skip_style_tags=True)
            infobox_current_template=code_current.filter_templates()
            infobox_previous_template=code_previous.filter_templates()
        except Exception as e:
            print('Exception from template name: ', str(e))
        if 'Infobox' in infobox_current_template[0].name and 'Infobox' in infobox_previous_template[0].name:
            self.infobox_count_with_error=self.infobox_count_with_error+1
            try:
                current_infobox_data=[remove_markup(str(item1)) for item1 in infobox_current_template[0].params]
                previous_infobox_data=[remove_markup(str(item2)) for item2 in infobox_previous_template[0].params]
                key_value_current=current_infobox_data
                key_value_previous=previous_infobox_data
                for text1,text2 in zip(key_value_previous,key_value_current):
                    key_previous=text1.split('=')[0].strip()
                    key_current=text2.split('=')[0].strip()                   
                    if key_previous==key_current:
                        value_previous=text1.split('=')[1].strip()
                        value_current=text2.split('=')[1].strip()
                        if value_current and value_previous:
                            value_previous=remove_markup(str(value_previous))
                            value_current=remove_markup(str(value_current))
                            diff1 = d.compare([''.join(value_previous)], [value_current])
                            if diff1:
                                for line in diff1:
                                    if not line.startswith(' '):
                                        if line.startswith('-'):
                                            old_value=line[1:]
                                        if line.startswith('+'):
                                            new_value=line[1:]
                            if old_value and new_value and len(str(key_current))<12 and len(str(old_value))<50 and len(str(new_value))<50:
                                create_revision_dict={ "domain": key_current,"old_value": old_value, "new_value": new_value}
                                create_revision_list.append(create_revision_dict)
                                print('domain: ',key_current,'old_cell: ',old_value,  'new_cell: ', new_value)
            except Exception as e:
                print('Exception from infobox data: ', str(e))
                    
        return create_revision_list
    def evaluate_model (self,model_type,total_error, total_error_to_repair, total_correction):
        if total_error_to_repair==0:
            precision=0
        else:
            precision=total_correction/total_error_to_repair
        if total_error==0:
            recall=0
        else:
            recall=total_correction/total_error
        if (precision+recall)==0:
            f_score=0
        else:
            f_score=(2 * precision * recall) / (precision + recall)      
        logfile_name=model_type+".log"
        logging.basicConfig(filename=logfile_name, level=logging.INFO)
        performance_string="Time: "+ str(datetime.datetime.now()) + " Model Type: " + str(model_type) +" Precision: " +str(precision) +" Recall: "+ str(recall) +"F-score: " +str(f_score)
        logging.info(performance_string)
        print("Performance: {}\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(model_type,precision, recall, f_score))
    def train_domain_based_model_edit_distance(self,train_dataset,train_dataset_path, data_type, domain_type): #initial location, date
        domain_location=None
        location_corpus=[]
        if domain_type=="location":
            domain_location=['Country', 'COUNTRY', 'country', 'CITY', 'City','city','Location','LOCATION','location','Place','PLACE','place','VENUE','venue','Venue','Town','town','TOWN', 'birth_place','death_place']       
        for rf in train_dataset:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(train_dataset_path, rf), encoding="utf-8"))
                    one_item=revision_list[-1]
                    if domain_location:
                        if one_item[0]['errored_column'] in domain_location:                                                 
                            column_values=one_item[0]['domain']
                            column_values=re.sub("[^\w]", " ",  column_values).split()
                            old_value=str(one_item[0]['old_value'].strip())
                            if old_value and not old_value.isdigit():
                                new_value=str(one_item[0]['new_value'].strip())
                                column_values.remove(old_value)
                                column_values.append(new_value)
                            location_corpus.extend(column_values)
                except Exception as e:
                    print(e)
                    continue
        location_corpus = [s.strip() for s in location_corpus if not s.isdigit()]
        corpus = Counter(location_corpus)
        spell_corrector = SpellCorrector(dictionary=corpus, verbose=1)
        with open("model/edit_distance_domain_location.pickle","wb") as f:
            pickle.dump(spell_corrector, f)
    def train_general_model_edit_distance(self,train_dataset,train_dataset_path, data_type):#data_taype: table or infobox #now I am working only on table, data_type is not usefulnow
        #general_corpus=[]
        train_data_rows=[]
        for rf in train_dataset:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(train_dataset_path, rf), encoding="utf-8"))    
                    one_item=revision_list[-1]
                    #old_value=str(one_item[0]['old_value'].strip())
                    #new_value=str(one_item[0]['new_value'].strip())
                    dirty_table=one_item[0]['dirty_table']
                    for index, row in enumerate(dirty_table):
                        if index==0:
                            continue
                        row=remove_markup(str(row))
                        row= ast.literal_eval(row)
                        row=list(filter(None, row))
                        row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                        if row:
                            row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                            train_data_rows.extend(row)
                except Exception as e:
                    print('Exception: ',str(e))     
        general_corpus = [s.strip() for s in train_data_rows if not s.isdigit()]
        corpus = Counter(general_corpus)
        spell_corrector = SpellCorrector(dictionary=corpus, verbose=1)
        with open("model/edit_distance_general.pickle","wb") as f:
            pickle.dump(spell_corrector, f)
    def error_correction_edit_distance(self,datasets_type,dataset_1,dataset_2, model_type,domain_type):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="edit_distance_general":
                if datasets_type=="wiki":
                    error_correction=self.prepare_testing_datasets_wiki(dataset_1,dataset_2) #dataset1 : json_list, dataset_1: path of json_filelist
                if datasets_type=="real_world":
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
            if model_type=="edit_distance_domain":
                if datasets_type=="wiki":
                    error_correction=self.prepare_domain_testing_datasets_wiki(dataset_1,dataset_2,domain_type) #dataset1 : json_list, dataset_1: path of json_filelist
                    total_error=self.calculate_total_error_wiki(dataset_1,dataset_2)
                if datasets_type=="real_world":
                    total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                    error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                    #print(error_correction)
                with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file: #edit_distance_domain_location
                    model_edit_distance = pickle.load(pickle_file)
        except Exception as e:
            print('Exception: ',str(e))
        
        for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
            try:    
                if model_type=="edit_distance_domain":
                    pass
                else:
                    total_error=total_error+1
                if not any(x1.isdigit() for x1 in error_value):
                    total_error_to_repaired=total_error_to_repaired+1    
                    first=model_edit_distance.correction(error_value)
                    first=first.lower()
                    actual_value=actual_value.lower()
                    #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print("Exception : ", str(e))
        model_type=model_type+" "+ datasets_type+" Not retrain"
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def error_correction_edit_distance_retrain(self,datasets_type,dataset_1,dataset_2, model_type):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="edit_distance_general":
                if datasets_type=="wiki":
                    error_correction=self.prepare_testing_datasets_wiki(dataset_1,dataset_2) #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_datasets_retrain_wiki(dataset_1,dataset_2)
                if datasets_type=="real_world":
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
                    #total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
            if model_type=="edit_distance_domain":
                if datasets_type=="real_world":
                    error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
                    total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                if datasets_type=="wiki":
                    error_correction=self.prepare_domain_testing_datasets_wiki(dataset_1,dataset_2,"location") #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_datasets_retrain_wiki(dataset_1,dataset_2)
                    total_error=self.calculate_total_error_wiki(dataset_1,dataset_2)
                with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
        except Exception as e:
            print('Model Error: ',str(e))
        if datasets_type=="real_world":  
            train_data_rows=[]   
            data_for_retrain=data_for_retrain.values.tolist()
            for row in data_for_retrain:
                row = list(map(str, row))
                row=list(filter(None, row))
                train_data_rows.extend(row)
        else:
            train_data_rows=data_for_retrain
        if train_data_rows:
            dict1=model_edit_distance.dictionary
            general_corpus = [str(s) for s in train_data_rows]
            corpus = Counter(general_corpus)
            corpus.update(dict1)
            model_edit_distance = SpellCorrector(dictionary=corpus, verbose=1)
        total_p=0
        total_error_to_repaired=0
        for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
            total_p=total_p+1
            print('total process: ', total_p)
            try:
                if model_type=="edit_distance_domain":
                    pass
                else:
                    total_error=total_error+1
                if  len(error_value)<30:
                    
                    #print("total_error_to_repaired ", total_error_to_repaired)
                    error_value=str(error_value)
                    error_value= error_value.strip()
                    #flag_textual=textual_value()
                    first,prob=model_edit_distance.correction(error_value)
                    #first=first.l
                    #actual_value=actual_value.lower()
                    #print('Before Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    #if type(first) !="int and type(actual_value) not int:
                    actual_value=str(actual_value)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    flag_textual=textual_value(actual_value)
                    if error_value==first  or prob==0 or first=="":
                        continue
                    else:
                        if flag_textual and first:
                            total_error_to_repaired=total_error_to_repaired+1
                            print("total_error_to_repaired ", total_error_to_repaired)
                            if first==actual_value:
                                #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                                total_repaired=total_repaired+1
                                print(total_repaired)
            except Exception as e:
                print('Exception: ', str(e))
                continue
        print(total_error,total_error_to_repaired,total_repaired )
        model_type=model_type+" "+ datasets_type+" Retrain"
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    
 
    def split_train_test_data(self,json_file_list):
       random.shuffle(json_file_list)
       training = json_file_list[:int(len(json_file_list)*0.9)]
       testing = json_file_list[-int(len(json_file_list)*0.1):]
       return training, testing #tr,tt=spilt()
    def train_fasttext_all_domain(self,train_dataset,rd_folder_path):
        train_data_rows=[]
        max_shape=0
        for rf in train_dataset:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    one_item=revision_list[-1]
                    old_value=str(one_item[0]['old_value'].strip())
                    new_value=str(one_item[0]['new_value'].strip())
                    dirty_table=one_item[0]['dirty_table']
                    for index, row in enumerate(dirty_table):
                        if index==0:
                            continue
                        shape=len(row)
                        if shape>max_shape:
                            max_shape=shape
                        row=remove_markup(str(row))
                        row= ast.literal_eval(row)
                        row=list(filter(None, row))
                        row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                        if row:
                            row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                            train_data_rows.append(row)
                except Exception as e:
                    print('Exception: ',str(e))                  
        if train_data_rows:
            model_fasttext = FastText(train_data_rows, min_count=1, workers=8, iter=500, window=max_shape, sg=1)
            model_fasttext.save("model/Fasttext_All_Domain.w2v")
    def train_fastext_all_domain_cv(self,file_list,rd_folder_path):
        fold=1
        kf5 = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf5.split(file_list):
            print('Processing ..... Fold: ',fold)
            train_data_rows=[]
            shape_l=[]
            for train_id in train_index:
                rf=file_list[train_id]
                if rf.endswith(".json"):
                    try:
                        revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                        one_item=revision_list[-1]
                        old_value=str(one_item[0]['old_value'].strip())
                        new_value=str(one_item[0]['new_value'].strip())
                        dirty_table=one_item[0]['dirty_table']
                        for index, row in enumerate(dirty_table):
                            if index==0:
                                continue                           
                            shape_len=len(row)
                            shape_l.append(shape_len)
                            row=remove_markup(str(row))
                            row= ast.literal_eval(row)
                            row=list(filter(None, row))
                            #remove all digit 
                            row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                            if row:
                                row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                                train_data_rows.append(row)
                    except Exception as e:
                        pass
            shape=max(shape_l)
            print(shape)
            if train_data_rows:
                model_fasttext = FastText(train_data_rows, min_count=1, workers=8, iter=500, window=shape, sg=1)
                model_fasttext.save("model/Fasttext_CV_Fold.w2v")
            total_error=0
            total_error_to_repaired=0
            total_repaired=0
            for test_id in test_index:
                rft=file_list[test_id]
                if rft.endswith(".json"):
                    try:
                        revision_list = json.load(io.open(os.path.join(rd_folder_path, rft), encoding="utf-8"))
                        for test_item in revision_list:
                            old_value=str(test_item[0]['old_value'].strip())
                            new_value=str(test_item[0]['new_value'].strip())
                            total_error=total_error+1
                            if not any(x1.isdigit() for x1 in old_value):
                                total_error_to_repaired=total_error_to_repaired+1    
                                similar_value=model_fasttext.most_similar(old_value)
                                first,b=similar_value[0]
                                first=first.lower()
                                new_value=new_value.lower()
                                print('Error : ', old_value, ' Fixed: ', first, ' Actual: ', new_value)
                                if first==new_value:
                                    total_repaired=total_repaired+1                      
                    except Exception as e:
                        pass
            print('Total error',total_error,'total repaired: ', total_repaired)
            if total_error_to_repaired==0:
                precision=0
            else:
                precision=total_repaired/total_error_to_repaired
            if total_error==0:
                recall=0
            else:
                recall=total_repaired/total_error
            if (precision+recall)==0:
                f_score=0
            else:
                f_score=(2 * precision * recall) / (precision + recall)
            print("Fold {}: performance wiki testing:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format( fold,precision, recall, f_score))
    def train_domain_based_fasttext_model(self,domain_type,train_dataset,rd_folder_path):
        train_data_rows=[]
        max_shape=0
        if domain_type=="location":
            domain_location=['Country', 'COUNTRY', 'country', 'CITY', 'City','city','Location','LOCATION','location','Place','PLACE','place','VENUE','venue','Venue','Town','town','TOWN', 'birth_place','death_place']      
        for rf in train_dataset:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    one_item=revision_list[-1]
                    if domain_location:
                        if one_item[0]['errored_column'] in domain_location:                                                 
                            old_value=str(one_item[0]['old_value'].strip())
                            new_value=str(one_item[0]['new_value'].strip())
                            dirty_table=one_item[0]['dirty_table']
                            for index, row in enumerate(dirty_table):
                                if index==0:
                                    continue
                                shape=len(row)
                                if shape>max_shape:
                                    max_shape=shape
                                row=remove_markup(str(row))
                                row= ast.literal_eval(row)
                                row=list(filter(None, row))
                                row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                                if row:
                                    row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                                    train_data_rows.append(row)
                except Exception as e:
                    print('Exception: ',str(e))
        if train_data_rows:
            model_fasttext = FastText(train_data_rows, min_count=1, workers=8, iter=500, window=max_shape, sg=1)
            model_fasttext.save("model/Fasttext_Location_Domain.w2v")
    def error_correction_fasttext(self,model_type,datasets_type,dataset_1,dataset_2): #model=edit distance, fasttext, dataset_type=wiki_realword
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_All_Domain":
                if datasets_type=="wiki":
                    error_correction=self.prepare_testing_datasets_wiki(dataset_1,dataset_2) #dataset1 : json_list, dataset_1: path of json_filelist
                if datasets_type=="real_world":
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_Domain_Location":
                if datasets_type=="wiki":
                    error_correction=self.prepare_domain_testing_datasets_wiki(dataset_1,dataset_2,"location") #dataset1 : json_list, dataset_1: path of json_filelist
                    total_error=self.calculate_total_error_wiki(dataset_1,dataset_2)
                if datasets_type=="real_world":
                    error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                    total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
            if model_type=="Fasttext_CV_Fold":
                model_fasttext=FastText.load("model/Fasttext_CV_Fold.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
            if model_type=="Fasttext_Domain_Location":
                pass
            else:
                total_error=total_error+1
            try:
                if not any(x1.isdigit() for x1 in error_value):
                    total_error_to_repaired=total_error_to_repaired+1    
                    similar_value=model_fasttext.most_similar(error_value)
                    first,b=similar_value[0]
                    first=first.lower()
                    actual_value=actual_value.lower()
                    first=first.strip()
                    actual_value=actual_value.strip()
                    #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    if first==actual_value:
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
                continue
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def error_correction_fasttext_with_retrain_wiki(self,model_type,datasets_type,dataparam_1,dataparam_2):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        if model_type=="Fasttext_All_Domain": #every time it will load the pretrained model to test new wiki table
            error_correction=self.prepare_testing_datasets_wiki(dataparam_1,dataparam_2) #dataparam_1 : json_list, dataparam_2: path of json_filelist
            model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
        if model_type=="Fasttext_CV_Fold":
            model_fasttext=FastText.load("model/Fasttext_CV_Fold.w2v")
        if model_type=="Fasttext_Domain_Location":
            model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
            error_correction=self.prepare_domain_testing_datasets_wiki(dataparam_1,dataparam_2,"location")
            total_error=self.calculate_total_error_wiki(dataparam_1,dataparam_2)
        if datasets_type=="wiki":  
            train_data_rows=[]
            for rf in dataparam_1:               
                if rf.endswith(".json"):
                    try:
                        revision_list = json.load(io.open(os.path.join(dataparam_2, rf), encoding="utf-8"))    
                        one_item=revision_list[-1]
                        old_value=str(one_item[0]['old_value'].strip())
                        new_value=str(one_item[0]['new_value'].strip())
                        vicinity=one_item[0]['vicinity']
                        vicinity=remove_markup(str(vicinity))
                        vicinity= ast.literal_eval(vicinity)
                        #print('Before: ',vicinity)
                        train_vicinity_index = vicinity.index(old_value)
                        del vicinity[train_vicinity_index]
                        vicinity.append(new_value)
                        vicinity = [x for x in vicinity if not any(x1.isdigit() for x1 in x)]
                        vicinity=[x for x in vicinity if len(x)!=0] #remove empty item from list
                        #vicinity=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in vicinity]
                        #print('After: ', vicinity)
                        #row=list(filter(None, row))
                        dirty_table=one_item[0]['dirty_table']
                        for index, row in enumerate(dirty_table):
                            if index==0:
                                continue
                            shape=len(row)
                            row=remove_markup(str(row))
                            row= ast.literal_eval(row)
                            row=list(filter(None, row))
                            #remove all digit 
                            row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                            row=[x for x in row if len(x)!=0] #remove empty item from list
                            if row:
                                row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                                train_data_rows.append(row)
                    except Exception as e:
                        print('Exception: ',str(e))
            if train_data_rows:             
                model_fasttext.build_vocab(train_data_rows, update=True)
                model_fasttext.train(sentences=train_data_rows, total_examples = len(train_data_rows), epochs=5)
            for error_value, actual_value in zip(error_correction['error'],error_correction['actual']):
                try:
                    if model_type=="Fasttext_Domain_Location":
                        pass
                    else:
                        total_error=total_error+1
                    
                    if not any(x1.isdigit() for x1 in error_value):
                        total_error_to_repaired=total_error_to_repaired+1    
                        similar_value=model_fasttext.most_similar(error_value)
                        #print('Actual value: ',  actual_value,'Most similar value of : ',error_value, ' ' , similar_value)
                        first,b=similar_value[0]
                        #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                        first=first.strip()
                        actual_value=actual_value.strip()
                        if first==actual_value:
                            print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                            total_repaired=total_repaired+1  
                except: 
                    continue                                     
        print(total_error,total_error_to_repaired, total_repaired )             
        model_type=model_type+' retrain wiki '
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def error_correction_fasttext_with_retrain_realworld(self,datasets_type,dataset_1,dataset_2, model_type):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        try:
            if model_type=="Fasttext_All_Domain":
                if datasets_type=="real_world":
                    data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="Fasttext_Domain_Location":
                data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
                error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
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
        dirty_data=pd.read_csv(dataset_2)
        dirty_row=[]
        for error_value, actual_value,index in zip(error_correction['error'],error_correction['actual'],error_correction['index']):
            dirty_row=[]
            error_value=str(error_value)
            dirty_row.append(dirty_data.at[index,'city'])
            #dirty_row.append(dirty_data.at[index,'state'])
            #dirty_row.append(dirty_data.at[index,'zip'])
            #dirty_row.append(dirty_data.at[index,'has_child'])
            #dirty_row.append(dirty_data.at[index,'gender'])
            dirty_row.append(str(dirty_data.at[index,'zip']))
            #area_code_str=str(dirty_data.at[index,'f_name'])
            #dirty_row.append(area_code_str)
            if error_value in dirty_row:
                dirty_row.remove(error_value)
            print(dirty_row)
            error_value=str(error_value)

            #error_value=error_value + "907"
            if model_type=="Fasttext_Domain_Location" and datasets_type=="real_world":
                pass
            else:
                total_error=total_error+1
            try:
                if len(error_value)<30:
                    total_error_to_repaired=total_error_to_repaired+1    
                    similar_value=model_fasttext.most_similar(error_value)
                    #similar_value=model_fasttext.most_similar(positive=dirty_row, negative=[error_value])

                   #positive=['baghdad', 'england'], negative=['london']
                    
                    first,b=similar_value[0]
                    #print('Before Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    actual_value=str(actual_value)
                    first=first.lower()
                    actual_value=actual_value.lower()
                    
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        #print('After Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
                continue
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)
    def prepare_testing_datasets_wiki(self, file_list_wiki,rd_folder_path):
        total_data=0
        actual_error = pd.DataFrame(columns = ['actual', 'error'])
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    for one_item in revision_list:
                        old_value=str(one_item[0]['old_value'].strip())
                        old_value=remove_markup(str(old_value))
                        old_value=re.sub('[^a-zA-Z0-9.-]+', ' ', old_value)
                        old_value=old_value.strip()

                        new_value=str(one_item[0]['new_value'].strip())
                        new_value=remove_markup(str(new_value))
                        new_value=re.sub('[^a-zA-Z0-9.-]+', ' ', new_value)
                        new_value=new_value.strip()
                        if  old_value and new_value and old_value !=" " and new_value!=" " and len(old_value)>3 and len(new_value)>3 and old_value!="none" and new_value!="none" and old_value!="None" and new_value!="None":
                            actual_error.loc[-1] = [new_value, old_value]
                            actual_error.index = actual_error.index + 1  # shifting index
                            actual_error = actual_error.sort_index()
                            total_data=total_data+1
                except Exception as e:
                    print('Exception from wiki: ', str(e))
        print("total_data: ",total_data)
        return actual_error
    def prepare_domain_testing_datasets_wiki(self, file_list_wiki,rd_folder_path,domain_type):
        total_data=0
        if domain_type=="location":
            domain_location=['Country', 'COUNTRY', 'country', 'CITY', 'City','city','Location','LOCATION','location','Place','PLACE','place','VENUE','venue','Venue','Town','town','TOWN', 'birth_place','death_place']
        actual_error = pd.DataFrame(columns = ['actual', 'error'])
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    for one_item in revision_list:
                        if domain_location:
                            if one_item[0]['errored_column'] in domain_location: 
                                old_value=str(one_item[0]['old_value'].strip())
                                old_value=remove_markup(str(old_value))
                                old_value=re.sub('[^a-zA-Z0-9.-]+', ' ', old_value)
                                old_value=old_value.strip()
                                new_value=str(one_item[0]['new_value'].strip())
                                new_value=remove_markup(str(new_value))
                                new_value=re.sub('[^a-zA-Z0-9.-]+', ' ', new_value)
                                new_value=new_value.strip()
                                if old_value and new_value and old_value !=" " and new_value!=" " and len(old_value)>3 and len(new_value)>3 and old_value!="none" and new_value!="none" and old_value!="None" and new_value!="None":
                                    actual_error.loc[-1] = [new_value, old_value]
                                    actual_error.index = actual_error.index + 1  # shifting index
                                    actual_error = actual_error.sort_index()
                                    total_data=total_data+1
                except Exception as e:
                    print('Exception from wiki: ', str(e))
        print("Total data to repair: ", total_data)
        return actual_error
    def prepare_domain_testing_datasets_realworld(self, clean_data_path, dirty_data_path):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index'])
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        #Hospital
        #clean_data_col=['City','State','ZipCode']
        #dirty_data_col=['city','state','zip']
        clean_data_col=['city','state']
        dirty_data_col=['city','state']
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                value1=str(value1)
                value2=str(value2)
                if value1==value2:
                    continue
                else:
                    #value3=sub_df.iloc[indx]['A']
                    #value4=sub_df.iloc[indx]['A']
                    #value5=sub_df.iloc[indx]['A']
                    actual_error.loc[-1] = [value1, value2, indx]
                    actual_error.index = actual_error.index + 1  # shifting index
                    actual_error = actual_error.sort_index()
        return actual_error
    def calculate_total_error_realworld(self,clean_data_path, dirty_data_path):
        total_error=0
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
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
    def calculate_total_error_wiki(self,file_list_wiki,rd_folder_path):
        total_error=0
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))
                    for one_item in revision_list:
                        old_value=str(one_item[0]['old_value'].strip())
                        if old_value:
                            total_error=total_error+1                   
                except Exception as e:
                    print('Exception: ',str(e))  
        return total_error

    def prepare_testing_datasets_real_world(self,clean_data_path, dirty_data_path):
        actual_error = pd.DataFrame(columns = ['actual', 'error','index'])
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data_col=clean_data.columns.values
        dirty_data_col=dirty_data.columns.values
        for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
            for indx, (value1, value2) in enumerate(zip(clean_data[clean_col],dirty_data[dir_col])):
                value1=str(value1)
                value2=str(value2)
                if value1==value2:
                    continue
                else:
                    actual_error.loc[-1] = [value1, value2,indx]
                    actual_error.index = actual_error.index + 1  # shifting index
                    actual_error = actual_error.sort_index()
        return actual_error
    def prepare_dataset_for_retrain_realworld(self, clean_data_path, dirty_data_path): #send two datasets
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data = clean_data.applymap(str)
        dirty_data = dirty_data.applymap(str)
        if clean_data.shape != dirty_data.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        else:
            clean_data_col=clean_data.columns.values
            #Hospital
            #clean_data_col=['Address1','City','State','CountyName']
            #dirty_data_col=['address_1','city','state','county']
            #clean_data_col=['city','state']
            #dirty_data_col=['city','state']
            dirty_data_col=dirty_data.columns.values
            for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = None #replace error value with NaN
        return dirty_data
    def prepare_dataset_for_retrain_realworld_domain(self, clean_data_path, dirty_data_path): #send two datasets do not use this for good result
        clean_data=pd.read_csv(clean_data_path)
        dirty_data=pd.read_csv(dirty_data_path)
        clean_data = clean_data.applymap(str)
        dirty_data = dirty_data.applymap(str)
        if clean_data.shape != dirty_data.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        else:
            clean_data_col=clean_data.columns.values
            #Hospital
            #clean_data_col=['Address1','City','State','CountyName']
            #dirty_data_col=['address_1','city','state','county']
            clean_data_col=['city','state']
            dirty_data_col=['city','state']
            dirty_data_col=dirty_data.columns.values
            for dir_col, clean_col in zip(dirty_data_col, clean_data_col):
                dirty_data.loc[dirty_data[dir_col] != clean_data[clean_col], dir_col] = None #replace error value with NaN
        dirty_data = dirty_data[["area_code", "city","state"]]
        return dirty_data
    def prepare_datasets_retrain_wiki(self,file_list_wiki,rd_folder_path): ###only for edit distance
        train_data_rows=[]
        for rf in file_list_wiki:
            if rf.endswith(".json"):
                try:
                    revision_list = json.load(io.open(os.path.join(rd_folder_path, rf), encoding="utf-8"))    
                    one_item=revision_list[-1]
                    #old_value=str(one_item[0]['old_value'].strip())
                    #new_value=str(one_item[0]['new_value'].strip())
                    dirty_table=one_item[0]['dirty_table']
                    for index, row in enumerate(dirty_table):
                        if index==0:
                            continue
                        row=remove_markup(str(row))
                        row= ast.literal_eval(row)
                        row=list(filter(None, row))
                        row = [x for x in row if not any(x1.isdigit() for x1 in x)]
                        if row:
                            row=[re.sub('[^a-zA-Z0-9.-]+', ' ', _) for _ in row]
                            train_data_rows.extend(row)
                except Exception as e:
                    print('Exception: ',str(e))  
        return train_data_rows
    def retrain_edit_fasttext_realworld_domain(self,datasets_type,dataset_1,dataset_2, model_type):
        total_error=0
        total_error_to_repaired=0
        total_repaired=0
        #data_for_retrain=[]
        actual_error_fixed = pd.DataFrame(columns = ['actual', 'error','fixed','index'])
        try:
            if model_type=="general":
                if datasets_type=="wiki":
                    error_correction=self.prepare_testing_datasets_wiki(dataset_1,dataset_2) #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_datasets_retrain_wiki(dataset_1,dataset_2)
                if datasets_type=="real_world":
                    error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
                    #total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                with open("model/edit_distance_general.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
            if model_type=="domain":
                if datasets_type=="real_world":
                    error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2) # for hospital domain, chose, domain hospital training
                    total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                if datasets_type=="wiki":
                    error_correction=self.prepare_domain_testing_datasets_wiki(dataset_1,dataset_2,"location") #dataset_1 clean data for real world
                    data_for_retrain=self.prepare_datasets_retrain_wiki(dataset_1,dataset_2)
                    total_error=self.calculate_total_error_wiki(dataset_1,dataset_2)
                with open("model/edit_distance_domain_location.pickle", 'rb') as pickle_file:
                    model_edit_distance = pickle.load(pickle_file)
        
            #data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
            if datasets_type=="real_world":  
                train_data_rows=[]   
                data_for_retrain=data_for_retrain.values.tolist()
                for row in data_for_retrain:
                    row = list(map(str, row))
                    row=list(filter(None, row))
                    train_data_rows.extend(row)
            else:
                train_data_rows=data_for_retrain
            if train_data_rows:
                dict1=model_edit_distance.dictionary
                general_corpus = [str(s) for s in train_data_rows]
                corpus = Counter(general_corpus)
                corpus.update(dict1)
                model_edit_distance = SpellCorrector(dictionary=corpus, verbose=1)
            total_p=0
            total_error_to_repaired=0
        except Exception as e:
            print('Model Error: ',str(e))
        for error_value, actual_value, index in zip(error_correction['error'],error_correction['actual'], error_correction['index']):
            total_p=total_p+1
            print('total process: ', total_p)
            try:
                if model_type=="edit_distance_domain":
                    pass
                else:
                    total_error=total_error+1
                if  len(error_value)<30:
                    total_error_to_repaired=total_error_to_repaired+1
                    print("total_error_to_repaired ", total_error_to_repaired)
                    error_value=str(error_value)  
                    first=model_edit_distance.correction(error_value)
                    #first=first.l
                    #actual_value=actual_value.lower()
                    #print('Before Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    #if type(first) !="int and type(actual_value) not int:
                    actual_value=str(actual_value)
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        #print('Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                        total_repaired=total_repaired+1
                        print(total_repaired)
                   
                    actual_error_fixed.loc[-1] = [actual_value, error_value,first,index]
                    actual_error_fixed.index = actual_error_fixed.index + 1  # shifting index
                    actual_error_fixed = actual_error_fixed.sort_index()
            except Exception as e:
                print('Exception: ', str(e))
                continue
        print(total_error,total_error_to_repaired,total_repaired)
        print("Fasttext Starts: ")
        try:
            if model_type=="general":
                if datasets_type=="real_world":
                    data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
                    #error_correction=self.prepare_testing_datasets_real_world(dataset_1,dataset_2) #dataset_1 clean data for real world
                model_fasttext=FastText.load("model/Fasttext_All_Domain.w2v")
            if model_type=="domain":
                #data_for_retrain=self.prepare_dataset_for_retrain_realworld_domain(dataset_1,dataset_2)
                #error_correction=self.prepare_domain_testing_datasets_realworld(dataset_1,dataset_2) #dataset_1 clean data for real world
                #total_error=self.calculate_total_error_realworld(dataset_1,dataset_2)
                model_fasttext=FastText.load("model/Fasttext_Location_Domain.w2v")
        except Exception as e:
            print('Model Error: ',str(e))
        data_for_retrain=self.prepare_dataset_for_retrain_realworld(dataset_1,dataset_2)
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
        dirty_data=pd.read_csv(dataset_2)
        dirty_row=[]
        for error_value, actual_value,fixed_value,index in zip(actual_error_fixed['error'],actual_error_fixed['actual'],actual_error_fixed['fixed'],actual_error_fixed['index']):
            dirty_row=[]
            error_value=str(error_value)
            #dirty_row.append(str(dirty_data.at[index,'area_code']))
            #dirty_row.append(dirty_data.at[index,'state'])
            #dirty_row.append(dirty_data.at[index,'city'])
            #dirty_row.append(dirty_data.at[index,'has_child'])
            #dirty_row.append(dirty_data.at[index,'gender'])
            #dirty_row.append(str(dirty_data.at[index,'zip']))
            #area_code_str=str(dirty_data.at[index,'f_name'])
            #dirty_row.append(area_code_str)
            #if error_value in dirty_row:
            #    dirty_row.remove(error_value)
            #print(dirty_row)
            error_value=str(error_value)

            #error_value=error_value + "907"
            try:
                if error_value==fixed_value:
                    #total_error_to_repaired=total_error_to_repaired+1    
                    similar_value=model_fasttext.most_similar(error_value)
                    #similar_value=model_fasttext.most_similar(positive=dirty_row, negative=[error_value])

                   #positive=['baghdad', 'england'], negative=['london']
                    
                    first,b=similar_value[0]
                    #print('Before Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                    actual_value=str(actual_value)
                    first=first.lower()
                    actual_value=actual_value.lower()
                    
                    first=first.strip()
                    actual_value=actual_value.strip()
                    if first==actual_value:
                        #print('After Error : ', error_value, ' Fixed: ', first, ' Actual: ', actual_value)
                        total_repaired=total_repaired+1
            except Exception as e:
                print('Error correction model: ',str(e))
                continue
        
         

        model_type=model_type+" "+ datasets_type+" Retrain: "+ "Fasttext on top of  edit"
        print(total_error,total_error_to_repaired,total_repaired)
        self.evaluate_model(model_type,total_error,total_error_to_repaired,total_repaired)

if __name__ == "__main__":
    app = Jharu()
    #For table parsing, uncommnet the below line
    #app.extract_revisions(wikipedia_dumps_folder="datasets", parsing_type="table") #works well
    #
    #For infobox parsing, uncommnet the below line
    #app.extract_revisions(wikipedia_dumps_folder="datasets", parsing_type="infobox") #works well
    #
    #For parsing infobox and table at a time, uncommnet the below line
    #app.extract_revisions(wikipedia_dumps_folder="datasets", parsing_type="both")
    #     
    #Create old_new value from table
    #app.extract_old_new_value(revision_data_folder="datasets/revision-data/table",extract_type="table") #works well
    #
    #Create old_new value from infobox
    #app.extract_old_new_value(revision_data_folder="datasets/revision-data/infobox",extract_type="infobox") #should work well too

   
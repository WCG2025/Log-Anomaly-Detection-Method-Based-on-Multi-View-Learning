import pandas as pd
import csv

def load_parsed_bgl():
    parsed_bgl = pd.read_csv('/home/jpy/graduation_design_final/BGL/BGL_2k.log_structured.csv')
    parsed_bgl["Label"] = parsed_bgl["Label"].apply(lambda x: int(x != "-"))
    labels = parsed_bgl['Label'].tolist()
    contents = parsed_bgl['Content'].to_list()
    eventtemplates = parsed_bgl['EventTemplate'].to_list()
    label_content_tuples = list(zip(labels,contents,eventtemplates))

    return label_content_tuples

def load_raw_bgl():
    raw_bgl=pd.read_csv('/home/jpy/graduation_design_final/BGL/BGL.log_structured.csv')
    raw_bgl["Label"]=raw_bgl["Label"].apply(lambda x: int(x != "-"))
    labels = raw_bgl['Label'].tolist()
    contents = raw_bgl['Content'].to_list()
    label_content_tuples =list(zip(labels,contents))

    return label_content_tuples
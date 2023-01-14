import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import glob

#Start to create the Dataframe by using Data that can be imported and merged automaticly
path = "Data"
file_path = glob.glob(path + "/*.csv")

#Using encoding for German letters 
list_of_df_files = map(lambda file_path: pd.read_csv(file_path, sep = ";", encoding = 'utf-8'), file_path)
list_of_df_files = list(list_of_df_files)

#Checking if imports went well
'''
for i in list_of_df_files:
  i.info()
'''

#without endcoding some more cleaning is necessary
'''
for i in list_of_df_files:
  i.columns= i.columns.str.replace('[ï»¿]', '')
'''
#creating a function to merge my dataframes
def merge_df(dataframes, column_names):
    result = dataframes[0]
    for i in range(1, len(dataframes)):
        result = pd.merge(result, dataframes[i], on=column_names)
    return result

#merging acctual dataframes on municipality_id and municipality name 
columns = ("municipality_id",	"municipality_name")
data_df = merge_df(list_of_df_files, columns)

#importing csv, which needed some more cleaning before merging
covid_doses_municipalities = pd.read_csv("COVID19_vaccination_municipalities_doses_timeline.csv", sep = ";")
covid_doses_municipalities_df = pd.DataFrame(covid_doses_municipalities)
mask_last_dates = covid_doses_municipalities_df["date"] == "2022-02-01" #using only newest data
covid_doses_municipalities_df = covid_doses_municipalities_df[mask_last_dates]
covid_doses_municipalities_df= covid_doses_municipalities_df.drop("date", axis = 1 )
covid_doses_municipalities_df = covid_doses_municipalities_df.reset_index()
covid_doses_municipalities_df = covid_doses_municipalities_df.drop("index", axis =1)
#covid_doses_municipalities_df.info()

#merge the new df to the other 
new_list_to_merge = (data_df, covid_doses_municipalities_df)
data_df = merge_df(new_list_to_merge, "municipality_id")
#data_df.info()

antivax_docs = pd.read_csv("antivax_docs.csv", sep = ";")
antivax_docs_df = pd.DataFrame(antivax_docs)
antivax_docs_df= antivax_docs_df.groupby("municipality_id", group_keys=False).sum().reset_index()
antivax_docs_df.columns
#antivax_docs_df.info())

#merge and fill municipalities with no antivax docs with 0 
data_df = data_df.merge(antivax_docs_df,how='left', left_on='municipality_id', right_on='municipality_id')
data_df.fillna(0)
data_df.info()







import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn as sk

# Start to create the Dataframe by using Data that can be imported and merged automaticly
path = "Data"
file_path = glob.glob(path + "/*.csv")

# Using encoding for German letters
list_of_df_files = map(lambda file_path: pd.read_csv(
    file_path, sep=";", encoding='utf-8', decimal=",", thousands="."), file_path)
list_of_df_files = list(list_of_df_files)
# print(list_of_df_files)

# Checking if imports went well
'''
for i in list_of_df_files:
  i.info()
'''

# without endcoding some more cleaning is necessary
'''
for i in list_of_df_files:
  i.columns= i.columns.str.replace('[ï»¿]', '')
'''
# creating a function to merge my dataframes


def merge_df(dataframes, column_names):
    result = dataframes[0]
    for i in range(1, len(dataframes)):
        result = pd.merge(result, dataframes[i], on=column_names)
    return result


# merging acctual dataframes on municipality_id and municipality name
columns = ("municipality_id",	"municipality_name")
data_df = merge_df(list_of_df_files, columns)


# importing csv, which needed some more cleaning before merging
covid_doses_municipalities = pd.read_csv(
    "COVID19_vaccination_municipalities_doses_timeline.csv", sep=";", thousands=".", decimal=",")
covid_doses_municipalities_df = pd.DataFrame(covid_doses_municipalities)

# using Vaccination data from 2021-12-14 - the day the open letter was published
mask_last_dates = covid_doses_municipalities_df["date"] == "2021-12-14"
covid_doses_municipalities_df = covid_doses_municipalities_df[mask_last_dates]
covid_doses_municipalities_df = covid_doses_municipalities_df.drop(
    "date", axis=1)
covid_doses_municipalities_df = covid_doses_municipalities_df.reset_index()
covid_doses_municipalities_df = covid_doses_municipalities_df.drop(
    "index", axis=1)
# covid_doses_municipalities_df.info()

# merge the new df to the other
new_list_to_merge = (data_df, covid_doses_municipalities_df)
data_df = merge_df(new_list_to_merge, "municipality_id")
# data_df.info()

antivax_docs = pd.read_csv(
    "antivax_docs.csv", sep=";", thousands=".", decimal=",")
antivax_docs_df = pd.DataFrame(antivax_docs)
antivax_docs_df = antivax_docs_df.groupby(
    "municipality_id", group_keys=False).sum().reset_index()
antivax_docs_df.columns
# antivax_docs_df.info())

# merge and fill municipalities with no antivax docs with 0
data_df = data_df.merge(antivax_docs_df, how='left',
                        left_on='municipality_id', right_on='municipality_id')
data_df = data_df.fillna(0)


urban_rural = pd.read_csv("urban_rural.csv", sep=";",
                          thousands=".", decimal=",")
urban_rural_df = pd.DataFrame(urban_rural)
urban_rural_df = urban_rural_df.drop("municipality_name", axis=1)
data_df = data_df.merge(urban_rural_df, how='left',
                        left_on='municipality_id', right_on='municipality_id')
# since missing values are from Vienna wich is an urban city categoriesed as 1
data_df = data_df.fillna(1)

political_parties = pd.read_csv(
    "nrw_2019_r.csv", sep=";", encoding='utf-8', thousands=",", decimal=".")
political_parties_df = pd.DataFrame(political_parties)
political_parties_df = political_parties_df.drop(
    ["Gebietsname", "eligible_voters", "retrieved", "invalid", "valid"], axis=1)
new_list_to_merge = (data_df, political_parties_df)
data_df = merge_df(new_list_to_merge, "municipality_id")


# checking if all values are not fine:

data_df.isnull().sum()
data_df.isna().sum()
# print(data_df.dtypes) #checking if all columns are numeric
# pop_uni didnt read right so i have to change the object to float
data_df["sh_pop_uni"] = data_df["sh_pop_uni"].apply(
    pd.to_numeric, errors='coerce')
data_df["pop_uni"] = data_df["pop_uni"].apply(pd.to_numeric, errors='coerce')

# 2 navalues in sh_pop_uni and pop_uni
# to not alter result filled with the mean
mean_pop_uni = data_df["pop_uni"].mean()
mean_sh_pop_uni_ = data_df["sh_pop_uni"].mean()

data_df["sh_pop_uni"] = data_df["sh_pop_uni"].fillna(mean_sh_pop_uni_)
data_df["pop_uni"] = data_df["pop_uni"].fillna(mean_pop_uni)

# setting setting excess skeptic doc in municpalities to 1 to creat a
# boolean with either 0,1 for a municipality with or withoutskeptic
# data_df.loc[data_df['vaccine_skeptic'] > 0, 'vaccine_skeptic'] = 1

# instead of above create a share of vaccine skeptic doc per population
data_df["sh_vaccine_skeptic"] = data_df["vaccine_skeptic"] / \
    data_df["municipality_population"]

# streamlit!!!!!!!!
# correlation matrix
'''
to_drop = ["municipality_id","municipality_name","sh_pop_0_14","sh_pop_15_29","sh_pop_30_44",
            "sh_pop_45_59","sh_pop_60_74","sh_pop_75_99","sh_pop_bms_bhs", "sh_pop_foreign",
            "sh_pop_lehre","sh_pop_pflichtschule","sh_pop_uni","salaries"
            ,"dose_1","dose_3","FPOE"]
data_df_copy_corr = data_df.copy()
data_df_copy_corr = data_df_copy_corr.drop(to_drop,axis=1)
corr_matrix = data_df_copy_corr.corr()
sn.heatmap(corr_matrix, cmap="Blues", annot=False, vmax=1, vmin=-1, linewidths= 0.01, linecolor= "black")
plt.show()
'''
# _____________________________________

data_df["dose_1_sh"] = data_df["dose_1"]/data_df["municipality_population"]
data_df["dose_2_sh"] = data_df["dose_2"]/data_df["municipality_population"]
data_df["dose_3_sh"] = data_df["dose_3"]/data_df["municipality_population"]


mask_vaccine_skeptice = data_df["vaccine_skeptic"] > 0
mask_not_vaccine_skeptice = data_df["vaccine_skeptic"] == 0
data_df_vaccine_skeptic = data_df[mask_vaccine_skeptice]
data_df_not_vaccine_skeptic = data_df[mask_not_vaccine_skeptice]


# plots
mean_vaccine_dose1_vs = data_df_vaccine_skeptic["dose_1_sh"].mean()
mean_vaccine_dose2_vs = data_df_vaccine_skeptic["dose_2_sh"].mean()
mean_vaccine_dose3_vs = data_df_vaccine_skeptic["dose_3_sh"].mean()

mean_vaccine_dose1_nvs = data_df_not_vaccine_skeptic["dose_1_sh"].mean()
mean_vaccine_dose2_nvs = data_df_not_vaccine_skeptic["dose_2_sh"].mean()
mean_vaccine_dose3_nvs = data_df_not_vaccine_skeptic["dose_3_sh"].mean()

means_vs = [mean_vaccine_dose1_vs,
            mean_vaccine_dose2_vs, mean_vaccine_dose3_vs]
means_nvs = [mean_vaccine_dose1_nvs,
             mean_vaccine_dose2_nvs, mean_vaccine_dose3_nvs]

# plot1
'''
fig, ax = plt.subplots()
index = np.arange(3)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index,means_vs, bar_width,
alpha=opacity,
color='b',
label='VaccineSkepticeDoc')

rects2 = plt.bar(index + bar_width, means_nvs, bar_width,
alpha=opacity,
color='lightblue',
label='Other')

plt.xlabel('Municipalities with or without Vs Doc')
plt.ylabel('Vaccinationrates in percentage')
plt.title('Vaccinationrates in Comparison for municipalities with or without Skeptical doc')
plt.xticks(index + bar_width, ('Dose_1', 'Dose_2', 'Dose_3'))
plt.legend()
#plt.show()
'''
# ____________________________________________________________________

# plot2

mask_vaccine_skeptice_and_rural = (data_df["vaccine_skeptic"] > 1) & (
    data_df["type_urban_rural"] > 1)
data_df_vaccine_skeptic_and_rural = data_df[mask_vaccine_skeptice_and_rural]

mean_vaccine_dose1_vs_r = data_df_vaccine_skeptic_and_rural["dose_1_sh"].mean()
mean_vaccine_dose2_vs_r = data_df_vaccine_skeptic_and_rural["dose_2_sh"].mean()
mean_vaccine_dose3_vs_r = data_df_vaccine_skeptic_and_rural["dose_3_sh"].mean()

means_vs_r = [mean_vaccine_dose1_vs_r,
              mean_vaccine_dose2_vs_r, mean_vaccine_dose3_vs_r]


fig, ax = plt.subplots()
index = np.arange(3)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_vs_r, bar_width,
                 alpha=opacity,
                 color='b',
                 label='VaccineSkepticeDoc_rural')

rects2 = plt.bar(index + bar_width, means_nvs, bar_width,
                 alpha=opacity,
                 color='lightblue',
                 label='Other')

plt.xlabel('Rural municipalities with or without Vs Doc')
plt.ylabel('Vaccinationrates in percentage')
plt.title('Vaccinationrates in Comparison for rural municipalities with or without Skeptical doc ')
plt.xticks(index + bar_width, ('Dose_1', 'Dose_2', 'Dose_3'))
plt.legend()
# plt.show()

# _________________________________________________________________________

'''
#plot 3 
sn.set_style('darkgrid')
sn.histplot(data_df["municipality_population"], log_scale=True)
plt.show()
'''

'''
#plot4
mean_pop_lehre = data_df["pop_lehre"].mean()
mean_pop_pflichtschule = data_df["pop_pflichtschule"].mean()
mean_pop_bms_bhs = data_df["pop_bms_bhs"].mean()
y = [mean_pop_bms_bhs, mean_pop_lehre , mean_pop_pflichtschule, mean_pop_uni]
mylabels=["bms_bhs", "lehre", "pflichtschule", "uni"]
colors = ["Skyblue","LightBlue", "LightGrey", "Grey"]
plt.pie(y, labels = mylabels, startangle=90, colors= colors, wedgeprops = {'linewidth': 0.5, 'edgecolor' : "black" })
plt.show() 
'''


mask_rural = data_df["type_urban_rural"] > 1
data_df['skeptic_and_rural'] = np.select([mask_vaccine_skeptice, mask_rural],
                                         [data_df['vaccine_skeptic'],
                                             data_df['type_urban_rural']],
                                         default=0)


# Models
vaccination_rate = data_df["dose_1_sh"]
pensions = data_df["pensions"]
sh_vaccine_skeptic = data_df["sh_vaccine_skeptic"]
vaccine_skeptic_and_rural = data_df["skeptic_and_rural"]
skeptic_politcal_party = data_df["FPOE_per"]
sh_0_14 = data_df["sh_pop_0_14"]
sh_15_29 = data_df["sh_pop_15_29"]
sh_30_44 = data_df["sh_pop_30_44"]
sh_45_59 = data_df["sh_pop_45_59"]
sh_60_74 = data_df["sh_pop_60_74"]
sh_75_99 = data_df["sh_pop_75_99"]
sh_forgein = data_df["sh_pop_foreign"]
sh_bms_bhs = data_df["sh_pop_bms_bhs"]
sh_lehre = data_df["sh_pop_lehre"]
sh_pflichtschule = data_df["sh_pop_pflichtschule"]
sh_uni = data_df["sh_pop_uni"]
salries = data_df["salaries"]
urban_rural = data_df["type_urban_rural"]

# model = sm.OLS(y,x)
# results = model.fit()
# print(results.summary())

result = smf.ols(
    formula='''vaccination_rate ~ sh_vaccine_skeptic + vaccine_skeptic_and_rural + pensions
     + skeptic_politcal_party + sh_0_14 + sh_15_29 + sh_30_44 + sh_45_59 + sh_60_74 + sh_75_99
     + sh_forgein + sh_bms_bhs + sh_lehre + sh_pflichtschule + sh_uni + salaries + urban_rural''',
    data=data_df).fit()

#print(result.summary())
print(data_df["skeptic_and_rural_right"])

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.header("Austrian municipalities and covid vaccination")
st.write("This project is based on the dataset of the publication: Vaccine-Skeptic Physicians and COVID-19 Vaccination Rates. The primarly focus was to get introduced to working with a dataset on Python. Unfortunatly some data was only available for the pubblication so the resualts are not the same.")
st.subheader("Abstract of the paper")
st.write("What is the role of general practitioners (GPs) in supporting or hindering public health efforts? We investigate the influence of vaccine-skeptic GPs on their patients' decisions to get a COVID-19 vaccination. We identify vaccine-skeptic GPs from the signatories of an open letter in which 199 Austrian physicians expressed their skepticism about COVID-19 vaccines. We examine small rural municipalities where patients choose a GP primarily based on geographic proximity. These vaccine-skeptic GPs reduced the vaccination rate by 5.6 percentage points. This estimate implies that they discouraged 7.9% of the vaccinable population. The effect appears to stem from discouragement rather than from rationing access to the vaccine.")
# Start to create the Dataframe by using Data that can be imported and merged automaticly
path = "Data"
file_path = glob.glob(path + "/*.csv")

# Using encoding for German letters
list_of_df_files = map(lambda file_path: pd.read_csv(
    file_path, sep=";", encoding='utf-8', decimal=",", thousands="."), file_path)
list_of_df_files = list(list_of_df_files)
# print(list_of_df_files)

# Checking if imports went well

# for i in list_of_df_files:
# i.info()

# without endcoding some more cleaning is necessary
# for i in list_of_df_files:
#   i.columns= i.columns.str.replace('[ï»¿]', '')

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

data_df["dose_1_sh"] = data_df["dose_1"]/data_df["municipality_population"]
data_df["dose_2_sh"] = data_df["dose_2"]/data_df["municipality_population"]
data_df["dose_3_sh"] = data_df["dose_3"]/data_df["municipality_population"]


mask_vaccine_skeptice = data_df["vaccine_skeptic"] > 0
mask_not_vaccine_skeptice = data_df["vaccine_skeptic"] == 0
data_df_vaccine_skeptic = data_df[mask_vaccine_skeptice]
data_df_not_vaccine_skeptic = data_df[mask_not_vaccine_skeptice]


# instead of above create a share of vaccine skeptic doc per population
data_df["sh_vaccine_skeptic"] = data_df["vaccine_skeptic"] / \
    data_df["municipality_population"]

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

data_df = data_df.rename(columns={"pop_40_44": "pop_30_44"})

mask_vaccine_skeptice_and_rural = (data_df["vaccine_skeptic"] > 1) & (
    data_df["type_urban_rural"] > 1)
data_df_vaccine_skeptic_and_rural = data_df[mask_vaccine_skeptice_and_rural]

mean_vaccine_dose1_vs_r = data_df_vaccine_skeptic_and_rural["dose_1_sh"].mean()
mean_vaccine_dose2_vs_r = data_df_vaccine_skeptic_and_rural["dose_2_sh"].mean()
mean_vaccine_dose3_vs_r = data_df_vaccine_skeptic_and_rural["dose_3_sh"].mean()

means_vs_r = [mean_vaccine_dose1_vs_r,
              mean_vaccine_dose2_vs_r, mean_vaccine_dose3_vs_r]


# streamlit!!!!!!!!
st.subheader("Intersting Plots")
# correlation matrix
plots = st.selectbox("Select plots:", ("Correlation Matrix", "Vaccinationrates for municipalities with or without skeptical doc in comparison",
                     "Population Distribution for Austrian Municipalities", "Mean Instrucition Level of Austrian Population", "Age Distribution for Austrian Municipalities", "Boxplot of salaries and pensions"))
if plots == "Correlation Matrix":
    st.write("Correlation Matrix")
    fig = plt.figure(figsize=(14, 12))
    to_drop = ["municipality_id", "municipality_name", "sh_pop_0_14", "sh_pop_15_29", "sh_pop_30_44",
               "sh_pop_45_59", "sh_pop_60_74", "sh_pop_75_99", "sh_pop_bms_bhs", "sh_pop_foreign",
               "sh_pop_lehre", "sh_pop_pflichtschule", "sh_pop_uni", "salaries", "dose_1", "dose_2", "dose_3", "FPOE", "dose_2_sh", "dose_3_sh"]
    data_df_copy_corr = data_df.copy()
    data_df_copy_corr = data_df_copy_corr.drop(to_drop, axis=1)
    corr_matrix = data_df_copy_corr.corr()
    sn.heatmap(corr_matrix, cmap="Blues", annot=True, vmax=1,
               vmin=-1, linewidths=0.01, linecolor="black", fmt=".2f").set_title('Correlation Matrix for numeric Variabels of the Dataset')
    st.pyplot(fig)


# plots

# Plot1
if plots == "Vaccinationrates for municipalities with or without skeptical doc in comparison":
    col_1, col_2 = st.columns(2)
    with col_1:
        fig, ax = plt.subplots(figsize=(8, 6))
        index = np.arange(3)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_vs, bar_width,
                         alpha=opacity,
                         color='b',
                         label='VaccineSkepticeDoc')

        rects2 = plt.bar(index + bar_width, means_nvs, bar_width,
                         alpha=opacity,
                         color='lightblue',
                         label='Other')

        plt.xlabel('Municipalities with or without Vs Doc')
        plt.ylabel('Vaccinationrates in percentage')
        plt.title(
            'Vaccinationrates in Comparison for municipalities with or without Skeptical doc')
        plt.xticks(index + bar_width, ('Dose_1', 'Dose_2', 'Dose_3'))
        plt.legend()
        st.pyplot(fig)

    with col_2:
        fig, ax = plt.subplots(figsize=(8, 6))
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
        plt.title(
            'Vaccinationrates in Comparison for rural municipalities with vs without Skeptical doc ')
        plt.xticks(index + bar_width, ('Dose_1', 'Dose_2', 'Dose_3'))
        plt.legend()
        plt.show()
        st.pyplot(fig)

# plot 3
if plots == "Population Distribution for Austrian Municipalities":
    fig = plt.figure(figsize=(10, 8))
    sn.set_style('darkgrid')
    sn.histplot(data_df["municipality_population"], log_scale=True).set_title(
        'Population Distribution for Austrian Municipalities')
    st.pyplot(fig)

# plot4
if plots == "Mean Instrucition Level of Austrian Population":
    fig = plt.figure(figsize=(10, 8))
    mean_pop_lehre = data_df["pop_lehre"].mean()
    mean_pop_pflichtschule = data_df["pop_pflichtschule"].mean()
    mean_pop_bms_bhs = data_df["pop_bms_bhs"].mean()
    y = [mean_pop_bms_bhs, mean_pop_lehre,
         mean_pop_pflichtschule, mean_pop_uni]
    mylabels = ["bms_bhs", "lehre", "pflichtschule", "uni"]
    colors = ["Skyblue", "LightBlue", "LightGrey", "Grey"]
    plt.title('Mean Instrucition Level of Austiran Population',
              fontweight="bold")
    plt.pie(y, labels=mylabels, startangle=90, colors=colors,
            wedgeprops={'linewidth': 0.5, 'edgecolor': "black"}, autopct='%.2f%%')
    st.pyplot(fig)

# plot5
if plots == "Age Distribution for Austrian Municipalities":
    fig, axes = plt.subplots(3, 2, figsize=(
        10, 8), sharey=True, constrained_layout=True)
    fig.suptitle('Population Distribution for each age Group',
                 fontweight="bold")
    sn.set_style('darkgrid')
    sn.histplot(data_df["pop_0_14"], log_scale=True, ax=axes[0, 0])
    sn.histplot(data_df["pop_15_29"], log_scale=True, ax=axes[0, 1])
    sn.histplot(data_df["pop_30_44"], log_scale=True, ax=axes[1, 0])
    sn.histplot(data_df["pop_45_59"], log_scale=True, ax=axes[1, 1])
    sn.histplot(data_df["pop_60_74"], log_scale=True, ax=axes[2, 0])
    sn.histplot(data_df["pop_75_99"], log_scale=True, ax=axes[2, 1])
    st.pyplot(fig)

# plot 6
if plots == "Boxplot of salaries and pensions":
    fig = plt.figure(figsize=(14, 8))
    ax = sn.boxplot(
        data=data_df[["salaries", "pensions"]], orient="h", palette="Blues")
    ax.set(xlabel=' Yearly salaries and pensions in 1000 €',
           title="Boxplots for Pensions and Salaries")
    st.pyplot(fig)

mask_rural = data_df["type_urban_rural"] > 1
data_df['skeptic_and_rural'] = np.select([mask_vaccine_skeptice, mask_rural],
                                         [data_df['vaccine_skeptic'],
                                             data_df['type_urban_rural']],
                                         default=0)


# Models
vaccination_rate = data_df["dose_1_sh"]
pensions = data_df["pensions"]
sh_vaccine_skeptic = data_df["sh_vaccine_skeptic"]
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
    formula='''vaccination_rate ~ sh_vaccine_skeptic  + pensions
     + skeptic_politcal_party + sh_0_14 + sh_15_29 + \
         sh_30_44 + sh_45_59 + sh_60_74 + sh_75_99
     + sh_forgein + sh_bms_bhs + sh_lehre + sh_pflichtschule + sh_uni + salaries + urban_rural''',
    data=data_df).fit()

st.subheader("Models")
models_choice = st.selectbox(
    "Select the model:", ("OLS with coefficents", "Linear Regression"))
if models_choice == "OLS with coefficents":
    st.write(result.summary())

if models_choice == "Linear Regression":

    y = vaccination_rate

    model = LinearRegression()

    variable_choices = st.multiselect('Select the variables to construct the model:', ["sh_vaccine_skeptic",
                                                                                       "pensions", "FPOE_per", "sh_pop_0_14", "sh_pop_15_29", "sh_pop_30_44",
                                                                                       "sh_pop_45_59", "sh_pop_60_74", "sh_pop_75_99", "sh_pop_foreign",
                                                                                       "sh_pop_bms_bhs", "sh_pop_lehre", "sh_pop_pflichtschule", "sh_pop_uni",
                                                                                       "salaries", "type_urban_rural"])
    test_size = st.slider('Choose test size: ', min_value=0.1,
                          max_value=0.9, step=0.1)
    if len(variable_choices) > 0 and st.button('Run Model'):
        with st.spinner('Loading'):
            x = data_df[variable_choices]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=2)
            model.fit(x_train[variable_choices], y_train)
            x_test = x_test[variable_choices].to_numpy(
            ).reshape(-1, len(variable_choices))
            y_pred = model.predict(x_test)
            r_squared = r2_score(y_test, y_pred)
            st.write("R-squared:", r_squared)

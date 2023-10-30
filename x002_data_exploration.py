# -*- coding: utf-8 -*-
"""
Author:       Ashley Avis
Filename:     x002_data_exploration
Start Date:   December 2021
Finish Date:  May 2021
Last Updated: October 2023
Inputs:       NCSchoolData_analysis.csv (created in x001_create_school_data.py)
Outputs:      X_train, X_valid, X_test, y_train, y_valid, y_test 
              (training, validation, and testing pickle files)

File Abstract:
This file:
    - Generates training, testing, and validation data
    - Examines general variable distributions
    - Examines the relationship between school performance scores and the percentage of teachers
      with advanced degrees
    - Examines the relationship between school performance scores and the percentage of teachers
      with masters degrees
"""

#%% 
##############################################################################
##############################################################################
###                           Environment Set-Up                           ###
##############################################################################
##############################################################################

# Pakages
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default='svg'
from pandas.api.types import is_numeric_dtype
from scipy.stats import kruskal
from scipy.stats import normaltest
from sklearn.model_selection import train_test_split

# Read in data
dir = "C:\\Users\\amlaw\\Documents\\Institute for Advanced Analytics\\SideProjects\\NC School Data\\FullData\\"
masters=pd.read_csv(dir + "NCSchoolRemade.csv")
masters.head()

#%%
##############################################################################
##############################################################################
###                          Train, Test, Validation                       ###
##############################################################################
##############################################################################

# Predictors and outcome
X=masters.drop(columns='spg_score')
y=masters['spg_score']

# Generate test data
X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, test_size=0.1, random_state=844)
# Verify test is 10%
print('Test %', len(X_test)/len(X))
print('Test count',len(X_test))

# Generate training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid, test_size=0.22, random_state=844)
# Verify train is 70% and validation is 20%
print('Train %', len(X_train)/len(X))
print('Valid %', len(X_valid)/len(X))

# Function to save name of dataframe
# https://stackoverflow.com/questions/72150857/passing-dataframe-and-using-its-name-to-create-the-csv-file
def get_df_name(df):
   name =[x for x in globals() if globals()[x] is df][0]
   return name

# Write to pickle - saving for analysis files
dfs = [X_train, X_valid, X_test, y_train, y_valid, y_test]
for df in dfs:
    nm = get_df_name(df)
    df.to_pickle(dir + nm + '.pkl')

#%%
##############################################################################
##############################################################################
###                            Data Exploration                            ###
##############################################################################
##############################################################################

# Examine unique values - using full data to make sure we aren't missing a value due to train/valid/test split
for col in masters.columns:
    print(col, masters[col].unique())

# Visualize numeric data with histograms and categorical data with bar plots
# Everythig is already broken into binary variables, so few bar plots should show
for col in X_train.columns:
    if is_numeric_dtype(X_train[col])==True:
        fig1=px.histogram(X_train[col])
        fig1.show()
    else:
        if X_train[col].nunique() < 20:
            X_train[col].value_counts().plot(kind='bar')


#%%
##############################################################################
##############################################################################
###      Relationship between school performance GRADES and percentage of  ###
###                      teachers with advanced degrees                    ###
##############################################################################
##############################################################################

# Is the percentage of teachers with advanced degrees different across whether a school 
# did not meet, met growth, or exceeded growth?

##############################################################################
###       Did not meet growth, Met growth, and Exceed Growth Groups        ###
##############################################################################
# Create dataframes for schools that did not meet growth, met growth, or exceeded growth
NotMet=X_train[X_train['eg_status']=='NotMet']
Met=X_train[X_train['eg_status']=='Met']
Exceeded=X_train[X_train['eg_status']=='Exceeded']

# Need to remove schools where pct_adv_degree is missing (will mess up K-W test)
NotMet = NotMet.loc[NotMet['pct_adv_degree'].notna(), ]
Met = Met.loc[Met['pct_adv_degree'].notna(), ]
Exceeded = Exceeded.loc[Exceeded['pct_adv_degree'].notna(), ]

###############################################################################
###  ANOVA: School Performanc Categories & % of teachers w/advanced degrees ###
###############################################################################
# Determine if data is normal enough to use an ANOVA
fig2=px.histogram(NotMet, x='pct_adv_degree', 
                  title='Histogram of teachers with advanced degrees for schools that did not meet growth',
                  labels={'pct_adv_degree': 'Percent of teachers with advanced degrees'})
fig3=px.histogram(Met, x='pct_adv_degree', 
                  title='Histogram of teachers with advanced degrees for schools that met growth',
                  labels={'pct_adv_degree': 'Percent of teachers with advanced degrees'})
fig4=px.histogram(Exceeded, x='pct_adv_degree', 
                  title='Histogram of teachers with advanced degrees for schools that exceeded growth',
                  labels={'pct_adv_degree': 'Percent of teachers with advanced degrees'})
fig2.show()
fig3.show()
fig4.show()

# Conclusion: Some are questionable on their normality, so we'll do a statistical test of normality

# Normality test
k1, p1 = normaltest(NotMet['pct_adv_degree'])
k2, p2 = normaltest(Met['pct_adv_degree'])
k3, p3 = normaltest(Exceeded['pct_adv_degree'])
print(p1, p2, p3)
# Conclusion: Not normal

# Use Kruskal-Wallis test since data is not normal
# Test if there is a difference in distribution of teachers with advanced degrees in schools with 
# different levels of EVAAS growth
stat, p=kruskal(NotMet['pct_adv_degree'], Met['pct_adv_degree'], Exceeded['pct_adv_degree'])
print('The p-value is ' + str(p))

if (p>0.0054):
    print('There is NOT a difference in percentage of teachers with advanced degrees.')
else:
    print('There IS a difference in percentage of teachers with advanced degrees.')

# Visualize differences in percentage of teachers with advanced degrees with boxplots
fig5 = px.box(X_train[X_train['eg_status']!=0], x="eg_status", y="pct_adv_degree", 
              labels={'eg_status': 'EVAAS Growth Status',
                        'pct_adv_degree': '% of teachers with advanced degrees'})
fig5.update_layout(title={'text':'Growth Status and % of teachers with advanced degrees', 'x':0.5})
fig5.show()


# Conclusion: The percentage of teachers with advanced degrees does not vary that much by growth status.

#%%
##############################################################################
##############################################################################
###      Relationship between school performance SCORES and percentage of  ###
###                      teachers with advanced degrees                    ###
##############################################################################
##############################################################################

# Combine X and y for ease
Xy_train=pd.concat([X_train, y_train], axis=1)

##############################################################################
###         School Performance Scores & Teachers w/Advanced Degrees        ###
##############################################################################
# Linear regression: School performance score & % of teachers with advanced degrees
fig6=px.scatter(Xy_train, x='pct_adv_degree', y='spg_score', trendline='ols', 
                labels={'pct_adv_degree':'% of teachers with advanced degrees',
                       'spg_score':'2017 School Performance Score'})
fig6.update_layout(title={'text':'Relationship between school performance scores and teacher advanced degrees','x':0.5})
fig6.show()


##############################################################################
###          School Performance Scores & Teachers w/Masters Degrees        ###
##############################################################################

# Visualize for normality
fig7=px.histogram(NotMet, x='Masters', 
                  title="Histogram of teachers with master's degrees for schools that did not meet growth",
                  labels={'Masters': "Percent of teachers with master's degrees"})
fig8=px.histogram(Met, x='Masters', 
                  title="Histogram of teachers with master's degrees for schools that met growth",
                  labels={'Masters': "Percent of teachers with master's degrees"})
fig9=px.histogram(Exceeded, x='Masters', 
                  title="Histogram of teachers with master's degrees for schools that exceeded growth",
                  labels={'Masters': "Percent of teachers with master's degrees"})
fig7.show()
fig8.show()
fig9.show()

# Conclusion: Definitely not normal. Use Kruskal-Wallis test.

# Need to remove schools where Masters is missing (will mess up K-W test)
NotMet = NotMet.loc[NotMet['Masters'].notna(), ]
Met = Met.loc[Met['Masters'].notna(), ]
Exceeded = Exceeded.loc[Exceeded['Masters'].notna(), ]
    
# K-W: Test if there is a difference in distribution of teachers with masters degrees in schools with 
# different levels of EVAAS growth
stat, p=kruskal(NotMet['Masters'], Met['Masters'], Exceeded['Masters'])
print('The p-value is ' + str(p))

if (p>0.0054):
    print('There is NOT a difference in percentage of teachers with masters degrees.')
else:
    print('There IS a difference in percentage of teachers with masters degrees.')


# Visualize differences in percentage of teachers with masters degrees with boxplots
fig10 = px.box(Xy_train[Xy_train['eg_status']!=0], x="eg_status", y="Masters", 
               labels={'eg_status': 'EVAAS Growth Status',
                    'Masters': "% of teachers with master's degrees"})
fig10.update_layout(title={'text': "Growth Status and % of teachers with master's degrees", 'x':0.5})
fig10.show()

# It seems like only outliers in NC Schools have more than 30% of teachers with master's degrees. 
# Those that met growth seem to have a lower percentage of teachers with master's degrees overall.


# Linear Regression: School Performance Score and % of teachers with masters degrees.
fig11=px.scatter(Xy_train, x='Masters', y='spg_score', trendline='ols', 
                labels={'Masters':"% of teachers with master's degrees",
                       'spg_score':'2017 School Performance Score'})
fig11.update_layout(title={'text':"Relationship between school performance scores and teacher master's degrees",'x':0.5})
fig11.show()

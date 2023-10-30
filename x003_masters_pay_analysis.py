# -*- coding: utf-8 -*-
"""
Author:       Ashley Avis
Filename:     x003_masters_pay_analysis
Start Date:   December 2021
Finish Date:  May 2021
Last Updated: October 2023
Inputs:       X_train, X_valid, X_test, y_train, y_valid, y_test 
              (training, validation, and testing pickle files)
Outputs:      Decision Tree PDFs (Features and Splits) and 
              LASSO_X3_Sig, LASSO_X4_Sig, LASSO_X5_Sig (Coefficients from LASSO models)


File Abstract:
This file generates Decision Trees to examine features related to:
    - Growth Status
    - School Performance Grades (A-F)
It also generates LASSO Regressions for predicting School Performance Scores with
the following variable categories:
    - Variables identified from Growth Status decision tree
    - Variables identified from School Performance Grades decision tree
    - Most variables available with some collinar variables removed
    
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
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Read in data
dir = "C:\\Users\\amlaw\\Documents\\Institute for Advanced Analytics\\SideProjects\\NC School Data\\FullData\\"
pkls = ['X_train', 'X_valid', 'X_test', 'y_train', 'y_valid', 'y_test']
dfs = []
for pk in pkls:
    pk = pd.read_pickle(dir + pk + '.pkl')
    dfs.append(pk)
    
# https://stackoverflow.com/questions/70180608/separate-a-list-of-data-frames-into-multiple-data-frames
for i in range(len(dfs)):
    globals()[pkls[i]] = dfs[i]


#%%
##############################################################################
##############################################################################
###     Features related to whether a school did or did not meet growth    ###
##############################################################################
##############################################################################

# Combine X and y for ease
Xy_train=pd.concat([X_train, y_train], axis=1)
Xy_train.fillna(0, inplace=True)

##############################################################################
###                      Decision Tree for Growth Status                   ###
##############################################################################
# Create matrices for a decision tree
Xy_train = Xy_train.loc[(Xy_train['eg_status'].notna())&(Xy_train['eg_status']!=0), ]
X = Xy_train.drop(columns=['eg_status', 'agency_code', 'School_Name', 'LEA_Name', 'eg_score', 'School_Type_Desc', 'ma_spg_score',
                         'rd_spg_score', 'ma_eg_status', 'rd_eg_status', 'ma_eg_score', 'rd_eg_score', 'LEA_Name.1', 'Free',
                         'Reduced', 'Total', 'ACALL', 'ACCO', 'ACEN', 'ACMA', 'ACRD', 'ACSC', 'ACWR', 'avg_sat_score',
                         'pct_sat_participation', 'EDS%', 'Final_ADM', 'pct_ap_pass', 'pct_ap_participation', 'year', 
                         'WAP_Count', 'NumClassrooms', 'crime', 'short_term', 'long_term', 'expulsion', 
                         'Does_Not_Meet_Expected_Growth', 'Exceeds_Expected_Growth', 'Meets_Expected_Growth', 'Locale2', 
                         'spg_score', 'spg_grade', 'Advanced', 'In_Process', 'Total_Degrees'])
y = Xy_train.loc[(Xy_train['eg_status'].notna())&(Xy_train['eg_status']!=0), 'eg_status']

# Initialize decision tree
dt = tree.DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=50)

# I tried gini and entropy and gini proved to be the best. 
# I tried different pruning depths and 10 with a minimum leaf size of 50 seemed to produce a reasonably sized tree

# Fit decision tree
dt = dt.fit(X, y)

# Visualize the tree itself (and save as a PDF)
X_feature_names = list(X.columns)
plt.figure()
tree.plot_tree(dt, filled=True, feature_names=X_feature_names, class_names=['Not Met', 'Met', 'Exceeded'])
plt.savefig(dir + 'decision_tree_growth.pdf', format='pdf', bbox_inches='tight')

# Predict
y_pred = dt.predict(X)
print("Goodness of fit:",metrics.accuracy_score(y, y_pred))

##############################################################################
###                              Validation Data                           ###
##############################################################################
# Create matrices for decision tree
X_valid_simp = X_valid.loc[(X_valid['eg_status'].notna())&(X_valid['eg_status']!=0), ]
Valid_growth = X_valid_simp['eg_status']
X_valid_simp = X_valid_simp.drop(columns=['eg_status', 'agency_code', 'School_Name', 'LEA_Name', 'eg_score', 'School_Type_Desc', 
                         'rd_spg_score', 'ma_eg_status', 'rd_eg_status', 'ma_eg_score', 'rd_eg_score', 'LEA_Name.1', 'Free',
                         'Reduced', 'Total', 'ACALL', 'ACCO', 'ACEN', 'ACMA', 'ACRD', 'ACSC', 'ACWR', 'avg_sat_score',
                         'pct_sat_participation', 'EDS%', 'Final_ADM', 'pct_ap_pass', 'pct_ap_participation', 'year', 
                         'WAP_Count', 'NumClassrooms', 'crime', 'short_term', 'long_term', 'expulsion', 
                         'Does_Not_Meet_Expected_Growth', 'Exceeds_Expected_Growth', 'Meets_Expected_Growth', 'Locale2', 
                         'ma_spg_score', 'spg_grade', 'Advanced', 'In_Process', 'Total_Degrees'])
X_valid_simp.fillna(0, inplace=True)
y_pred_valid = dt.predict(X_valid_simp)

# Accuracy on validation
print(metrics.accuracy_score(Valid_growth, y_pred_valid))

# Gini performed slightly better on validation

# Variable importance
feat_importance = dt.tree_.compute_feature_importances(normalize=False)
var_importance = {}
var_import=[]
for i in range(0, len(X_feature_names)):
    if feat_importance[i] > 0:
        var_importance[X_feature_names[i]] = feat_importance[i]
        # Store a list of significant variables for later use
        var_import.append(X_feature_names[i])

    
sort_var_import=sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
for i in sort_var_import:
    print(str(i[0]) + ': ' + str(i[1]))

# Degrees matter for school growth, but teacher experience and student features also contribute.

#%%
##############################################################################
##############################################################################
###            Features related to school performance grades (A-F)         ###
##############################################################################
##############################################################################

# Create data matrix for seeing relationship between school performance grade and other variables
X2=Xy_train.drop(columns=['eg_status', 'agency_code', 'School_Name', 'LEA_Name', 'eg_score', 'School_Type_Desc', 
                         'rd_spg_score', 'ma_eg_status', 'rd_eg_status', 'ma_eg_score', 'rd_eg_score', 'LEA_Name.1', 'Free',
                         'Reduced', 'Total', 'ACALL', 'ACCO', 'ACEN', 'ACMA', 'ACRD', 'ACSC', 'ACWR', 'avg_sat_score',
                         'pct_sat_participation', 'EDS%', 'Final_ADM', 'pct_ap_pass', 'pct_ap_participation', 'year', 
                         'WAP_Count', 'NumClassrooms', 'crime', 'short_term', 'long_term', 'expulsion', 
                         'Does_Not_Meet_Expected_Growth', 'Exceeds_Expected_Growth', 'Meets_Expected_Growth', 'Locale2', 
                         'ma_spg_score', 'spg_grade', 'Advanced', 'In_Process', 'Total_Degrees', 'spg_score'])
y2=Xy_train['spg_grade']

# Initialize tree
dt2 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=50)

# Fit tree
dt2 = dt2.fit(X2, y2)

# Visualize tree
X2_feature_names = list(X2.columns)
plt.figure()
tree.plot_tree(dt2, filled=True, feature_names=X2_feature_names, class_names=['A', 'B', 'C', 'D', 'F', 'A+NG'])
plt.savefig(dir + 'decision_tree_grades.pdf', format='pdf', bbox_inches='tight')

# Variable Importance
feat_importance2 = dt2.tree_.compute_feature_importances(normalize=False)
var_importance2 = {}
var_import2=[]
for i in range(0, len(X2_feature_names)):
    if feat_importance2[i] > 0:
        var_importance2[X2_feature_names[i]] = feat_importance2[i]
        #Storing a list of significant variables for later use
        var_import2.append(X2_feature_names[i])

    
sort_var_import2=sorted(var_importance2.items(), key=lambda x: x[1], reverse=True)
for i in sort_var_import2:
    print(str(i[0]) + ': ' + str(i[1]))

# Teacher experience and turnover matter, but so do student features and SES

#Predicting values based on tree
y_pred2 = dt2.predict(X2)
print("Goodness of fit:",metrics.accuracy_score(y2, y_pred2))

##############################################################################
###                              Validation Data                           ###
##############################################################################

# Create matrices
X_valid_simp = X_valid.loc[(X_valid['spg_grade'].notna())&(X_valid['spg_grade']!=0), ]
Valid_grade=X_valid_simp['spg_grade']
X_valid_simp.fillna(0, inplace=True)
X_valid_simp=X_valid_simp.drop(columns=['eg_status', 'agency_code', 'School_Name', 'LEA_Name', 'eg_score', 'School_Type_Desc', 
                         'rd_spg_score', 'ma_eg_status', 'rd_eg_status', 'ma_eg_score', 'rd_eg_score', 'LEA_Name.1', 'Free',
                         'Reduced', 'Total', 'ACALL', 'ACCO', 'ACEN', 'ACMA', 'ACRD', 'ACSC', 'ACWR', 'avg_sat_score',
                         'pct_sat_participation', 'EDS%', 'Final_ADM', 'pct_ap_pass', 'pct_ap_participation', 'year', 
                         'WAP_Count', 'NumClassrooms', 'crime', 'short_term', 'long_term', 'expulsion', 
                         'Does_Not_Meet_Expected_Growth', 'Exceeds_Expected_Growth', 'Meets_Expected_Growth', 'Locale2', 
                         'ma_spg_score', 'spg_grade', 'Advanced', 'In_Process', 'Total_Degrees'])
y_pred2_valid = dt2.predict(X_valid_simp)


# Accuracy
print("Accuracy:",metrics.accuracy_score(Valid_grade, y_pred2_valid))

# # Section 4: What features contribute to school performance scores?

#%%
##############################################################################
##############################################################################
###               Features related to school performance SCORES            ###
##############################################################################
##############################################################################

##############################################################################
###                     Prepare Data for LASSO Regression                  ###
##############################################################################
# Standardize data for LASSO Regression
scaler = StandardScaler()
XL = Xy_train.loc[(Xy_train['spg_score'].notna())&(Xy_train['spg_score']!=0)]
y2=XL['spg_score']
XL=XL.drop(columns=['eg_status', 'agency_code', 'School_Name', 'LEA_Name', 'eg_score', 'School_Type_Desc', 
                    'rd_spg_score', 'ma_eg_status', 'rd_eg_status', 'ma_eg_score', 'rd_eg_score', 'LEA_Name.1', 'Free',
                    'Reduced', 'Total', 'ACALL', 'ACCO', 'ACEN', 'ACMA', 'ACRD', 'ACSC', 'ACWR', 'avg_sat_score',
                    'pct_sat_participation', 'EDS%', 'Final_ADM', 'pct_ap_pass', 'pct_ap_participation', 'year', 
                    'WAP_Count', 'NumClassrooms', 'crime', 'short_term', 'long_term', 'expulsion', 
                    'Does_Not_Meet_Expected_Growth', 'Exceeds_Expected_Growth', 'Meets_Expected_Growth', 'Locale2', 
                    'ma_spg_score', 'spg_grade', 'Advanced', 'In_Process', 'Total_Degrees', 'spg_score'])
stand_X = scaler.fit_transform(XL)
XL_feature_names = list(XL.columns)
stand_X_train = pd.DataFrame(stand_X, columns=XL_feature_names)

np_y_train = y2.to_numpy()

stand_y_train = scaler.fit_transform(np_y_train.reshape(-1,1))

#Standardize validation data
X_v = X_valid.fillna(0)
X_v=X_v.drop(columns=['eg_status', 'agency_code', 'School_Name', 'LEA_Name', 'eg_score', 'School_Type_Desc', 
                      'rd_spg_score', 'ma_eg_status', 'rd_eg_status', 'ma_eg_score', 'rd_eg_score', 'LEA_Name.1', 'Free',
                      'Reduced', 'Total', 'ACALL', 'ACCO', 'ACEN', 'ACMA', 'ACRD', 'ACSC', 'ACWR', 'avg_sat_score',
                      'pct_sat_participation', 'EDS%', 'Final_ADM', 'pct_ap_pass', 'pct_ap_participation', 'year', 
                      'WAP_Count', 'NumClassrooms', 'crime', 'short_term', 'long_term', 'expulsion', 
                      'Does_Not_Meet_Expected_Growth', 'Exceeds_Expected_Growth', 'Meets_Expected_Growth', 'Locale2', 
                      'ma_spg_score', 'spg_grade', 'Advanced', 'In_Process', 'Total_Degrees'])
X_v = X_v.loc[y_valid.notna(), ]
stand_X_v = scaler.fit_transform(X_v)
stand_X_valid = pd.DataFrame(stand_X_v, columns=X_v.columns)

y_valid = y_valid[y_valid.notna()]
np_y_valid = y_valid.to_numpy()

stand_y_valid = scaler.fit_transform(np_y_valid.reshape(-1,1))

##############################################################################
###                              Variable Subsets                          ###
##############################################################################
# I used the significant variables from the decision trees for growth and school performance grade as a starting point

# X3 has variables from school performance grade model
X3=stand_X_train[var_import2]
X3_v=stand_X_valid[var_import2]

# X4 has variables from the growth model
X4=stand_X_train[var_import]
X4_v=stand_X_valid[var_import]

##############################################################################
###                        LASSO Regression Functions                      ###
##############################################################################

# Create a function to repeat LASSO

def lasso_model(x, y, x_valid, y_valid, alpha):
    # fit
    model = Lasso(alpha=alpha, normalize=False, random_state=543)
    model.fit(x, y)
    pred = model.predict(x_valid)
    coef = model.coef_
    #MAE
    diff=abs(y_valid-pred)
    print("Alpha", alpha, "MAE:",np.mean(diff))

# Create list of alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1e-1, 1, 5, 10]

# Create function to put variable coefficients back into regular units
def unstdev(l, coef):
    for elem in coef['Variable']:
        l.append(coef.loc[elem, 'Coefficient']*(np.std(y_train)/np.std(X_train[elem])))
    for i in range(0, len(coef)):
        print(coef.Variable[i] +': ' + str(l[i]))

##############################################################################
###            LASSO Regression: School Performance Grade Features         ###
##############################################################################

# Run function for X3
for elem in alpha_lasso:
    lasso_model(X3, stand_y_train, X3_v, stand_y_valid, elem)

# An alpha >= 1 gave all 0 coefficients. So I'm using 0.1 as my alpha level.

# LASSO regression using significant variables for school performance grades
model = Lasso(alpha=0.1,normalize=False, random_state=543)
model.fit(X3, stand_y_train)

for i in range(0, len(var_importance2)):
    print(var_import2[i], model.coef_[i])

# Create a pandas dataframe to print values
dict_LASSO_X3 = {'Variable':var_import2,'Coefficient':model.coef_}
LASSO_X3 = pd.DataFrame(dict_LASSO_X3)

# Remove zero coefficients and print
LASSO_X3_Sig = LASSO_X3[LASSO_X3['Coefficient'] != 0]
LASSO_X3_Sig.reset_index(drop=True, inplace=True)
prop=pd.DataFrame(['4-Year Graduation Rate', '% of White Students', '% of Economically Disadvantaged Students',
                   '% of Attendance to ADM', '1 Year Teacher/Principal Turnover %', '% of Free Lunch Students', 
                   'Short Term Suspensions per 100'], columns=['Variable Name'])
LASSO_X3_Sig=pd.concat([LASSO_X3_Sig, prop], axis=1)
LASSO_X3_Sig['Index']=LASSO_X3_Sig['Variable']
LASSO_X3_Sig.set_index('Index', inplace=True)
LASSO_X3_Sig

# Unstandardize coefficients
coef_unstd_X3=[]
unstdev(coef_unstd_X3, LASSO_X3_Sig)

# Create a plot ot visualize information 
fig12 = px.bar(LASSO_X3_Sig, x="Variable Name", y="Coefficient")
fig12.update_layout(title={'text':'Coefficients for each Selected Variable', 'x':0.5})
fig12.show()


##############################################################################
###           LASSO Regression: School Performance Growth Features         ###
##############################################################################

# Run function for X4
for elem in alpha_lasso:
    lasso_model(X4, stand_y_train, X4_v, stand_y_valid, elem)

# An alpha >= 1 gave all 0 coefficients. So I'm using 0.1 as my alpha level.

# LASSO regression using significant variables from school growth decision tree
model = Lasso(alpha=0.1, normalize=False, random_state=543)
model.fit(X4, stand_y_train)
for i in range(0, len(var_importance)):
    print(var_import[i], model.coef_[i])

# Create pandas dataframe for examining coefficients
dict_LASSO_X4 = {'Variable':var_import,'Coefficient':model.coef_}
LASSO_X4 = pd.DataFrame(dict_LASSO_X4)

# Remove zero coefficients
LASSO_X4_Sig = LASSO_X4[LASSO_X4['Coefficient'] != 0]
LASSO_X4_Sig.reset_index(drop=True, inplace=True)
prop=pd.DataFrame(['5-Year Graduation Rate', '% of Black Students', '% of Attendance to ADM',
                   'Average Media Age', '% of Teachers with < 3 Years Experience',
                   '% of Teachers with 10+ Years Experience', 'Percent of Teachers with Advanced Degrees',
                   'Short Term Suspensions per 100'],
                   columns=['Variable Name'])
LASSO_X4_Sig=pd.concat([LASSO_X4_Sig, prop], axis=1)
LASSO_X4_Sig['Index']=LASSO_X4_Sig['Variable']
LASSO_X4_Sig.set_index('Index', inplace=True)

LASSO_X4_Sig

# Unstandardize the coefficients
coef_unstd_X4=[]
unstdev(coef_unstd_X4, LASSO_X4_Sig)

# Create a plot ot visualize information 
fig13 = px.bar(LASSO_X4_Sig, x="Variable Name", y="Coefficient")
fig13.update_layout(title={'text':'Coefficients for each Selected Variable', 'x':0.5})
fig13.show()


##############################################################################
###                 LASSO Regression: Most Variables Included              ###
##############################################################################

# Remove variables that measure similar ideas in school and graduation rates to prevent multicollinearity
X5=stand_X_train.drop(columns=['total_expense', 'GradRate5Yr', 'Male_Tot', 'FreePct', 'ReducedPct', 'local_rank', 'state_rank', 
                               'federal_rank', 'total_rank'])

# Validation data
X5_v=stand_X_valid.drop(columns=['total_expense', 'GradRate5Yr', 'Male_Tot', 'FreePct', 'ReducedPct', 'local_rank', 'state_rank', 
                               'federal_rank', 'total_rank'])

# Run function for X5
for elem in alpha_lasso:
    lasso_model(X5, stand_y_train, X5_v, stand_y_valid, elem)

# An alpha >= 1 gave all 0 coefficients. So I'm using 0.1 as my alpha level.

# LASSO regression using most variables
model=Lasso(alpha=0.1, normalize=False, random_state=543)
model.fit(X5, stand_y_train)
sig_X5=[]
for i in range(0, len(X5.columns)):
    print(X5.columns[i], model.coef_[i])
    if model.coef_[i]!=0:
        sig_X5.append(X5.columns[i])

# Create pandas dataframe to print values
dict_LASSO_X5 = {'Variable':X5.columns,'Coefficient':model.coef_}
LASSO_X5 = pd.DataFrame(dict_LASSO_X5)

# Remove zero coefficients
LASSO_X5_Sig = LASSO_X5[LASSO_X5['Coefficient'] != 0]
LASSO_X5_Sig.reset_index(drop=True, inplace=True)
prop=pd.DataFrame(['4-Year Graduation Rate', 'Community Eligibility Provision Indicator', '% of Female Students',
                   '% of Black Students', '% of White Students', '% of Economically Disadvantaged Students',
                   '% of Attendance to ADM', 'Average Media Age', '% of Teachers with < 3 Years Experience',
                   'Short Term Suspensions per 100'], columns=['Variable Name'])
LASSO_X5_Sig=pd.concat([LASSO_X5_Sig, prop], axis=1)
LASSO_X5_Sig['Index']=LASSO_X5_Sig['Variable']
LASSO_X5_Sig.set_index('Index', inplace=True)
LASSO_X5_Sig

# Unstandardize the coefficients
coef_unstd_X5=[]
unstdev(coef_unstd_X5, LASSO_X5_Sig)

# Create a plot ot visualize information 
fig14 = px.bar(LASSO_X5_Sig, x="Variable Name", y="Coefficient")
fig14.update_layout(title={'text':'Coefficients for each Selected Variable', 'x':0.5})
fig14.show()

# Results: The model using X4 (school growth parameters) had the best results

# Model X4 Results printed again
model = Lasso(alpha=0.1, normalize=False, random_state=543)
model.fit(X4, stand_y_train)
for i in range(0, len(var_importance)):
    print(var_import[i], model.coef_[i])

#%%
##############################################################################
##############################################################################
###                       Save Coefficients for Graphs                     ###
##############################################################################
##############################################################################

# Function to save name of dataframe
# https://stackoverflow.com/questions/72150857/passing-dataframe-and-using-its-name-to-create-the-csv-file
def get_df_name(df):
   name =[x for x in globals() if globals()[x] is df][0]
   return name

# Write results to pickles
df_out = [LASSO_X3_Sig, LASSO_X4_Sig, LASSO_X5_Sig]
for f in df_out:
    nm = get_df_name(f)
    f.to_pickle(dir + nm + '.pkl')


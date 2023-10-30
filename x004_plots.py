# -*- coding: utf-8 -*-
"""
Author:       Ashley Avis
Filename:     x004_plots
Start Date:   December 2021
Finish Date:  May 2021
Last Updated: October 2023
Inputs:       LASSO_X3_Sig, LASSO_X5_Sig, LASSO_X5_Sig
Outputs:      Plots


File Abstract:
This file generates plots for a presentation and document about the analysis
    
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
import plotly.graph_objects as go

# Read in data
dir = "C:\\Users\\amlaw\\Documents\\Institute for Advanced Analytics\\SideProjects\\NC School Data\\FullData\\"
pkls = ['LASSO_X3_Sig', 'LASSO_X4_Sig', 'LASSO_X5_Sig']
dfs = []
for pk in pkls:
    pk = pd.read_pickle(dir + pk + '.pkl')
    dfs.append(pk)
    
# https://stackoverflow.com/questions/70180608/separate-a-list-of-data-frames-into-multiple-data-frames
for i in range(len(dfs)):
    globals()[pkls[i]] = dfs[i]

# Full data
masters=pd.read_csv(dir + "NCSchoolRemade.csv")

#%%
##############################################################################
##############################################################################
###                    Plot Generation of Model X4 Results                 ###
##############################################################################
##############################################################################

##############################################################################
###               School Performance Score & Teacher Experience            ###
##############################################################################
# Fit a linear regression to see the relationship between school score and the percent of teachers with <3 years experience.
fig15=px.scatter(masters, x='pct_experience_0', y='spg_score', trendline='ols', 
                labels={'pct_experience_0':"% of teachers with < 3 Years of Experience",
                       'spg_score':'2017 School Performance Score'})
fig15.update_layout(title={'text':"Relationship between school performance scores and teacher experience",'x':0.5})
fig15.show()

fig15.show()


##############################################################################
###             School Performance Score & Economic Disadvantage           ###
##############################################################################
# Fit a linear regression to see the relationship between school score and the percent of teachers with masters degrees.
fig16=px.scatter(masters, x='pct_eds', y='spg_score', trendline='ols', 
                labels={'pct_eds':"Percentage of Economically Disadvantaged Students",
                       'spg_score':'2017 School Performance Score'})
fig16.update_layout(title={'text':"Relationship between school performance scores and economic disadvantage",'x':0.5})

fig16.show()


##############################################################################
###             School Performance Score & Short Term Suspensions          ###
##############################################################################
# Fit a linear regression to see the relationship between school score and number of short term suspensions.
fig17=px.scatter(masters, x='shortsusper100', y='spg_score', trendline='ols', 
                labels={'shortsusper100':"Number of Short Term Suspensions per 100 Students",
                       'spg_score':'2017 School Performance Score'})
fig17.update_layout(title={'text':"Relationship between school performance scores and short term suspensions",'x':0.5})
fig17.show()


##############################################################################
###                   Variable Significance Across 3 Models                ###
##############################################################################
# Create a plot to compare variable significance in the 3 models to get a sense of what matters for school performance scores
fig=go.Figure()
fig.add_trace(go.Bar(
    y=LASSO_X3_Sig['Variable Name'], 
    x=LASSO_X3_Sig['Coefficient'], 
    orientation='h',
    name='Growth Decision Tree Variables',
    marker={'color':'orange'}))
fig.add_trace(go.Bar(
    y=LASSO_X4_Sig['Variable Name'], 
    x=LASSO_X4_Sig['Coefficient'], 
    orientation='h',
    name='Grade Decision Tree Variables',
    marker={'color':'blue'}))
fig.add_trace(go.Bar(
    y=LASSO_X5_Sig['Variable Name'], 
    x=LASSO_X5_Sig['Coefficient'], 
    orientation='h',
    name='Selected Variables',
    marker={'color':'light green'}))
fig.update_layout(title='Variable Significance', title_x=0.5, legend=dict(
    yanchor="bottom",
    y=-0.25,
    xanchor="left",
    x=0
))
layout = go.Layout(
    autosize=False,
    width=800,
    height=700)
fig.update_layout(layout)


##############################################################################
###              School Performance Score & % of White Students            ###
##############################################################################
#Fitting a linear regression to see the relationship between school score and the percent of white students
fig18=px.scatter(masters, x='White', y='spg_score', trendline='ols', 
                labels={'White':"% of White Students",
                       'spg_score':'2017 School Performance Score'})
fig18.update_layout(title={'text':"Relationship between school performance scores and white students",'x':0.5})
fig18.show()


##############################################################################
###      % of White Students & % of Economically Diadvantaged Students     ###
##############################################################################
#Fitting a linear regression to see the relationship between percent of white students and percent of economically disadvantaged
#students
fig19=px.scatter(masters, x='White', y='pct_eds', trendline='ols', 
                labels={'White':"% of White Students",
                       'eds_pct':'% Economically Disadvantaged'})
fig19.update_layout(title={'text':"Relationship between race and economic disadvantage",'x':0.5})
fig19.show()


##############################################################################
###      School Performance Score & % of Teachers with Master's Degrees    ###
##############################################################################
#Fitting a linear regression to see the relationship between school score and the percent of teachers with masters degrees.
fig20=px.scatter(masters, x='Masters', y='spg_score', trendline='ols', 
                labels={'Masters':"Percentage of Teachers with Master's Degrees",
                       'spg_score':'2017 School Performance Score'})
fig20.update_layout(title={'text':"Relationship between school performance scores and percent of teachers with master's degrees",'x':0.5})

fig20.show()


##############################################################################
###              SAT Score & % of Teachers with Master's Degrees           ###
##############################################################################
#Fitting a linear regression to see the relationship between SAT score and the percent of teachers with masters degrees.
sat_masters=masters[masters['avg_sat_score']>0]
fig21=px.scatter(sat_masters, x='Masters', y='avg_sat_score', trendline='ols', 
                labels={'Masters':"Percentage of Teachers with Master's Degrees",
                       'avg_sat_score':'2017 SAT Scores'})
fig21.update_layout(title={'text':"Relationship between SAT scores and percent of teachers with master's degrees",'x':0.5})

fig21.show()


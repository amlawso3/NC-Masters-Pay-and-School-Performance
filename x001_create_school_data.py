# -*- coding: utf-8 -*-
"""
Author:       Ashley Avis
Filename:     x001_create_school_data
Start Date:   December 2021
Finish Date:  May 2021
Last Updated: October 2023
Inputs:       See Data Summary (input files)
Outputs:      NCSchoolData_analysis.csv (File with one row per School)


File Abstract:
This file prepares and merges 30 publicly available files from NC DPI for analysis.
Each file requires different adjustments, so a common cleaning and manipulation process is not possible.
This file generates several functions for data cleaning including:
    - Common functions across datasets (filtering year and subpopulation, removing LEAs and SEAs, setting the index)
    - Data specific functions to drop columns, rename, generate indexes, or aggregations
This file follows the following order:
    1) Ingest and clean/manipulate data
    2) Join all data sources together
    3) Generate calculations (rates)
    4) Modify the dataset for analysis
    5) Output dataset
The final format needed for analysis is one row per School with a variety of features detailed below.

Data Summary (input files):
The following types of data were included:
    - Base school information (name, type, calendar, region, urbancity)
    - School Performance Grades (main outcome for analysis)
    - School Outcome Features (Graduation Rate, SAT/ACT Scores, and Advanced Placement)
    - School Demographics (Race, Gender, Free & Reduced Lunch, Economically Disadvantaged Youth)
    - School Attendance (Average Daily Membership, Attendance, Class Size, Crimes & Suspensions)
    - Teacher Features (Experience, Effectiveness, Degrees, Licenses, National Board Certification)
    
"""


#%% 
##############################################################################
##############################################################################
###                           Environment Set-Up                           ###
##############################################################################
##############################################################################

# Import packages
import pandas as pd

# Directory 
dir = 'C:\\Users\\amlaw\\Documents\\Institute for Advanced Analytics\\SideProjects\\NC School Data\\FullData\\'

# Year 
yr = 2017

#%% 
##############################################################################
##############################################################################
###                            Common Functions                            ###
##############################################################################
##############################################################################

# Year filter
def yr_filt(df, yr_var = 'year'):
    df = df[df[yr_var] == yr].copy()
    return df

# Subgroup filter
def sub_filt(df, sub='subgroup'):
    df = df[df[sub] == 'ALL'].copy()
    return df

# Set index to agency
def ind_ag(df, ind='agency_code'):
    df.set_index(ind, inplace=True)
    return df

# LEA and SEA clean up
def lea_sea(df):
    # Drop LEAs
    df=df[~df['agency_code'].str.endswith('LEA')]
    # Drop state
    df=df[~df['agency_code'].str.endswith('SEA')]
    return df


#%% 
##############################################################################
##############################################################################
###                           School Information                           ###
##############################################################################
##############################################################################

#This will be the base file for merging everything together

school=pd.read_csv(dir + 'custom_report_school.csv')

def school_clean(school):
    # Rename
    school.rename(columns={'School Number': 'agency_code', 'School Calendar Description': 'School Calendar'}, inplace=True)
    # Select only public
    school=school[school['School Designation Type']=='P'].copy()
    # Drop extra columns
    school.drop(columns=['School Calendar Type', 'School Designation Type', 'School Designation Desc'], inplace=True)
    # Change LEA Number  and agency_code to string
    school['LEA Number']=school['LEA Number'].apply(str).str.zfill(3)
    school['agency_code']=school['agency_code'].apply(str).str.zfill(6)
    return school

school = school_clean(school)
school = ind_ag(school, 'LEA Number')

#%% 
##############################################################################
##############################################################################
###                        School Performance Grades                       ###
##############################################################################
##############################################################################

# Outcome measure for main analysis
SPG=pd.read_excel(dir + 'rcd_acc_spg1.xlsx')

# No data specific function needed
SPG = yr_filt(SPG)
SPG = ind_ag(SPG)

# Drop unnecessary columns
SPG.drop(columns=['year', 'asm_option', 'subgroup', 'ma_score', 'ma_score_masking', 'ma_grade', 'rd_score', 'rd_score_masking',
                 'rd_grade', 'sc_score', 'sc_score_masking', 'm1_score', 'm1_score_masking', 'e2_score', 'e2_score_masking', 
                  'bi_score', 'bi_score_masking', 'act_score', 'act_score_masking', 'wk_score', 'wk_score_masking', 
                  'mcr_score', 'mcr_score_masking', 'cgrs_score', 'cgrs_score_masking', 'ach_score', 'ach_score_masking'],
        inplace=True)


#%% 
##############################################################################
##############################################################################
###                         School Outcome Features                        ###
##############################################################################
##############################################################################

# Included school outcome features because these can indicate school effectiveness. The School Performance
# Score is based on specific End of Grade or End of Course tests.

############################################################
###                   Graduation Rates                   ###
############################################################
# Import 4-year graduation rates and 5-year graduation rates

# Read in files
grad4=pd.read_excel(dir + '4yGraduation.xlsx')
grad5=pd.read_excel(dir + '5yGraduation.xlsx')

# Generate function for grad rate cleaning
def grad_clean(grad, yr):
    # Rename columns for easier merging
    grad.rename(columns={'school_code': 'agency_code', 'pct': 'GradRate'+str(yr)+'Yr'}, inplace=True)
    grad.drop(columns=['reporting_year', 'cgr_type', 'subgroup', 'denominator', 'school_name', 'grade_span', 'sbe_region'], 
              inplace=True)
    # Values >95 and <5 are suppressed for student protection.
    # I assumed values of 95, 5, or 0 in these instances because some other datasets are already 
    # adjusted for suppression and I wanted to remain consistent in methods
    grad['GradRate' + str(yr) + 'Yr'].replace(['>95', '<5', '*'], [95, 5, 0], inplace=True)
    # Drop of LEA info
    grad=grad[~grad['agency_code'].str.endswith('LEA')]
    # Drop State of North Carolina
    grad.drop(grad.tail(1).index, inplace=True)
    return grad

# Loop through functions
grad = [grad4, grad5]
vals = [4, 5]

for i in range(0, len(grad)):
    grad[i] = yr_filt(grad[i], 'reporting_year')
    grad[i] = sub_filt(grad[i])
    grad[i] = grad_clean(grad[i], vals[i])
    grad[i] = ind_ag(grad[i])

grad4 = grad[0]
grad5 = grad[1]


############################################################
###                      ACT Scores                      ###
############################################################
ACT=pd.read_excel(dir + 'rcd_acc_act.xlsx')

def act(ACT):
    # Drop LEA and state info
    ACT=ACT[~ACT['agency_code'].str.endswith('LEA')]
    ACT=ACT[ACT['agency_code']!='00B000']
    # Drop unnecessary columns
    ACT.drop(columns=['year', 'subgroup', 'den', 'masking'], inplace=True)
    # Make table wide
    ACT=ACT.pivot(index='agency_code', columns='subject', values='pct').reset_index()
    # Drop State of North Carolina
    ACT.drop(ACT.tail(1).index, inplace=True)
    return ACT

ACT = yr_filt(ACT)
ACT = sub_filt(ACT)
ACT = act(ACT)
ACT = ind_ag(ACT)


############################################################
###                      SAT Scores                      ###
############################################################
SAT=pd.read_excel(dir + 'rcd_sat.xlsx')

# No data specific function needed
SAT = yr_filt(SAT)
SAT = lea_sea(SAT)

# Drop columns
SAT.drop(columns=['year', 'category_code'], inplace=True)
# Change agency code to string
SAT.agency_code=SAT.agency_code.apply(str).str.zfill(6)

SAT = ind_ag(SAT)


############################################################
###                   Advanced Placement                 ###
############################################################
AP=pd.read_excel(dir + 'rcd_ap.xlsx')

# No data specific function needed
AP = yr_filt(AP)

# Drop year
AP.drop(columns=['year'], inplace=True)
# Change agency code to string
AP.agency_code=AP.agency_code.apply(str).str.zfill(6)

AP = ind_ag(AP)


#%% 
##############################################################################
##############################################################################
###                      School Population Demographics                    ###
##############################################################################
##############################################################################

# Included School Demographics because SPGs have been criticized for being indicative of poverty, opportunity,
# race, and access.

############################################################
###                    Race and Gender                   ###
############################################################
RaceGender=pd.read_csv(dir + 'ec_pupils.csv')

def race_gend_clean(RaceGender):
    # Rename
    RaceGender.columns=RaceGender.columns.str.replace(' ', '_')
    # Change total to numeric
    RaceGender['Total']=RaceGender['Total'].str.replace(',', '').apply(float)
    
    # Gender totals
    for gend in ['Female', 'Male']:
        gend_list = []
        for i in range(0, len(RaceGender.columns)):
            if (RaceGender.columns[i].endswith(gend)):
                gend_list.append(RaceGender.columns[i])
        RaceGender[gend+'_Tot']=RaceGender[gend_list].sum(axis=1).divide(RaceGender['Total'])
    
    # Race totals
    for race in ['Indian', 'Asian', 'Hispanic', 'Black', 'White', 'Pacific_Island', 'Two_or__More']:
        RaceGender[race]=RaceGender[[race+'_Male', race+'_Female']].sum(axis=1).divide(RaceGender['Total'])
    
    # Rename for consistency and interpretability
    RaceGender.rename(columns={'Two_or__More':'Multiracial','Pacific_Island':'PacificIsland' }, inplace=True)
    
    # Create agency_code
    RaceGender['LEA']=RaceGender['LEA'].apply(str).str.zfill(3)
    RaceGender['School']=RaceGender['School'].apply(str)
    RaceGender['agency_code']=RaceGender['LEA']+RaceGender['School']
    # Drop extra columns
    RaceGender.drop(columns=['Year', 'LEA', 'LEA_Name', 'School', 'School_Name', 'Indian_Male', 'Indian_Female', 'Asian_Male', 
                             'Asian_Female', 'Hispanic_Male', 'Hispanic_Female', 'Black_Male', 'Black_Female', 'White_Male',
                             'White_Female', 'Pacific_Island_Male', 'Pacific_Island_Female', 'Two_or__More_Male', 
                             'Two_or__More_Female'], inplace=True)
    return RaceGender

RaceGender = race_gend_clean(RaceGender)
RaceGender = ind_ag(RaceGender)


############################################################
###                 Free and Reduced Lunch               ###
############################################################
# This gets at poverty levels in the county
FreeRed=pd.read_excel(dir+'FreeReducedLunch.xlsx')

def free_clean(FreeRed):
    # Concatenate ID and School ID to get agency_code
    FreeRed['SFA #']=FreeRed['SFA #'].apply(str).str.zfill(3)
    FreeRed['Site #']=FreeRed['Site #'].apply(str)
    FreeRed['agency_code']=FreeRed['SFA #']+FreeRed['Site #']
    # Create CEP column
    FreeRed.rename(columns={'Provision': 'CEP_Eligible', '%EDS': 'EDS%', 'SFA Name': 'LEA_Name'}, inplace=True)
    FreeRed['CEP_Eligible'].replace({'CEP': 1}, inplace=True)
    # Fill in 0s for missing Eligible Values
    FreeRed['CEP_Eligible'].fillna('0', inplace=True)
    # Drop unnecessary columns
    FreeRed.drop(columns=['SFA #', 'Site #', 'Site Name'], inplace=True)
    # Replace less than 20 with 19
    FreeRed['Reduced'].replace({'less than 20': 20}, inplace=True)
    FreeRed['Free'].replace({'less than 20': 20}, inplace=True)
    # Remove Total info
    FreeRed=FreeRed[~FreeRed['agency_code'].str.contains('Total')].copy()
    # Drop State of North Carolina
    FreeRed.drop(FreeRed.tail(2).index, inplace=True)
    return FreeRed
    
FreeRed = free_clean(FreeRed)
FreeRed = ind_ag(FreeRed)

############################################################
###             Economically Disadvantaged Youth         ###
############################################################
# Gets at poverty
EDS=pd.read_excel(dir + 'rcd_acc_eds.xlsx')

# No data specific function needed
EDS = yr_filt(EDS)
EDS = lea_sea(EDS)
EDS = ind_ag(EDS)

# Drop unnecessary columns
EDS.drop(columns=['year', 'pct_eds_masking'], inplace=True)


#%% 
##############################################################################
##############################################################################
###                     School Attendance and Suspensions                  ###
##############################################################################
##############################################################################

# Attendance is necessary to learning. Attendance rates and suspensions are key factors related to learning.
# Large class sizes can also impact learning due to ability to receive small group and one on one attention
# which is similar in nature to attendance impacting learning.

############################################################
###                 Average Daily Membership             ###
############################################################
ADM=pd.read_excel(dir + 'rcd_adm.xlsx')

# No data specific function needed
ADM = yr_filt(ADM)
ADM = ind_ag(ADM)

# Drop columns
ADM.drop(columns=['year','category_code'], inplace=True)
# Rename
ADM.rename(columns={'avg_student_num': 'ADM'}, inplace=True)


############################################################
###                      Attendance                      ###
############################################################
ATT=pd.read_excel(dir + 'rcd_att.xlsx')

# No data specific function needed
ATT = yr_filt(ATT)
ATT = ind_ag(ATT)

# Drop category code
ATT.drop(columns=['category_code'], inplace=True)


############################################################
###                 Suspensions and Crimes               ###
############################################################
Incidences=pd.read_excel(dir + 'rcd_inc1.xlsx')

def inc_clean(Incidences):
    # Drop columns
    Incidences.drop(columns=['year', 'category_code'], inplace=True)
    # Change to object
    Incidences['agency_code']=Incidences['agency_code'].apply(str)
    # Change agency code to string
    Incidences.agency_code=Incidences.agency_code.apply(str).str.zfill(6)
    return Incidences

Incidences = yr_filt(Incidences)
Incidences = lea_sea(Incidences)
Incidences = inc_clean(Incidences)
Incidences = ind_ag(Incidences)


############################################################
###                       Class Size                     ###
############################################################
Class_Size=pd.read_excel(dir + 'rcd_sar.xlsx')

# No data specific function needed
Class_Size = yr_filt(Class_Size)

# Drop columns
Class_Size.drop(columns=['year', 'category_code'], inplace=True)

Class_Size = lea_sea(Class_Size)

# Aggregate average class size
Class_Size=Class_Size.groupby(Class_Size['agency_code']).mean().reset_index()

Class_Size = ind_ag(Class_Size)


#%% 
##############################################################################
##############################################################################
###                             School Resources                           ###
##############################################################################
##############################################################################

# School resources are often tied to school growth and learning since funds can help students access necessary
# resources, hire great teachers, or hire teachers beyond what the state can fund.

############################################################
###               Media and Book Information             ###
############################################################

# Average number of books
AvgBooks=pd.read_excel(dir + 'AvgNumBookTitles.xlsx')
# Media Collection Age
MediaAge=pd.read_excel(dir + 'MediaCollectionAge.xlsx')
# School 1-1 program
School1_1=pd.read_excel(dir + 'School1-1.xlsx')
# Wireless Access Points
WAP=pd.read_excel(dir + 'WAP.xlsx')

def media_clean(df, school='School Name1'):
    # Rename columns
    df.rename(columns={'School Number': 'agency_code'}, inplace=True)
    # Drop columns
    df.drop(columns=[school, 'LEA Name'], inplace=True)
    #change agency code to string
    df.agency_code=df.agency_code.apply(str).str.zfill(6)
    # Set index
    df.set_index('agency_code', inplace=True)
    return df

AvgBooks = media_clean(AvgBooks, school='School Name')
MediaAge = media_clean(MediaAge)
School1_1 = media_clean(School1_1)
WAP = media_clean(WAP)


############################################################
###                  Per pupil expenditure               ###
############################################################
Funds=pd.read_excel(dir + 'rcd_funds.xlsx')

# Values for looping over rank
ranks = ['Local', 'state', 'federal', 'total']
vals = ['local_perpupil', 'state_perpupil', 'federal_perpupil', 'total_expense']

def exp_clean(Funds):
    # Drop columns
    Funds=Funds[['agency_code', 'total_expense', 'local_perpupil', 'state_perpupil', 'federal_perpupil']]
    # Keep LEAs (funding is typically assigned locally and you don't have a school by school information piece)
    Funds=Funds[Funds['agency_code'].str.endswith('LEA')]
    # Generate ranks
    for j in range(0, len(ranks)):
        Funds[ranks[j] + '_rank'] = Funds[vals[j]].rank(method ='max', ascending=False) 
    
    # Rename agency code since it's really the LEA code
    Funds.rename(columns={'agency_code':'LEA'}, inplace=True)
    # Split LEA
    Funds['LEA Number']=Funds['LEA'].astype(str).str[0:3]
    # Change agency code to string
    Funds['LEA Number']=Funds['LEA Number'].apply(str).str.zfill(3)
    # Drop LEA 
    Funds.drop(columns=['LEA'], inplace=True)
    return Funds

Funds = yr_filt(Funds)
Funds = exp_clean(Funds)
Funds = ind_ag(Funds, ind='LEA Number')


############################################################
###                   Teacher Supplements                ###
############################################################
supplement=pd.read_excel(dir + 'Local_Salary.xlsx')

def sup_clean(supplement):
    # Drop years
    supplement.drop(columns=['Year', 'Teacher No. of Position', 'Teacher No. Rec. Supplmt.', 'Principal No. of Position',
                            'PrincipalNo. Rec. Supplmt.', 'Assistant PrincipalsNo. of Position', 
                            'Assistant PrincipalsNo. Rec. Supplmt.'], inplace=True)
    # Rename columns
    supplement.rename(columns={'Teacher Average Supplmt.': 'AvgTeachSuppl',
                               'Principal Average Supplmt.': 'AvgPrincSuppl',
                               'Assistant PrincipalsAverage Supplmt.': 'AvgAssistPrincSuppl',
                               'LEA': 'LEA Number'}, inplace=True)
    # Change LEA number to string
    supplement['LEA Number']=supplement['LEA Number'].apply(str).str.zfill(3)
    return supplement

supplement = sup_clean(supplement)
supplement = ind_ag(supplement, 'LEA Number')


#%% 
##############################################################################
##############################################################################
###                             Teacher Features                           ###
##############################################################################
##############################################################################

# Teacher effectiveness ratings, experience, certifications, degrees, and licenses may contribute to student
# learning due to training or lessons learned over time.

############################################################
###                  Teacher Effectiveness               ###
############################################################
Teach_Effect=pd.read_excel(dir + 'rcd_effectiveness.xlsx')

def eff_clean(Teach_Effect):
    # Select only student growth
    Teach_Effect=Teach_Effect[Teach_Effect['ee_standard']=='TS']
    # Drop columns
    Teach_Effect.drop(columns=['year', 'category_code', 'count', 'ee_standard'], inplace=True)
    # Make table wide
    Teach_Effect=Teach_Effect.pivot(index='agency_code', columns='ee_rating', values='pct_rating').reset_index()
    # Remove spaces from column names
    Teach_Effect.columns = Teach_Effect.columns.str.replace(' ', '_')
    # Replace missing with 0
    Teach_Effect.fillna(0)
    return Teach_Effect

Teach_Effect = yr_filt(Teach_Effect)
Teach_Effect = lea_sea(Teach_Effect)
Teach_Effect = eff_clean(Teach_Effect)
Teach_Effect = ind_ag(Teach_Effect)


############################################################
###                   Teacher Experience                 ###
############################################################
Teach_Experience=pd.read_excel(dir + 'rcd_experience.xlsx')

# No data specific function needed
Teach_Experience = yr_filt(Teach_Experience)

# Only teachers
Teach_Experience=Teach_Experience[Teach_Experience['staff']=='Teacher']
# Drop columns
Teach_Experience.drop(columns=['year', 'category_code', 'staff', 'total_class_teach'], inplace=True)

Teach_Experience = lea_sea(Teach_Experience)

Teach_Experience = ind_ag(Teach_Experience)


############################################################
###                    Teacher Licenses                  ###
############################################################
Teach_Licenses=pd.read_excel(dir + 'rcd_licenses.xlsx')

# No data specific function needed
Teach_Licenses = yr_filt(Teach_Licenses)

# Drop columns
Teach_Licenses.drop(columns=['year', 'category_code'], inplace=True)

Teach_Licenses = lea_sea(Teach_Licenses)

Teach_Licenses = ind_ag(Teach_Licenses)


############################################################
###              National Board Certification            ###
############################################################
Teach_NBPTS=pd.read_excel(dir + 'rcd_nbpts.xlsx')

# No data specific function needed
Teach_NBPTS = yr_filt(Teach_NBPTS)
Teach_NBPTS = lea_sea(Teach_NBPTS)
Teach_NBPTS = ind_ag(Teach_NBPTS)

# Drop columns
Teach_NBPTS.drop(columns=['year', 'category_code', 'total_nbpts_num'], inplace=True)


############################################################
###                     Teacher Degrees                  ###
############################################################
Teach_Degrees=pd.read_excel(dir + 'TeacherDegrees.xlsx')

def deg_clean(Teach_Degrees):
    # Creating agency_code
    Teach_Degrees['agency_code']=Teach_Degrees['LEA#']+Teach_Degrees['Sch#']
    # Drop extra columns
    Teach_Degrees.drop(columns=['LEA#', 'Sch#', 'LEA Name', 'School Name'], inplace=True)
    # Change agency code to string
    Teach_Degrees.agency_code=Teach_Degrees.agency_code.apply(str).str.zfill(6)
    return Teach_Degrees

Teach_Degrees = deg_clean(Teach_Degrees)
Teach_Degrees = ind_ag(Teach_Degrees)


#%% 
##############################################################################
##############################################################################
###                             Merge Data Files                           ###
##############################################################################
##############################################################################

################################################################################## 
###                        Join in supplement information                      ###
###      (this is the only dataset that needs to be joined by LEA Number)      ###
################################################################################## 
# Use school information file as the base
schoolsup=school.join(supplement, on='LEA Number').join(Funds, on='LEA Number')

# Set index to agency code for joining the remaining tables
schoolsup.set_index('agency_code', inplace=True)

# Remaining tables to join
tables=[SPG, grad4, grad5, FreeRed, RaceGender, ACT, SAT, EDS, ADM, AP, ATT, AvgBooks, MediaAge, School1_1, WAP, 
        Incidences, Class_Size, Teach_Effect, Teach_Experience, Teach_Licenses, Teach_NBPTS, Teach_Degrees]

# Use the basis as the dataframe in prior section
total=schoolsup

# Loop over and join tables together
for table in tables:
    total=total.join(table, on='agency_code')


#%% 
##############################################################################
##############################################################################
###                      Calculations and Data Cleaning                    ###
##############################################################################
##############################################################################

############################################
### Create percentages and rates per 100 ###
############################################

# Function for rate/percentage
def pct(out, num, denom='ADM'):
    total[out] = total[num].divide(total[denom])*100
    return total

out_list = ['FreePct','ReducedPct','crimeper100','shortsusper100','longsusper100','expulsionper100']
num_list = ['Free','Reduced','crime','short_term','long_term','expulsion']

for k in range(0, len(out_list)):
    total = pct(out_list[k], num_list[k])
    
################################################
###           Final Data Cleaning            ###
################################################
# Remove spaces from column names
total.columns=total.columns.str.replace(' ', '_')

# Split locale type
total[['Locale1', 'Locale2']]=total.Locale_Type_Desc.str.split(', ', expand=True) 
# Removing comma from locale type
total.Locale_Type_Desc=total.Locale_Type_Desc.str.replace(', ', '')

# Remove space from region
total.SBE_Region_Names=total.SBE_Region_Names.str.replace(' ', '')

# Change values for School1_1 to binary
total['School1-1']=total['School1-1'].str.replace('Yes', '1')
total['School1-1']=total['School1-1'].str.replace('No', '0')

# Create dummy variables for several columns
total=pd.get_dummies(total, prefix=['SBERegion', 'SchoolType', 'Calendar', 'LocaleType', 'Locale', 'Grade'],
               columns=['SBE_Region_Names', 'School_Type', 'School_Calendar', 'Locale_Type_Desc', 'Locale1', 'category_code'], 
                     drop_first=True)

# Remove spaces from column names
total.columns=total.columns.str.replace(' ', '_')


################################################
###   Additional Preparation for Analysis    ###
################################################

# Replace missing values with 0
total.fillna(0, inplace=True)

# Remove data without a spg score (main outcome)
total=total.loc[(total['spg_score'].notna())&(total['spg_score']!=0), ]

# View data
total.head()

#%% Write to a file to use in analysis
total.to_csv(dir + "NCSchoolData_analysis.csv")


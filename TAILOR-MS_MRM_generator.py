#TriAcylglycerol Identifier for Low Resolution Mass Spectrometers (TAILOR-MS) 08/10/2019, by Kang-Yu Peng
#The script is a useful tool for setting MRM transitions on a mass spectrometer and deciphering TG structures.
#It consists of two parts.
#For the first part, it creates a list of triacylglycerol MRM transitions (Q1 and Q3) based with selected nominal fatty acyl groups.
#For the second part, it determines the most possible TG structures based on peak intensities (area) and time information of the LC-MS data.
import itertools
import pandas as pd

#PART I: generate all possible fatty acyl combinations for TG using the selected fatty acids and create a list of unique MRMs for the use in instrumental acquisition method
#List all the possible combinations with the given fatty acids and save them as a dataframe
FA_input = pd.read_csv('FA_MRM.csv',delimiter=',',header=0)
df = FA_input #Selected FAs
cwr = list(itertools.combinations_with_replacement(df['Fatty_acid'],3))
df = pd.DataFrame(cwr,columns=['FA1','FA2','FA3'])
df['TG structure'] = 'TG(' + df['FA1'] + '_' + df['FA2'] +'_' + df['FA3'] +')'

#Exclude TG structures that don't meet the "can't appear more than once" rule
from collections import Counter

Reapp_YN = FA_input.set_index('Fatty_acid')
Reapp_N = Reapp_YN.loc[Reapp_YN.loc[:, 'Reappearance'] == 'N'] #FAs that are only allowed to appear once are in here
Reapp_Y = Reapp_YN.loc[Reapp_YN.loc[:, 'Reappearance'] == 'Y'] #FAs that can appear more than once are in here
if Reapp_Y.empty ==True:
    Reapp_Y =pd.DataFrame(['YE']).rename(index={0:'99x99'},columns={0:'Reappearance'}) #Avoid Empty DataFrame
if Reapp_N.empty ==True:
    Reapp_N =pd.DataFrame(['NE']).rename(index={0:'99x99'},columns={0:'Reappearance'}) #Avoid Empty DataFrame. This should never occur as no structures will exist

def Label_undesired_rep_FA(x):
    df_FA_YN = df.loc[x, ['FA1', 'FA2', 'FA3']]
    df_FA_YN = df_FA_YN.replace(list(Reapp_N.index), 'N').replace(list(Reapp_Y.index), 'Y') #Replace the FAs with Y or N labels
    cnt = Counter()
    for Cnt_FA in df_FA_YN:
        cnt[Cnt_FA] += 1
    Count_result = pd.DataFrame.from_dict(cnt, orient='index')
    Reapp_label = Count_result.loc[Count_result[0] > 1].rename(columns={0: 'Label_reapp'}).reset_index().iloc[:,0]  #Y labelled TG structures (ie having 2 or 3 replicates of FA in Y list) will be kept later
    return Reapp_label
Label_Reappearance = pd.DataFrame([Label_undesired_rep_FA(x) for x in df.index]).rename(columns={0:'Reapp'}).reset_index(drop=True)
df['Reapp'] = Label_Reappearance
df = df.loc[df['Reapp']=='Y']

#Create columns with carbon chain and double bond numbers only and concatenate with the original dataframe
df_FA1_info = df['FA1'].str.split('x',expand=True).rename(columns={0:'FA1-C',1:'FA1-DB'}).astype('int64')
df_FA2_info = df['FA2'].str.split('x',expand=True).rename(columns={0:'FA2-C',1:'FA2-DB'}).astype('int64')
df_FA3_info = df['FA3'].str.split('x',expand=True).rename(columns={0:'FA3-C',1:'FA3-DB'}).astype('int64')

df = pd.concat([df,df_FA1_info,df_FA2_info,df_FA3_info],axis=1,join ='inner',sort=False)

#Set up Q1/Q3 MRMs using LipidMaps generated m/z (+ ion mode, with one NH4+ adduct)
df_C = df.loc[:,df.columns.str.endswith('-C')].astype('int64')
df['Total-C'] = df_C.iloc[:,0] + df_C.iloc[:,1] + df_C.iloc[:,2]
df_DB = df.loc[:,df.columns.str.endswith('-DB')].astype('int64')
df['Total-DB'] = df_DB.iloc[:,0] + df_DB.iloc[:,1] + df_DB.iloc[:,2] #Calculte total FA carbon and double bond numbers for each TG molecule

df['Q1'] = round(152.019 + df['Total-C']*14.01565 - df['Total-DB']*2.015655,4) #Calculate Q1 masses, 152.019 is TG backbone + NH4+; each carbon adds 14.01565; each double bond deducts 2.0157

df['Q3-FA1'] = round(df['Q1'] - 49.0164 -14.01565*df['FA1-C'] + 2.015655*df['FA1-DB'],4) #Q3 m/z (neutral loss of 1st FA)
df['Q3-FA2'] = round(df['Q1'] - 49.0164 -14.01565*df['FA2-C'] + 2.015655*df['FA2-DB'],4) #Q3 m/z (neutral loss of 2nd FA)
df['Q3-FA3'] = round(df['Q1'] - 49.0164 -14.01565*df['FA3-C'] + 2.015655*df['FA3-DB'],4) #Q3 m/z (neutral loss of 3rd FA)

df['Brutto level TG'] = 'TG(' + df['Total-C'].astype('str') + 'x' + df['Total-DB'].astype('str') +')'

df_summary = df[['Brutto level TG','TG structure']].sort_values(by=['Brutto level TG','TG structure'],axis=0).reset_index(drop=True)
df_summary.index = df_summary.index.rename('No.') + 1

#Create a comprehensive and non-redundant MRM list
MRM_L_FA1 = df[['Brutto level TG','Q1','FA1','Q3-FA1']].rename(columns={'FA1':'FA','Q3-FA1':'Q3'})
MRM_L_FA1['Q1_Q3 Identity'] = MRM_L_FA1['Brutto level TG'] + '_' + MRM_L_FA1['FA']
MRM_L_FA1I = MRM_L_FA1.set_index('Q1_Q3 Identity')

MRM_L_FA2 = df[['Brutto level TG','Q1','FA2','Q3-FA2']].rename(columns={'FA2':'FA','Q3-FA2':'Q3'})
MRM_L_FA2['Q1_Q3 Identity'] = MRM_L_FA1['Brutto level TG'] + '_' + MRM_L_FA2['FA']
MRM_L_FA2I = MRM_L_FA2.set_index('Q1_Q3 Identity')

MRM_L_FA3 = df[['Brutto level TG','Q1','FA3','Q3-FA3']].rename(columns={'FA3':'FA','Q3-FA3':'Q3'})
MRM_L_FA3['Q1_Q3 Identity'] = MRM_L_FA1['Brutto level TG'] + '_' + MRM_L_FA3['FA']
MRM_L_FA3I = MRM_L_FA3.set_index('Q1_Q3 Identity')

MRM_L = pd.concat([MRM_L_FA1I[['Q1','Q3']],MRM_L_FA2I[['Q1','Q3']],MRM_L_FA3I[['Q1','Q3']]],axis=0,join='inner',sort=True).drop_duplicates().sort_values(by=['Q1_Q3 Identity'],axis=0) #Full list of MRMs for aquisition method setup

df_summary.to_csv('TG_by_MRM.csv') #1st csv file
MRM_L.to_csv('MRM_list.csv') #2nd csv file
#End of Part 1=======================================================================================================
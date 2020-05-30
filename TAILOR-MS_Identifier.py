#TriAcylglycerol Identifier for Low Resolution Mass Spectrometers (TAILOR-MS) 08/10/2019, by Kang-Yu Peng
#The script is a useful tool for setting MRM transitions on a mass spectrometer and deciphering TG structures.
#It consists of two parts.
#For the first part, it creates a list of triacylglycerol MRM transitions (Q1 and Q3) based with selected nominal fatty acyl groups.
#For the second part, it determines the most possible TG structures based on peak intensities (area) and time information of the LC-MS data.
import itertools
import pandas as pd
import numpy as np
import sys
#TAILOR-MS Identifier: Decipher sn1, sn2 and sn3 fatty acids of TGs using retention time and abundance information, based on fatty acid neutral loss input data.
FA_input = pd.read_csv('FA.csv',delimiter=',',header=0)
rdf = pd.read_csv('Input.csv',delimiter=',',header=0) #Read input data file
rdf['Time_dif'] = rdf['RT_right'] - rdf['RT_left'] #Calculate retention time ranges

#Ensure non-negative values from input data. Also RT_right-RT_left (Time_dif) must > 0. If conditions not met, exit the program and throw an error message
Test_neg_0 = (rdf.loc[rdf['RT_left'] <0].empty & rdf.loc[rdf['RT_right'] <=0].empty & rdf.loc[rdf['Time_dif'] <=0].empty & rdf.loc[rdf['Intensity'] <=0].empty & rdf.loc[rdf['Abundance_threshold(%)'] <0].empty & rdf.loc[rdf['RT_tolerance(%)'] <=0].empty) == True

if Test_neg_0 == False:
    sys.exit('Error: Negative and/or 0 values are present in input dataset. Check values in RT_left, RT_right, Time_dif, Area, Abundance_threshold(%) and RT_tolerance(%) columns.')

#Step1, calculate area for each NL peak, and then calculate the relative abundances
CTG = rdf['TG'].drop_duplicates().reset_index(drop=True)#List all TGs

def cal_rel_abu(x): #Caluculate relative abundances vs ID peak (ie peak with the largest area reading of the TGs with the same carbon number and double bonds(FAs not considered))
    a = rdf.loc[rdf['TG'] ==x]
    Area_max = max(a['Intensity'])
    a['Rel_abundance(%)'] = (a['Intensity'].div(Area_max)*100).round(2) #Calculate relative abundance (to the maximum peak)
    return a.reset_index(drop=True)

rdf2 = pd.concat([cal_rel_abu(x) for x in CTG],axis=0,join='inner',sort=False).reset_index(drop=True) #Dataframe with the relative abundances of all peaks
rdf2 = rdf2.loc[rdf2['Rel_abundance(%)']>0].reset_index(drop=True) #Remove area 0, which could have both RT left and RT right equivalent to 0, a situation that creates problems for overlap match.

#Step2, create all possible combinations for each TG, using the detected FA neutral losses
def Three_FA(x):
    rTG_info = rdf2.loc[rdf2['TG']==x,'TG'].str.split('x',expand=True).rename(columns={0:'TG-C',1:'TG-DB'}).astype('int64').drop_duplicates()
    b = list(itertools.combinations_with_replacement(rdf2.loc[rdf2['TG']==x,'FA'],2))
    c = pd.DataFrame(b,columns=['FA1','FA2'])#c dataframe is
    c['TG'] = x
    rFA1_info = c['FA1'].str.split('x',expand=True).rename(columns={0:'FA1-C',1:'FA1-DB'}).astype('int64')
    rFA2_info = c['FA2'].str.split('x',expand=True).rename(columns={0:'FA2-C',1:'FA2-DB'}).astype('int64')
    c = pd.concat([c,rFA1_info,rFA2_info],axis=1,join='outer',sort=False)
    c['FA3-C'] = rTG_info['TG-C'].values - c['FA1-C'].add(c['FA2-C'])
    c['FA3-DB'] = rTG_info['TG-DB'].values - c['FA1-DB'].add(c['FA2-DB'])
    c['FA3'] = c['FA3-C'].astype('str') + 'x' + c['FA3-DB'].astype('str')
    return c.drop(c[c['FA3-DB']<0].index,axis=0).drop(c[c['FA3-C']<0].index,axis=0)

rFA_df = pd.concat([Three_FA(x) for x in CTG],axis=0,join='outer',sort=False).reset_index(drop=True)
rFA_df = rFA_df.loc[rFA_df['FA3'].isin(values=FA_input.squeeze())] #Remove the third FAs that cannot be found in input FA list.(There are still redundant FA1,FA2 and FA3 combinations).
rFA_df = rFA_df[['TG','FA1','FA1-C','FA1-DB','FA2','FA2-C','FA2-DB','FA3','FA3-C','FA3-DB']].reset_index(drop=True)

def sort_FA_uniq_order(x): #This function fixes the order of FA1, FA2 and FA3 and generates TG structures with FA information with FAs in correct order. The removal of redundant TGs is done subsequently.
    d = pd.concat([rFA_df.iloc[x, [1,2,3]].reset_index(drop=True),
                   rFA_df.iloc[x, [4,5,6]].reset_index(drop=True),rFA_df.iloc[x,[7,8,9]].reset_index(drop=True)], axis=1, join='inner', sort=False, ignore_index=True)
    d = d.sort_values(by=[1,2],axis=1,ascending=True).T.reset_index(drop=True) #Sort the 3 FAs according to their carbon chain length and double bond numbers
    return d[0]

Sorted_FA = pd.DataFrame([sort_FA_uniq_order(x) for x in list(rFA_df.index)]).reset_index(drop=True).rename(columns={0:'FA1',1:'FA2',2:'FA3'})
Sorted_FA['TG_structure'] = 'TG(' + Sorted_FA['FA1'].astype('str') + '_' + Sorted_FA['FA2'].astype('str') + '_' + Sorted_FA['FA3'].astype('str') + ')' #This is used to identify redundant TGs
rFA_df_SU = pd.concat([rFA_df['TG'],Sorted_FA],axis=1,join='outer',sort=False).set_index('TG_structure',drop=True).drop_duplicates()#Redundant TG structures are removed here

#Exclude FA1, FA2 and FA3 combinations without overlapped retention time
def TG_identification(w): #This function contains three steps: (1)Generate a list that has identifiable TGs, based on the Q1/Q3 in aquisition data (2)Determine overlap (3) Write into the result table
    Mock_FA = pd.DataFrame([rFA_df_SU.loc[w,'TG']+'_#',rFA_df_SU.loc[w,'TG'],'#','#',0,max(rdf2['RT_right'])+1,0,0,0,max(rdf2['RT_right'])+1,101],columns=['#'],index=rdf2.columns).T #!!!(abundance and RT threshold)Create mock FA to be used when only two FAs are used to determine TG structure

    res_FA1 = rdf2.loc[rdf2['TG']== rFA_df_SU.loc[w,'TG']].loc[rdf2['FA']== rFA_df_SU.loc[w,'FA1']] #This part needs to be written as list comprehension
    if len(res_FA1.index) == 0:
        res_FA1 = Mock_FA
    else:
        res_FA1 = res_FA1.set_index('Peak',drop=True)#Index FA1 NLs eluting at different RTs (ie a, b, c etc. peaks). If FA1 doesn't exist, substitute the empty dataframe with mock FA dataframe.

    res_FA2 = rdf2.loc[rdf2['TG']== rFA_df_SU.loc[w,'TG']].loc[rdf2['FA']== rFA_df_SU.loc[w,'FA2']]
    if len(res_FA2.index) == 0:
        res_FA2 = Mock_FA
    else:
        res_FA2 = res_FA2.set_index('Peak',drop=True)#Index FA2 NLs eluting at different RTs (ie a, b, c etc. peaks). If FA2 doesn't exist, substitute the empty dataframe with mock FA dataframe.

    res_FA3 = rdf2.loc[rdf2['TG']== rFA_df_SU.loc[w,'TG']].loc[rdf2['FA']== rFA_df_SU.loc[w,'FA3']]
    if len(res_FA3.index) == 0:
        res_FA3 = Mock_FA
    else:
        res_FA3 = res_FA3.set_index('Peak',drop=True)#Index FA3 NLs eluting at different RTs (ie a, b, c etc. peaks). If FA3 doesn't exist, substitute the empty dataframe with mock FA dataframe.

    FA_combination_list = []
    for x in res_FA1.index:
        for y in res_FA2.index:
            for z in res_FA3.index:
                FA_combination_list.append((x,y,z)) #Create a combination list using the above FA1, FA2 and FA3 NLs with different RTs, as (x,y,z)

    FA_combination_df = pd.DataFrame(FA_combination_list,columns=['FA1','FA2','FA3'])

    #Determine if the three FAs overlap. Find the overlapped time segment (time left, time right). Compare it to the FA with least abundance of the three and see if it overlaps with the predetermined time segment
    def overlap(x):  # Determine if the three FA time segments overlap.
        FA1_RT = res_FA1.loc[FA_combination_df.iloc[x, 0], ['RT_left', 'RT_right']]
        FA2_RT = res_FA2.loc[FA_combination_df.iloc[x, 1], ['RT_left', 'RT_right']]
        FA3_RT = res_FA3.loc[FA_combination_df.iloc[x, 2], ['RT_left', 'RT_right']]

        if FA1_RT[1] >= FA2_RT[0] and FA2_RT[1] >= FA1_RT[0] and FA1_RT[1] >= FA3_RT[0] and FA3_RT[1] >= FA1_RT[0] and FA2_RT[1] >= FA3_RT[0] and FA3_RT[1] >= FA2_RT[0]:
            return (max(FA1_RT[0],FA2_RT[0],FA3_RT[0]),min(FA1_RT[1],FA2_RT[1],FA3_RT[1])) #RT coverage (left and right) for the overlapped time segment
        else:
            return (np.nan,np.nan) #Label combinations that don't overlap with nan values

    df_overlap_t = pd.DataFrame([overlap(x) for x in FA_combination_df.index],columns=['RT_left','RT_right'])
    FA_combination_df_overlap = pd.concat([FA_combination_df, df_overlap_t], axis=1, join='inner').reset_index(drop=True)
    FA_combination_df_overlap['Time_dif'] = FA_combination_df_overlap['RT_right'] - FA_combination_df_overlap['RT_left']# Find the FAs (position 1, 2 & 3) that overlap and the overlap

    def Outcome_table(x):  # Apply abundance and overlap thresholds, then export outcomes
        FA123_info = pd.concat([res_FA1.loc[FA_combination_df_overlap.loc[x, 'FA1']],
                                res_FA2.loc[FA_combination_df_overlap.loc[x, 'FA2']],
                                res_FA3.loc[FA_combination_df_overlap.loc[x, 'FA3']]], axis=1,join='inner').T.reset_index()  # This dataframe contains all needed inf for a FA1_FA2_FA3 combination
        FA123_info.index = ['FA1', 'FA2', 'FA3']
        FA123_info = FA123_info.rename(columns={'index': 'Peak'})
        Min_FA = FA123_info.astype({'Rel_abundance(%)': 'float'})['Rel_abundance(%)'].idxmin()  # May need to distinguish FAs with identical Rel_abundance(%)
        Find_Min_FAs = FA123_info.loc[FA123_info['FA'] == FA123_info.loc[FA123_info.index == Min_FA, 'FA'][0]]  #Find if there is repetitive minimum FA for the particular strucutral combination
        Min_FAs = Find_Min_FAs.index
        Repetition = len(Min_FAs) #Use later for correction
        FA123_info.loc[Min_FAs, 'Peak'] = FA123_info.loc[Min_FAs, 'Peak'].str.capitalize() #Capitalize the FA(s) that has minimum concentration and is thus used to test relative abundance and overlap

        Find_max_abundance = rdf2.loc[rdf2['TG'] == rFA_df_SU.loc[w,'TG']]
        Max_abundance = Find_max_abundance.loc[Find_max_abundance['Rel_abundance(%)'] == 100].loc[:,['FA','Intensity']]

        if FA_combination_df_overlap.iloc[x, 5] / FA123_info.loc[Min_FA, 'Time_dif'] * 100 > FA123_info.loc[Min_FA, 'RT_tolerance(%)'] \
                and FA123_info.loc[Min_FA, 'Rel_abundance(%)'] > FA123_info.loc[Min_FA, 'Abundance_threshold(%)']:
            # Conditions being filtered out are: The non-overlapped ones (as np.nan, and therefore excluded); overlapped, but doesn't reach the threshold; not reaching relative abundance threshold

            FA_structural_outcome = pd.DataFrame({'Brutto Level': 'TG(' + FA123_info.loc['FA1', 'TG'] + ')',
                                                    'TG Structure': w,
                                                    'Constructed Peaks': FA123_info.loc['FA1', 'Peak'] + FA123_info.loc['FA2', 'Peak'] + FA123_info.loc['FA3', 'Peak'],
                                                    'ID Peak': ['TG(' + FA123_info.loc[Min_FA, 'TG'] + ')_' + FA123_info.loc[Min_FA, 'FA'] + '_' + FA123_info.loc[Min_FA, 'Peak'].lower()],
                                                    'Name' : FA123_info.loc[Min_FA, 'Name'],
                                                    'Retention Time': [str(round(FA123_info.loc[Min_FA, 'RT_left'],2)) + '-' + str(round(FA123_info.loc[Min_FA, 'RT_right'],2))],
                                                    '% Relative Abundance': round(FA123_info.loc[Min_FA, 'Rel_abundance(%)'],2),
                                                    '% Relative Abundance (corrected)': round((FA123_info.loc[Min_FA, 'Rel_abundance(%)'] / Repetition),2),
                                                    'Intensity': FA123_info.loc[Min_FA, 'Intensity'],
                                                    'Intensity (corrected)': (FA123_info.loc[Min_FA, 'Intensity'] / Repetition)})

        else:
            FA_structural_outcome = pd.DataFrame({'Brutto Level': 'TG(' + FA123_info.loc['FA1', 'TG'] + ')',
                                                    'TG Structure': w,
                                                    'Constructed Peaks': FA123_info.loc['FA1', 'Peak'] + FA123_info.loc['FA2', 'Peak'] + FA123_info.loc['FA3', 'Peak'],
                                                    'ID Peak': ['TG(' + FA123_info.loc[Min_FA, 'TG'] + ')_' + FA123_info.loc[Min_FA, 'FA'] + '_' + FA123_info.loc[Min_FA, 'Peak'].lower()],
                                                    'Name' : FA123_info.loc[Min_FA, 'Name'],
                                                    'Retention Time': np.nan,
                                                    '% Relative Abundance': round(FA123_info.loc[Min_FA, 'Rel_abundance(%)'],2),
                                                    '% Relative Abundance (corrected)': round((FA123_info.loc[Min_FA, 'Rel_abundance(%)'] / Repetition),2),
                                                    'Intensity': FA123_info.loc[Min_FA, 'Intensity'],
                                                    'Intensity (corrected)': (FA123_info.loc[Min_FA, 'Intensity'] / Repetition)})

        return FA_structural_outcome

    Outcomes = pd.concat([Outcome_table(x) for x in FA_combination_df_overlap.index],axis=0,join='inner',sort=False)

    return Outcomes

FA_struct = pd.concat([TG_identification(w) for w in rFA_df_SU.index],axis=0,join='inner',sort=False).dropna().reset_index(drop=True)

Ident_Pred = FA_struct['Constructed Peaks'].str.contains('#',regex=False).astype('str').replace({'True':'P','False':'I'}) #Label TG species based on 2 (prediction) or 3 (identification) FAs
FA_struct.insert(value=Ident_Pred,loc=0,column='Identification/Prediction')

#Sorting by creating a detailed C and DB list
FA_struct['split_use'] = FA_struct['TG Structure'].str.replace('TG(','',regex=False).str.replace(')','',regex=False)
FA_struct_sp = FA_struct['split_use'].str.split(pat='_',expand=True)
FA_struct_sp_0 = FA_struct_sp[0].str.split(pat='x',expand=True).astype('int')
FA_struct_sp_0 = FA_struct_sp_0.rename(columns={0:'FA1_C',1:'FA1_DB'})
FA_struct_sp_1 = FA_struct_sp[1].str.split(pat='x',expand=True).astype('int')
FA_struct_sp_1 = FA_struct_sp_1.rename(columns={0:'FA2_C',1:'FA2_DB'})
FA_struct_sp_2 = FA_struct_sp[2].str.split(pat='x',expand=True).astype('int')
FA_struct_sp_2 = FA_struct_sp_2.rename(columns={0:'FA3_C',1:'FA3_DB'})
FA_struct_sp_all = pd.concat([FA_struct_sp_0,FA_struct_sp_1,FA_struct_sp_2],axis=1,sort=False,join='inner')
FA_struct = pd.concat([FA_struct,FA_struct_sp_all],join='inner',axis=1,sort=False) #Ready for sorting

FA_struct = FA_struct.sort_values(by=['Brutto Level','FA1_C','FA1_DB','FA2_C','FA2_DB','FA3_C','FA3_DB']).set_index(np.arange(1,len(FA_struct.index)+1),drop=True)
FA_struct.index = FA_struct.index.rename('No.')

for x in ['Brutto Level','TG Structure','ID Peak']:
    FA_struct[x] = FA_struct[x].str.replace('x',':',regex=False)

FA_struct.iloc[:,0:11].to_csv('Results.csv') #Output file
def clean_data(df_raw):
    df = df_raw.drop(columns = ['Dizzy','Ethnicity']) #too much NA values
    #drop unnecessary columns
    df = df.drop(columns = ['PatNum','EmplType', 'Certification', 'InjuryMech', 'Intubated', 'Paralyzed', 'Sedated', 'OSIExtremity', 'OSICut', 'OSICspine', 'OSIFlank', 'OSIAbdomen',
       'OSIPelvis', 'OSIOth', 'CTForm1', 'IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx', 'IndHA', 'IndHema',
       'IndLOC', 'IndMech', 'IndNeuroD', 'IndRqstMD', 'IndRqstParent',
       'IndRqstTrauma', 'IndSeiz', 'IndVomit', 'IndXraySFx', 'IndOth', 'CTSed', 'CTSedAgitate', 'CTSedAge', 'CTSedRqst', 'CTSedOth',
       'Observed', 'EDDisposition', 'EDCT','HospHead', 'PosCT', 'DeathTBI', 'HospHeadPosCT', 'Intub24Head',
       'Neurosurgery', 'Drugs', 'Finding1', 'Finding2', 'Finding3', 'Finding4', 'Finding5', 'Finding6',
       'Finding7', 'Finding8', 'Finding9', 'Finding10', 'Finding11',
       'Finding12', 'Finding13', 'Finding14', 'Finding20', 'Finding21',
       'Finding22', 'Finding23', 'AgeInMonth'])
    #drop rows that have na values for the final column
    df = df.dropna(subset=['PosIntFinal'])

    #Loc
    df['LOCSeparate'] = df['LOCSeparate'].fillna(92)
    df['LOCSeparate'] = df['LOCSeparate'].replace(2, 92)
    mode_loclen = df.loc[df['LOCSeparate'] == 1, 'LocLen'].mode()[0]
    df.loc[(df['LOCSeparate'] == 1) & (df['LocLen'].isna()), 'LocLen'] = mode_loclen
    df.loc[(df['LOCSeparate'] == 2) & (df['LocLen'].isna()), 'LocLen'] = 92

    #Headache
    df['HA_verb'] = df['HA_verb'].fillna(91)  # Treat NaN as unknown
    mode_hastart = df.loc[df['HA_verb'] == 1, 'HAStart'].mode()[0]
    df.loc[(df['HA_verb'] == 1) & (df['HAStart'].isna()), 'HAStart'] = mode_hastart
    mode_haseverity = df.loc[df['HA_verb'] == 1, 'HASeverity'].mode()[0]
    df.loc[(df['HA_verb'] == 1) & (df['HASeverity'].isna()), 'HASeverity'] = mode_haseverity

    #GCS
    df.drop(columns=['GCSMotor', 'GCSVerbal', 'GCSEye'], inplace=True)

    #vomit
    df['Vomit'] = df['Vomit'].fillna(0) #percentage of Nan is low and most of values are 0 (No), so we use 0 instead of "Unknown".
    mode_vomit_nbr = df.loc[df['Vomit'] == 1, 'VomitNbr'].mode()[0]
    df.loc[(df['Vomit'] == 1) & (df['VomitNbr'].isna()), 'VomitNbr'] = mode_vomit_nbr
    mode_vomit_start = df.loc[df['Vomit'] == 1, 'VomitStart'].mode()[0]
    df.loc[(df['Vomit'] == 1) & (df['VomitStart'].isna()), 'VomitStart'] = mode_vomit_start
    mode_vomit_last = df.loc[df['Vomit'] == 1, 'VomitLast'].mode()[0]
    df.loc[(df['Vomit'] == 1) & (df['VomitLast'].isna()), 'VomitLast'] = mode_vomit_last
    df.loc[df['Vomit'] == 0, ['VomitNbr', 'VomitStart', 'VomitLast']] = 92

    #Seiz
    df['Seiz'] = df['Seiz'].fillna(0) #same method as Vomit, since Nan is very low and most values in Seiz are 0(No vomit)
    mode_seiz_occur = df.loc[df['Seiz'] == 1, 'SeizOccur'].mode()[0]
    df.loc[(df['Seiz'] == 1) & (df['SeizOccur'].isna()), 'SeizOccur'] = mode_seiz_occur
    mode_seiz_len = df.loc[df['Seiz'] == 1, 'SeizLen'].mode()[0]
    df.loc[(df['Seiz'] == 1) & (df['SeizLen'].isna()), 'SeizLen'] = mode_seiz_len
    df.loc[df['Seiz'] == 0, ['SeizOccur', 'SeizLen']] = 92

    #hema
    df['Hema'] = df['Hema'].fillna(0)
    df.loc[df['Hema'] == 0, ['HemaLoc', 'HemaSize']] = 92
    mode_hemaloc = df.loc[df['Hema'] == 1, 'HemaLoc'].mode()[0]
    df.loc[(df['Hema'] == 1) & (df['HemaLoc'].isna()), 'HemaLoc'] = mode_hemaloc
    mode_hemasize = df.loc[df['Hema'] == 1, 'HemaSize'].mode()[0]
    df.loc[(df['Hema'] == 1) & (df['HemaSize'].isna()), 'HemaSize'] = mode_hemasize

    #NeuroD
    df['NeuroD'] = df['NeuroD'].fillna(0)
    df.loc[df['NeuroD'] == 0, ['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']] = 92

    for col in ['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']:
        mode_value = df.loc[df['NeuroD'] == 1, col].mode()[0]
        df.loc[(df['NeuroD'] == 1) & (df[col].isna()), col] = mode_value

    #Fill Na values as 92(unapplicable) for columns with higher NA percentage than 3%
    df['ActNorm'] = df['ActNorm'].fillna(92)
    df['Race'] = df['Race'].fillna(92)
    df['Amnesia_verb'] = df['Amnesia_verb'].fillna(92)

    #columns with NA percentage less than 1%
    missing_cols = df.columns[df.isna().any()].tolist()
    for col in missing_cols:
        mode_value = df[col].mode()[0]  # Get the most frequent value
        df[col] = df[col].fillna(mode_value)

    ##Additional work for clean data

    #Modifying data structure to see distribution clearly
    df['SFxPalp'] = df['SFxPalp'].replace(2, 92) #SFxpalp 2(Unclear) is changed to 92(unapplicable)
    df['High_impact_InjSev'] = df['High_impact_InjSev'].replace(1,0) #mild impact is 0
    df['High_impact_InjSev'] = df['High_impact_InjSev'].replace(2,1) #moderate, high is 1
    df['High_impact_InjSev'] = df['High_impact_InjSev'].replace(3,1)

    #Headache severity: severe to 1, and mild, moderate to 0. 
    df['HASeverity'] = df['HASeverity'].replace({1: 0, 2: 0, 3: 1})

    if df.isna().any().any()==0:
        print("No NaN values for all columns.")
    
    return df

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# load the dataset
loans = pd.read_csv('loans.csv',sep = ',', skiprows = 6)

def preprocess_data(loans):
    #combine expired loans and refunded loans into one Not_funded
    df = loans[loans['STATUS'].isin(['funded','expired','refunded'])]
    df['STATUS'].loc[df['STATUS'].isin(['refunded','expired'])] = "Not_funded"
    df = df.reset_index()

    # the whole dataset is highly imbalanced. 95%  of the loans can get approved
    # selected only the dataset in US. Approximately 30% loans expired in the past.
    df = df[df['COUNTRY_NAME'] == 'United States']

    # remove one description in spanish
    df = df[df['ORIGINAL_LANGUAGE'] == 'English']

    # composition of borrowers
    df = df.dropna(subset = ['BORROWER_GENDERS'])
    df['FEMALE_NUM']= df.BORROWER_GENDERS.apply(lambda x: str(x).split(', ').count('female'))
    df['MALE_NUM']= df.BORROWER_GENDERS.apply(lambda x: str(x).split(', ').count('male'))

    # select all the features for the final model
    to_keep = ['SECTOR_NAME','PARTNER_ID','TAGS','DESCRIPTION','LOAN_AMOUNT','FEMALE_NUM','MALE_NUM','STATUS','REPAYMENT_INTERVAL','DISTRIBUTION_MODEL']
    new_df = df[to_keep]
    return new_df

 df = preprocess_data(loans)

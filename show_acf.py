# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:01:31 2020

@author: Stefan
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'

def show_plots(csv_input):
    df = pd.read_csv(csv_input, parse_dates=[0])
    L=len(df.index)
    idx=pd.date_range(df['data'][0], df['data'][L-1])
        
    ### TREBUIE SA COMPLETEZI DATELE LIPSA CU VALORI (CONSTANTE)
    df=df.set_index(['data'])
    upsample_df=df.reindex(idx)
    upsample_df=upsample_df.interpolate(method='time')
    upsample_df['valoare']=upsample_df['valoare'].apply(np.floor)
        
    y=upsample_df.drop(['forecast'],axis=1)
    
    y.plot(figsize=(19,4))
    plt.show()
        
    plot_acf(y)
    plot_pacf(y,lag=50)

def main():
    csv_input="date_reale.csv"
    show_plots(csv_input)
   
if __name__ == "__main__":
    main()
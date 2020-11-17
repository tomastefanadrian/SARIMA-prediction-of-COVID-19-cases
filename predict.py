# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 00:28:54 2020

@author: Stefan
"""
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

import sys
import argparse
import csv

#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.graphics.tsaplots import plot_pacf

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('--days')
    parser.add_argument('-S')
    parser.add_argument('--auto')
    parser.add_argument('-p')
    parser.add_argument('-d')
    parser.add_argument('-q')
    parser.add_argument('-P')
    parser.add_argument('-D')
    parser.add_argument('-Q')
    args = parser.parse_args()
    
    if not args.input:
        print("Usage: predict.py --input inputFile --output outFile --days 20 -p ....")
        print("Input file is mandatory")
        sys.exit(0)
    else:
        csv_input=args.input
    
    if not args.input:
        out_data="default.csv"
    else:
        out_data=args.output
    
    if not args.days:
        print("Using default number of days: 5.")
        no_days=5
    else:
        no_days=int(args.days)
    
    if not args.S:
        print("Using default seasonal parameter: 0.")
        S=0
    else:
        S=int(args.S)
    
    if not args.auto:
        print("Using default seasonal parameter: 1.")
        auto=1
    else:
        auto=int(args.auto)
    
    if not args.p:
        print("Using default regression parameter: 1.")
        p=1
    else:
        p=int(args.p)
    
    if not args.d:
        print("Using default differential parameter: 1.")
        d=1
    else:
        d=int(args.d)
    
    if not args.q:
        print("Using default average parameter: 1.")
        q=1
    else:
        q=int(args.q)
    
    if not args.P:
        print("Using default seasonal regression parameter: 1.")
        P=1
    else:
        P=int(args.P)
    
    if not args.D:
        print("Using default seasonal differential parameter: 1.")
        D=1
    else:
        D=int(args.D)
    
    if not args.Q:
        print("Using default seasonal average parameter: 1.")
        Q=1
    else:
        Q=int(args.Q)
    
    
    
    return csv_input, out_data, no_days, S, auto, p, d, q, P, D, Q
    
def forecastTimeSeries(y,no_days=100,S=2,p=1,d=1,q=0,P=1,D=1,Q=1,auto=0):
    
    #
    if (auto==1):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], S) for x in list(itertools.product(p, d, q))]
        
        print('Examples of parameter combinations for Seasonal ARIMA...')
        print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
        
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
        
                    results = mod.fit()
        
                    print('ARIMA{}x{}4 - AIC:{}'.format(param, param_seasonal, results.aic))
                except:
                    continue
    else:
        
        mod = sm.tsa.statespace.SARIMAX(y,
                                        order=(1, 1, 0),
                                        seasonal_order=(1, 1, 1, S),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        
        results = mod.fit()
    
#    print(results.summary().tables[1])    
#    results.plot_diagnostics(figsize=(18, 8))
#    plt.show()
#    #
    pred = results.get_prediction(start=pd.to_datetime('2020-09-02'), dynamic=False)
#    pred_ci = pred.conf_int()
#    ax = y['2020':].plot(label='observed')
#    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
#    ax.fill_between(pred_ci.index,
#                    pred_ci.iloc[:, 0],
#                    pred_ci.iloc[:, 1], color='k', alpha=.2)
#    ax.set_xlabel('Date')
#    ax.set_ylabel('Retail_sold')
#    plt.legend()
#    plt.show()
    ##
    y_forecasted = pred.predicted_mean
#    y_truth = y['2020-06-01':]
#    mse = ((y_forecasted - y_truth) ** 2).mean()
#    print('The Mean Squared Error is {}'.format(round(mse, 2)))
#    print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))
    
    
    pred_uc = results.get_forecast(steps=no_days)
    pred_ci = pred_uc.conf_int()
#    ax = y.plot(label='observed', figsize=(14, 4))
#    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
#    ax.fill_between(pred_ci.index,
#                    pred_ci.iloc[:, 0],
#                    pred_ci.iloc[:, 1], color='k', alpha=.25)
#    ax.set_xlabel('Date')
#    ax.set_ylabel('Sales')
#    plt.legend()
#    plt.show()
#    
    y_forecasted = pred.predicted_mean
#    y_forecasted.head(12)
    
#    y_truth.head(12)
    
#    pred_ci.head(24)
    
    forecast = pred_uc.predicted_mean
#    forecast.head(12)
    return forecast,pred_ci


def main():
#    csv_input="date_reale.csv"
    #csv_input="export.csv"
    #parseArgs()
    #python predict.py --input export.csv --output result.csv --days 20 -S 20 --auto 0 -p 1 -d 1 -q 1 -P 1 -Q 1 -D 1
    [csv_input, out_data, no_days, S, auto, p, d, q, P, D, Q] = parseArgs()      
    df = pd.read_csv(csv_input, parse_dates=[0])
    df=df[df.forecast==0]
    L=len(df.index)
    idx=pd.date_range(df['data'][0], df['data'][L-1])
    
    ### TREBUIE SA COMPLETEZI DATELE LIPSA CU VALORI (CONSTANTE)
    df=df.set_index(['data'])
    upsample_df=df.reindex(idx)
    upsample_df=upsample_df.interpolate(method='time')
    upsample_df['valoare']=upsample_df['valoare'].apply(np.floor)
    
    y=upsample_df.drop(['forecast'],axis=1)
    pred,pred_ci=forecastTimeSeries(y,no_days,S,auto=auto)
    pred=pred.to_frame()
    pred['valoare'] = pred['predicted_mean']
    pred=pred.drop('predicted_mean',axis=1)
    result=pd.concat([df,pred])
    result=pd.concat([result,pred_ci],axis=1,sort=False)
    result['forecast']=result['forecast'].fillna(1)
    result.to_csv(out_data,index=True, quoting=csv.QUOTE_ALL)

    if (1):
        ax = y.plot(label='observed', figsize=(14, 4))
        pred.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Data')
        ax.set_ylabel('Nr. cazuri')
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    main()
    
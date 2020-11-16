# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 01:49:01 2020

@author: Stefan
"""

import json
import pandas as pd
import csv


with open('date_reale.json') as json_file:
    data = json.load(json_file)
print(data)

v= [v for v in data['historicalData'].items() ]

a={'data':[],'valoare':[],'forecast':[]}
n=0
for v in data['historicalData'].items():
    a['data'].append(str(v[0]))
    a['valoare'].append(str(v[1]['numberInfected']))
    a['forecast'].append("0")


df = pd.DataFrame.from_dict(a)
df = df.iloc[::-1]

#y=df.set_index(['data'])
csv_file = "date_reale.csv"
df.to_csv(csv_file,index=False, quoting=csv.QUOTE_ALL)


#try:
#    with open(csv_file, 'w') as csvfile:
#        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#        writer.writeheader()
#        for data in dict_data:
#            writer.writerow(data)
#except IOError:
#    print("I/O error")

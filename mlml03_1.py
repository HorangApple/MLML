import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

file_path = './train.csv'

dfl = pd.read_csv(file_path).drop('gl',axis=1)

sorted_dfl = dfl.sort_values(by='patientid').reset_index()

print(sorted_dfl)
sorted_dfl['di'].fillna(0,inplace=True)
sorted_dfl['chf'].fillna(0,inplace=True)

sorted_dfl['bps'] = sorted_dfl.groupby('patientid')['bps'].transform(lambda x: x.fillna(x.mean()))
sorted_dfl['bpd'] = sorted_dfl.groupby('patientid')['bpd'].transform(lambda x: x.fillna(x.mean()))
sorted_dfl['spo2'] = sorted_dfl.groupby('patientid')['spo2'].transform(lambda x: x.fillna(x.mean()))
sorted_dfl['hr'] = sorted_dfl.groupby('patientid')['hr'].transform(lambda x: x.fillna(x.mean()))

print(sorted_dfl.head().isnull().sum(axis=0))
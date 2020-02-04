import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

file_path = './train2.csv'

dfl = pd.read_csv(file_path).drop('gl',axis=1)

sorted_dfl = dfl.sort_values(by='patientid').reset_index()
grouped = dfl.groupby('patientid')
print(grouped.mean().loc[2.550000e+16,'bps'])
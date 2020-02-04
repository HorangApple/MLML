import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

file_path = './train.csv'
test_file_path = './test.csv'

dfl = pd.read_csv(file_path).drop('gl',axis=1)

sorted_dfl = dfl.sort_values(by='patientid').reset_index()

## NaN 을 전체 평균값으로 채움 시작
mean_bps = sorted_dfl['bps'].mean(axis=0).round(1)
mean_bpd = sorted_dfl['bpd'].mean(axis=0).round(1)
mean_spo2 = sorted_dfl['spo2'].mean(axis=0).round(1)
mean_hr = sorted_dfl['hr'].mean(axis=0).round(1)

# todo: 전체 평균->유저별 평균으로 고쳐야함
sorted_dfl['bps'].fillna(mean_bps, inplace=True)
sorted_dfl['bpd'].fillna(mean_bpd, inplace=True)
sorted_dfl['spo2'].fillna(mean_spo2, inplace=True)
sorted_dfl['hr'].fillna(mean_hr, inplace=True)

## di 또는 chf 를 측정하지 않은 환자를 위한 채움
sorted_dfl['di'].fillna(0,inplace=True)
sorted_dfl['chf'].fillna(0,inplace=True)

## 채움 끝

# Yes == 1, No == 0
sorted_dfl['alert'] = sorted_dfl['alert'].apply(lambda x: 1 if x == 'Yes' else 0)

# F == 1, M == 0
sorted_dfl['gender'] = sorted_dfl['gender'].apply(lambda x: 1 if x == 'F' else 0)

x = sorted_dfl[['bps','gender','bpd','hr','age','di','copd','chf','ht','afib']]
y = sorted_dfl[['alert']]
# sorted_dfl['age'] = sorted_dfl['age'].astype('int')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train.values.ravel())

y_pred = forest.predict_proba(x_test)

print(type(y_pred))
print(type(y_test.to_numpy()))
print(metrics.roc_auc_score(y_test.to_numpy(),y_pred))

print('test 시작')
test_dfl = pd.read_csv(test_file_path).drop('gl',axis=1)

sorted_test_dfl = test_dfl.sort_values(by='patientid').reset_index()

## NaN 을 전체 평균값으로 채움 시작
mean_bps = sorted_test_dfl['bps'].mean(axis=0).round(1)
mean_bpd = sorted_test_dfl['bpd'].mean(axis=0).round(1)
mean_spo2 = sorted_test_dfl['spo2'].mean(axis=0).round(1)
mean_hr = sorted_test_dfl['hr'].mean(axis=0).round(1)

# todo: 전체 평균->유저별 평균으로 고쳐야함
sorted_test_dfl['bps'].fillna(mean_bps, inplace=True)
sorted_test_dfl['bpd'].fillna(mean_bpd, inplace=True)
sorted_test_dfl['spo2'].fillna(mean_spo2, inplace=True)
sorted_test_dfl['hr'].fillna(mean_hr, inplace=True)

## di 또는 chf 를 측정하지 않은 환자를 위한 채움
sorted_test_dfl['di'].fillna(0,inplace=True)
sorted_test_dfl['chf'].fillna(0,inplace=True)

## 채움 끝

# F == 1, M == 0
sorted_test_dfl['gender'] = sorted_test_dfl['gender'].apply(lambda x: 1 if x == 'F' else 0)

x = sorted_test_dfl[['bps','gender','bpd','hr','age','di','copd','chf','ht','afib']]

y_pred = forest.predict(x_test)
result = pd.DataFrame(data=y_pred[:,0])
print(y_pred)
# result.to_csv('./result.csv')
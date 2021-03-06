import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

file_path = './train.csv'
test_file_path = './test.csv'

def run(path):
    dfl = pd.read_csv(path).drop('gl',axis=1)

    sorted_dfl = dfl.sort_values(by='patientid').reset_index()

    ## NaN 을 id 별 평균값으로 채움 시작
    sorted_dfl['bps'] = sorted_dfl.groupby('patientid')['bps'].transform(lambda x: x.fillna(x.mean()))
    sorted_dfl['bpd'] = sorted_dfl.groupby('patientid')['bpd'].transform(lambda x: x.fillna(x.mean()))
    sorted_dfl['spo2'] = sorted_dfl.groupby('patientid')['spo2'].transform(lambda x: x.fillna(x.mean()))
    sorted_dfl['hr'] = sorted_dfl.groupby('patientid')['hr'].transform(lambda x: x.fillna(x.mean()))

    # sorted_dfl['bps']=sorted_dfl['bps'].interpolate(method='nearest')
    # sorted_dfl['bpd']=sorted_dfl['bpd'].interpolate(method='nearest')
    # sorted_dfl['spo2']=sorted_dfl['spo2'].interpolate(method='nearest')
    # sorted_dfl['hr']=sorted_dfl['hr'].interpolate(method='nearest')

    sorted_dfl['w'] = sorted_dfl['w']-sorted_dfl['pw']
    sorted_dfl['w'] = pd.cut(sorted_dfl['w'],bins=[-9999,-1,1,9999],labels=[0,1,2])
    sorted_dfl['bps'] = pd.cut(sorted_dfl['bps'], bins=[0,85,180,9999],labels=[0,1,2])
    sorted_dfl['bpd'] = pd.cut(sorted_dfl['bpd'], bins=[0,40,110,9999],labels=[0,1,2])

    # spo2 없는 사람 처리
    sorted_dfl['spo2'].fillna(-1,inplace=True)
    sorted_dfl['spo2'] = pd.cut(sorted_dfl['spo2'], bins=[-2,0,88,9999],labels=[0,1,2])

    sorted_dfl.to_csv('./result1.csv')

    ## di 또는 chf 를 측정하지 않은 환자를 위한 채움
    sorted_dfl['di'].fillna(2,inplace=True)
    sorted_dfl['chf'].fillna(2,inplace=True)

    # gender -> F == 1, M == 0
    sorted_dfl['gender'] = sorted_dfl['gender'].apply(lambda x: 1 if x == 'F' else 0)

    return sorted_dfl

sorted_dfl = run(file_path)
# alert -> Yes == 1, No == 0
sorted_dfl['alert'] = sorted_dfl['alert'].apply(lambda x: 1 if x == 'Yes' else 0)
sorted_test_dfl = run(test_file_path)

x = sorted_dfl[['bps','gender','bpd','spo2','hr','age','di','copd','chf','ht','afib','w']]
y = sorted_dfl[['alert']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

forest = RandomForestClassifier(n_estimators=500,random_state=0)

forest.fit(x_train, y_train.values.ravel())

y_pred = forest.predict(x_test)

print('accuracy_score :',metrics.accuracy_score(y_test,y_pred))
print('roc_auc :',metrics.roc_auc_score(y_test.to_numpy(),y_pred))

'''
y_test = sorted_test_dfl[['bps','gender','bpd','spo2','hr','age','di','copd','chf','ht','afib']]

y_index = sorted_test_dfl[['index']]

y_pred = forest.predict(y_test)
result = pd.DataFrame({'alert':y_pred})
result['index'] = y_index
result = result.sort_values(by='index')
result = result.drop(columns=['index'])
result.to_csv('./result.csv', index=False)
'''
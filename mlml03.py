import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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

    ## di 또는 chf 를 측정하지 않은 환자를 위한 채움
    sorted_dfl['di'].fillna(0,inplace=True)
    sorted_dfl['chf'].fillna(0,inplace=True)

    # spo2 없는 사람이 꽤 됨
    sorted_dfl['spo2'].fillna(0,inplace=True)

    # alert -> Yes == 1, No == 0
    sorted_dfl['alert'] = sorted_dfl['alert'].apply(lambda x: 1 if x == 'Yes' else 0)

    # gender -> F == 1, M == 0
    sorted_dfl['gender'] = sorted_dfl['gender'].apply(lambda x: 1 if x == 'F' else 0)
    
    # sorted_dfl.to_csv('./result.csv')

    x = sorted_dfl[['bps','gender','bpd','spo2','hr','age','di','copd','chf','ht','afib']]
    y = sorted_dfl[['alert']]


    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

    forest = RandomForestClassifier(n_estimators=100, n_jobs=4,random_state=0)
    
    forest.fit(x_train, y_train.values.ravel())

    y_pred = forest.predict(x_test)
    print('accuracy_score :',metrics.accuracy_score(y_test,y_pred))
    print('roc_auc :',metrics.roc_auc_score(y_test.to_numpy(),y_pred))

run(file_path)
import pandas as pd
from sklearn.model_selection import train_test_split
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

    sorted_dfl['w'] = sorted_dfl['w']-sorted_dfl['pw']

    sorted_dfl['bps'] = pd.cut(sorted_dfl['bps'], bins=[0,85,180,9999],labels=[0,1,2])
    sorted_dfl['bpd'] = pd.cut(sorted_dfl['bpd'], bins=[0,40,110,9999],labels=[0,1,2])

    # spo2 없는 사람 처리
    sorted_dfl['spo2'].fillna(-1,inplace=True)
    sorted_dfl['spo2'] = pd.cut(sorted_dfl['spo2'], bins=[-2,0,88,9999],labels=[0,1,2])

    # sorted_dfl.to_csv('./result.csv')

    ## di 또는 chf 를 측정하지 않은 환자를 위한 채움
    sorted_dfl['di'].fillna(2,inplace=True)
    sorted_dfl['chf'].fillna(2,inplace=True)

    # gender -> F == 1, M == 0
    sorted_dfl['gender'] = sorted_dfl['gender'].apply(lambda x: 1 if x == 'F' else 0)

    return sorted_dfl

train_file = run(file_path)
test_file = run(test_file_path)

train_file = train_file[['bps','gender','bpd','spo2','hr','age','di','copd','chf','ht','afib']]
test_file = test_file[['bps','gender','bpd','spo2','hr','age','di','copd','chf','ht','afib']]

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

cosined = cosine_similarity(test_file.values, train_file.values)

print(cosined)
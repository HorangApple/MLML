import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,TimeDistributed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer

import string
d = dict(zip(string.ascii_lowercase, range(0,26)))

def change (inp):
    changed = []
    for i in inp:
        changed.append(d[i])
    return changed

def result (n1,n2,n3):
    ln_n1 = len(n1)
    ln_n2 = len(n2)
    ln_n3 = len(n3)
    target = [ln_n1]
    if ln_n3!=0:
        target.append(ln_n1+ln_n2)
    result = [0]*(ln_n1+ln_n2+ln_n3)
    result[0] = 1
    for i in target:
        result[i] = 1
    return result

file_path = './train.csv'
dfl = pd.read_csv(file_path)
dfl = dfl.fillna('')
dfl = dfl.applymap(change)
dfl['result'] = dfl.apply(lambda x: result(x['n1'],x['n2'],x['n3']), axis=1)

# dfl.to_csv('./result.csv',index=None)
x = dfl[['compound']]
y = dfl[['result']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

model = Sequential()
model.add(TimeDistributed(Embedding(26, 13)))
model.add(TimeDistributed(Bidirectional(LSTM(13, return_sequences=True, input_shape=(1050,1)))))
model.add(TimeDistributed(Dense(2, activation='softmax')))


model.build(input_shape = (None, None, 26))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model.fit(x_train, y_train,
          batch_size=32,
          epochs=4,
          validation_data=[x_test, y_test])
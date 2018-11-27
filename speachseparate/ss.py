import os

import numpy as np
import time
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop
from python_speech_features import mfcc
from scipy.io import wavfile
from scipy.signal import resample
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

path = '/Users/ben/Documents/datasets/data_thchs30/train'


# GET FILE
def get_wav_files(path=path):
    wav_files = []
    file_names = []
    for (dirpath, dirnames,filenames) in os.walk(path):
        for name in filenames:
            if name.endswith('.wav') or name.endswith('.WAV'):
                name_path = os.sep.join([dirpath,name])
                if os.stat(name_path).st_size < 240000:
                    continue
                wav_files.append(name_path)
                file_names.append(name)
    return wav_files, file_names

wav_files,file_names = get_wav_files(path)


train_y = []
train_x = []
begin_time = time.time()
i=0
for wav_file, file_name in tqdm(zip(wav_files,file_names)):
    label = file_name.split('_')[0]
    (fs,data) = wavfile.read(wav_file)
    mfcc_feat = mfcc(resample(data,len(data)//2),fs//2)
    mfcc_feat_div = np.concatenate((mfcc_feat[[0]],mfcc_feat[:-1]))
    mfcc_feat_div_div = mfcc_feat_div -  np.concatenate((mfcc_feat_div[[0]],mfcc_feat_div[:-1]))
    finalfeature = np.concatenate((mfcc_feat,mfcc_feat_div,mfcc_feat_div_div),axis=1)
    train_x.append(finalfeature)
    train_y.append(label[1:])



yy = LabelBinarizer().fit_transform(train_y)
train_x = [ np.concatenate((i,np.zeros((1561-i.shape[0],39)))) for i in train_x]
train_x = np.asarray(train_x)
train_y = np.asarray(yy)
print(train_x.shape,train_y.shape)


X_train,X_test,y_train,y_test = train_test_split(train_x,train_y,test_size=.3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


print(y_train.shape)
model = Sequential()
model.add(Conv1D(32,4,input_shape=((300,39))))
model.add(MaxPool1D(4))
model.add(Conv1D(64,4))
model.add(MaxPool1D(4))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(train_y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=[categorical_accuracy])

early_stopping = EarlyStopping(monitor='val_loss',patience=10)
model.fit(X_train[:,200:500,:],y_train,validation_data=[X_test[:,200:500,:],y_test],batch_size=16,epochs=100,callbacks=[early_stopping])

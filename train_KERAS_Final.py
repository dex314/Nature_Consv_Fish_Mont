# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:32:51 2017

@author: SchillW


This is mostly a take on Fishy-Keras originally authored by ZFTurbo:
    https://www.kaggle.com/zfturbo/the-nature-conservancy-fisheries-monitoring/fishy-keras-lb-1-25267/run/445238
"""

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.constraints import maxnorm

##to resolve an issue
import tensorflow as tf
tf.python.control_flow_ops = tf


#%%
'''CLIPPING and BLENDING'''
def blend_csv(csv_paths,path):
    if len(csv_paths) < 2:
        print("Blending takes two or more csv files!")
        return
    
    # Read the first file
    df_blend = pd.read_csv(csv_paths[0], index_col=0)
    
    # Loop over all files and add them
    for csv_file in csv_paths[1:]:
        df = pd.read_csv(csv_file, index_col=0)
        df_blend = df_blend.add(df)
        
    # Divide by the number of files
    df_blend = df_blend.div(len(csv_paths))

    # Save the blend file
    df_blend.to_csv(path+'blendSubmission.csv')
    print(df_blend.head(10))


def clip_csv(csv_file, clip, classes, path):
    # Read the submission file
    df = pd.read_csv(csv_file, index_col=0)
    # Clip the values
    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)
    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)
    #Sort
#    df = df.sort()
    # Save the new clipped values
    df.to_csv(path+'clipSubmission.csv')
    print(df.head(10))

#%%
def acanny(img, sig=0.33):
    eF = []
    v = np.median(img)
    lw = int(max(0, (1.0-sig)*v))
    up = int(min(255, (1.0-sig)*v))    
    eF = np.zeros(np.shape(img))
    for i in range(3):
        eL = cv2.Canny(img[:,:,i], lw, up)
        eF[:,:,i] = eL
    
    return eF

#%%
def get_im_cv2(path, mx, nx):
    img = cv2.imread(path)
    resized = cv2.resize(img, (mx, nx), cv2.INTER_LINEAR)

    return resized

#%%
def load_train(mx, nx, path):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        
        path0=path+'\\train\\train\\'
        
        pathf = os.path.join(path0,fld,'*.jpg')
        files = glob.glob(pathf)
        for fl in files:
            for da in range(1):
                flbase = os.path.basename(fl)   
                img = get_im_cv2(fl, mx, nx)      
                X_train.append(img)
                X_train_id.append(flbase)
                y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test(mx, nx, path):
    pathf = os.path.join(path,'test_stg1','test_stg1','*.jpg')
    files = sorted(glob.glob(pathf))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, mx, nx)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

def load_test2(mx, nx, path):
    path1 = os.path.join(path,'test_stg1','test_stg1','*.jpg')
    path2 = os.path.join(path,'test_stg2','*.jpg')
    
    files1 = sorted(glob.glob(path1))
    files2 = sorted(glob.glob(path2))

    X_test = []
    X_test_id = []
    for fl in files1:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, mx, nx)
        X_test.append(img)
        X_test_id.append(flbase)
    
    for fl in files2:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, mx, nx)
        X_test.append(img)
        X_test_id.append('test_stg2/'+flbase)

    return X_test, X_test_id


#%%
def create_submission(predictions, test_id, info, path):
    result1 = pd.DataFrame(predictions) #, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    tdf = pd.DataFrame(test_id)
    result1 = pd.concat([tdf, result1], axis=1, join='outer')
    result1.columns=['image','ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
#    result1.loc['image', :] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submissionK_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    pathf = path+'\\submission2\\'
    result1.to_csv(pathf+sub_file, index=False)

#%%
def read_and_normalize_train_data(mx, nx, path):
    train_data, train_target, train_id = load_train(mx, nx, path)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    print( np.shape(train_data))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = ((train_data/255.0) - 0.5) * 2.0 
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data(mx, nx, path):
    start_time = time.time()
    test_data, test_id = load_test2(mx, nx, path) ## !!!!! 13158 FILES!!!!!

    test_data = np.array(test_data, dtype=np.uint8)       
    test_data = test_data.astype('float32')
    test_data = ((test_data/255.0) - 0.5) * 2.0

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


#%%
def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def create_model(mx,nx):
    
    filts = [4,8,16]
    filts = [8,16,32]
    initN = 'he_uniform' #'glorot_uniform'
    
    model = Sequential()
      
#    model.add(ZeroPadding2D((1, 1), input_shape=(nx,mx,3))) 
#    model.add(Convolution2D(filts[0], 7, 7, activation='relu', init=initN)) 
#    model.add(Convolution2D(filts[0], 5, 5, activation='relu', init=initN)) 
#    model.add(Convolution2D(filts[0], 3, 3, activation='relu', init=initN)) 
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#    model.add(Dropout(0.2)) ##this was junk

#    model.add(ZeroPadding2D((1, 1), input_shape=(nx,mx,3))) 
#    model.add(Convolution2D(filts[0], 3, 3, activation='relu', init=initN)) 
#    model.add(Convolution2D(filts[0], 3, 3, activation='relu', init=initN)) 
#    model.add(Dropout(0.2)) 
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
      
    model.add(ZeroPadding2D((1, 1), input_shape=(nx,mx,3)))  
    model.add(Convolution2D(filts[1], 3, 3, activation='relu', init=initN)) 
    model.add(Convolution2D(filts[1], 3, 3, activation='relu', init=initN)) 
    model.add(Dropout(0.2)) 
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1))) 
    model.add(Convolution2D(filts[2], 3, 3, activation='relu', init=initN)) 
    model.add(Convolution2D(filts[2], 3, 3, activation='relu', init=initN)) 
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2)) 
      
    model.add(Flatten())
    model.add(Dense(96, activation='relu', init=initN))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='relu',init=initN))
    model.add(Dropout(0.2)) 
    model.add(Dense(8, activation='softmax'))

    opt = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    
    return model

#%%
def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

def run_cross_validation_create_models(nfolds, mx, nx, 
                                       train_data, train_target, train_id, path):
    # input image dimensions
    batch_size = 24
    nb_epoch = 15
    random_state = 51

#    train_data, train_target, train_id = read_and_normalize_train_data(mx, nx)

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    preds = []
    for train_index, test_index in kf:
        model = create_model(mx,nx)
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)
        preds.append(predictions_valid)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)
    
    rscr = np.around(score, 3)

    info_string = 'loss_' + str(rscr) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    allweights = 0
    return info_string, models, preds, allweights


def run_cross_validation_process_test(info_string, models, mx, nx, path):
    batch_size = 24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data(mx, nx, path)
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string, path)
    return info_string

#%%
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3

#%%    
    mx = 110
    nx = 70

    cdir = os.getcwd()
    print(cdir)
    
    cdir = cdir+'\\NatureConsvData'
    print(cdir)
    
#%%
    train_data, train_target, train_id = read_and_normalize_train_data(mx,nx,cdir)   
    
#%% SEPARATION TO KEEP DATA SIZE AND JUST RUN MODEL ONLY IF NECESSARY
    info_string, models, preds, all_weights = run_cross_validation_create_models(num_folds, mx, nx, 
                                                                                 train_data, train_target, 
                                                                                 train_id, cdir)
    
#%%
    docstr = run_cross_validation_process_test(info_string, models, mx, nx, cdir)
     
   
#%%
    clip = 0.90
    classes = 8
    
    cpath = cdir+'\\submission2\\'
    
    tims = '_2017-04-06-13-18'
    fulln = cpath+'submissionK_'+docstr+tims+'.csv'
    
#    fulln = cpath+'best.csv'
    
    clip_csv(fulln, clip, classes, cpath)
       
    csvlist = []
    for i in glob.glob(cpath+'\\*.csv'):
    #    print(i)
        csvlist.append(i)
    
    blend_csv(csvlist,cpath)
    
    
#%%
'''
OTHER NOTES:
    - had a couple of previous close submission to this one in the bank that I blended with this one.
    - starting from the top, I did all at 110x70
        - then pairs of: 4,3,3 at .2 drop then maxpool22
                         8,3,3 at .2 drop then maxpool22
                         16,3,3 maxpool22, then .2 drop
                         dense 96, .4drop
                         dense 24, .2drop
                         dense 8  
    
    
    
##end
'''


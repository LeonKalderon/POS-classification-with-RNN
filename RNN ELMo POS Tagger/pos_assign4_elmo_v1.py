# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Georgia Sarri, Leon Kalderon, George Vafeidis
"""


import os
import pyconll
import pyconll.util
import operator


#from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

from keras import backend as K # Importing Keras backend (by default it is Tensorflow)
from keras.layers import Input, Dense # Layers to be used for building our model
from keras.models import Model # The class used to create a model
from keras.optimizers import Adam
from keras.layers import  Bidirectional, Dropout, TimeDistributed, BatchNormalization, Masking, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers import LeakyReLU
from tensorflow import set_random_seed # Used for reproducible experiments
from tensorflow import keras
from keras.models import load_model

#!pip install tensorflow_hub
import tensorflow_hub as hub

#from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import add, concatenate
from keras.utils import np_utils # Utilities to manipulate numpy arrays
import tensorflow as tf


import gc
import matplotlib.pyplot as plt
import numpy as np
import itertools

import pickle

import pandas as pd
import seaborn as sn

from collections import defaultdict

import matplotlib

# ------------------------------------
# files and Directories for each user
#-------------------------------------

users = ['GS','LK','GVHome','GVWork']

active_user = users[2]

if active_user == 'GS':
    MODELS_PATH = r'C:\Users\Georgia\Desktop\Assignment3'
    FASSTEX_FILE = r'C:\Users\Georgia\Desktop\Assignment3\cc.en.300.vec\cc.en.300.vec'
    FASTEX_OUTPUT = r'C:\Users\Georgia\Desktop\Assignment3\fasttext.npy'
    UD_ENGLISH_TRAIN = r'C:\Users\Georgia\Desktop\Assignment3\data\corrected\en_cesl-ud-train.conllu'
    UD_ENGLISH_VALIDATION = r'C:\Users\Georgia\Desktop\Assignment3\data\corrected\en_cesl-ud-dev.conllu'
    UD_ENGLISH_TEST = r'C:\Users\Georgia\Desktop\Assignment3\data\corrected\en_cesl-ud-test.conllu'

elif active_user == 'LK':
    MODELS_PATH = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3'
    FASSTEX_FILE = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\cc.en.300.vec\cc.en.300.vec'
    FASTEX_OUTPUT = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\fasttext.npy'
    UD_ENGLISH_TRAIN = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\data\corrected\en_cesl-ud-train.conllu'
    UD_ENGLISH_VALIDATION = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\data\corrected\en_cesl-ud-dev.conllu'
    UD_ENGLISH_TEST = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\data\corrected\en_cesl-ud-test.conllu'

elif active_user == 'GVHome':
    MODELS_PATH = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 4'
    FASSTEX_FILE = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\cc.en.300.vec'
    FASTEX_OUTPUT = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\fasttext.npy'
    UD_ENGLISH_TRAIN = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\data\corrected\en_cesl-ud-train.conllu'
    UD_ENGLISH_VALIDATION = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\data\corrected\en_cesl-ud-dev.conllu'
    UD_ENGLISH_TEST = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\data\corrected\en_cesl-ud-test.conllu'

elif active_user == 'GVWork':
    MODELS_PATH = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 4'
    FASSTEX_FILE = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\cc.en.300.vec'
    FASTEX_OUTPUT = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\fasttext.npy'
    UD_ENGLISH_TRAIN = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\data\corrected\en_cesl-ud-train.conllu'
    UD_ENGLISH_VALIDATION = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\data\corrected\en_cesl-ud-dev.conllu'
    UD_ENGLISH_TEST = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\data\corrected\en_cesl-ud-test.conllu'

else:
    MODELS_PATH = ''
    FASSTEX_FILE = ''
    FASTEX_OUTPUT = ''
    UD_ENGLISH_TRAIN = ''
    UD_ENGLISH_VALIDATION = ''
    UD_ENGLISH_TEST = ''


model_file_name = os.path.join(MODELS_PATH, 'temp_elmo_model.h1')
log_file_name = os.path.join(MODELS_PATH, 'temp_elmo.log')

##############################################################################
# Metrics: given in class
    
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss    
    return loss    

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy

def recall(y_true, y_pred):
    
    """
    Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    
    """
    Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    Source
    ------
    https://github.com/fchollet/keras/issues/5400#issuecomment-314747992
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    
    """Calculate the F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))


def accuracy(y_true, y_pred):
    return categorical_accuracy(y_true,y_pred)

def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())

def my_class_accuracy(y_true, y_pred):
    '''
    This metric ignores PAD to calculate Accuracy
    '''
    to_ignore = 0
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)
 
    ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
    matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
    accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
    return accuracy
   
    
##############################################################################

def plot_history(hs, epochs, metric):
    plt.clf()
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['font.size'] = 16
    for label in hs:
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric))
        plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric))
    epochs = len(hs['RNN'].history['loss'])
    x_ticks = np.arange(0, epochs + 1, 5)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.show()

def data_extract(dataset):
  x_set=[]
  y_set=[]   
  for sentence in dataset:
     for j in range(len(sentence)):
        x_set.append(sentence[j].form)
        y_set.append(sentence[j].upos)
  return x_set, y_set


url = r'https://tfhub.dev/google/elmo/2'
elmo = hub.Module(url)

def my_one_hot_encoding(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

#Reference:
#https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede
def sentence_extract(dataset):
    sentences = []
    for sentence in dataset:
        sent = []
        for j in range(len(sentence)):
            sent.append((sentence[j].form, sentence[j].upos))
            #x_set.append(sentence[j].form)
            #y_set.append(sentence[j].upos)
        sentences.append(sent)
    return sentences


train = pyconll.load_from_file(UD_ENGLISH_TRAIN)
validation =  pyconll.load_from_file(UD_ENGLISH_VALIDATION)
test =  pyconll.load_from_file(UD_ENGLISH_TEST)

x_train_words, y_train_tags = data_extract(train)
x_valid_words, y_valid_tags = data_extract(validation)
x_test_words, y_test_tags = data_extract(test)

train_sentences = sentence_extract(train)
validation_sentences = sentence_extract(validation)
test_sentences = sentence_extract(test)

print("Train Sentence Lenght: {}".format(len(train)))

max_sen_length = max(len(sen) for sen in train_sentences)
print('Largest train sentence has {} words'.format(max_sen_length))

plt.hist([len(sent) for sent in train_sentences], bins=50)
plt.xlabel('Snetence length')
plt.ylabel('Occurances')
plt.legend('Histogram of Sentence length')
plt.show
#Most of the sentences have lenght less than 60, so we can use that as a max_len

SENT_LENGTH = 60

# TRAIN
X_train = [[w[0] for w in s] for s in train_sentences]
X_train_padded = []

for sent in X_train:
    new_sent = []
    for i in range(SENT_LENGTH):
        try:
            new_sent.append(sent[i])
        except:
            new_sent.append("<PAD>")
    X_train_padded.append(new_sent)
    
X_train[0]
X_train_padded[0]

pos2index = {t : i+1 for i,t in enumerate(set(y_train_tags))}
pos2index["<PAD>"] = 0
y_train = [[pos2index[w[1]] for w in s] for s in train_sentences]
y_train_padded = pad_sequences(maxlen=SENT_LENGTH, sequences=y_train, value=pos2index["<PAD>"], padding='post')
y_train[0]
y_train_padded[0]

# VALIDATION
X_valid = [[w[0] for w in s] for s in validation_sentences]
X_valid_padded = []

for sent in X_valid:
    new_sent = []
    for i in range(SENT_LENGTH):
        try:
            new_sent.append(sent[i])
        except:
            new_sent.append("<PAD>")
    X_valid_padded.append(new_sent)
    
X_valid[0]
X_valid_padded[0]

pos2index_valid = {t: i+1 for i,t in enumerate(set(y_valid_tags))}
pos2index_valid["<PAD>"] = 0
y_valid = [[pos2index[w[1]] for w in s] for s in validation_sentences]
y_valid_padded = pad_sequences(maxlen=SENT_LENGTH, sequences=y_valid, value=pos2index["<PAD>"], padding='post')
y_valid[0]
y_valid_padded[0]

# TEST
X_test = [[w[0] for w in s] for s in test_sentences]
X_test_padded = []

for sent in X_test:
    new_sent = []
    for i in range(SENT_LENGTH):
        try:
            new_sent.append(sent[i])
        except:
            new_sent.append("<PAD>")
    X_test_padded.append(new_sent)
    
X_test[0]
X_test_padded[0]

pos2index_test = {t: i+1 for i,t in enumerate(set(y_test_tags))}
pos2index_test["<PAD>"] = 0
y_test = [[pos2index[w[1]] for w in s] for s in test_sentences]
y_test_padded = pad_sequences(maxlen=SENT_LENGTH, sequences=y_test, value=pos2index_test["<PAD>"], padding='post')
y_test[0]
y_test_padded[0]

#ELMO
batch_size = 16

def ElmoEmbedding(x):
    return elmo(inputs={"tokens": tf.squeeze(tf.cast(x, 'string'))
                              ,"sequence_len": tf.constant(batch_size*[SENT_LENGTH])
                              }
                    ,signature="tokens"
                    ,as_dict=True)["elmo"]

def make_model_ELMO (EMBEDDING_DIM, SENT_LENGTH, classes_no,
                hidden_activation='relu',output_activation='softmax',dropout_rate=0.5):
    GRU_SIZE = 120
    DENSE = 160
    N_CLASSES = classes_no
    my_mask=np.zeros(N_CLASSES)
    my_mask[-1] = 1
    # Define the input layer.
    input_layer = Input(shape = (SENT_LENGTH,), dtype='string', name = 'Input')
    embed_layer = Lambda(ElmoEmbedding, output_shape=(SENT_LENGTH, EMBEDDING_DIM))(input_layer)
    # LSTM
#    norm_embed = BatchNormalization()(embed_layer)
    LSTM1 = Bidirectional(LSTM(180, return_sequences=True, recurrent_dropout = 0.25,dropout=dropout_rate))(embed_layer)
    LSTM2 = Bidirectional(LSTM(180, return_sequences=True, recurrent_dropout = 0.25,dropout=dropout_rate))(LSTM1)
#    gru_l1 = Bidirectional(GRU(GRU_SIZE, return_sequences=True, recurrent_dropout = 0.25,dropout = dropout_rate))(embed_layer)
#    gru_l2 = Bidirectional(GRU(GRU_SIZE, return_sequences=True, recurrent_dropout = 0.25,dropout = dropout_rate))(gru_l1)
#    x = concatenate([gru_l1, gru_l2]) 
#    x = concatenate([LSTM1, LSTM2])
#    LSTM2_norm = BatchNormalization()(LSTM2)
    hidden_layer1 = Dense(units=DENSE,activation='tanh')(LSTM2)
#    hidden_layer1 = LeakyReLU()(hidden_layer1)
    hidden_layer1_dropout=Dropout(dropout_rate)(hidden_layer1)
    # Define the output layer.    
    output = TimeDistributed(Dense(units=N_CLASSES,kernel_initializer='uniform',activation=output_activation,name='Output'))(hidden_layer1_dropout)
    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    weights = np.ones((N_CLASSES,))
    weights[0] = 0.05
    weights[1] = 1 # ADJ
    weights[8] = 1 # NOUN
    weights[16] = 1 # VERB
    
#   'categorical_crossentropy' ---- masked_categorical_crossentropy --- weighted_categorical_crossentropy(weights)
    model.compile(loss=weighted_categorical_crossentropy(weights),
                  optimizer=Adam(lr=0.0006),
#                  metrics=['accuracy'])
                  metrics=[precision, recall, f1, accuracy,my_class_accuracy])
    
    print('Finished training.')
    print('------------------')
    model.summary() # Print a description of the model.
    return model


#since we have 26 as the batch size, feeding the network 
#must be in chunks that are all multiples of 32:
X_tr, X_val = X_train_padded[:257*batch_size], X_valid_padded[-31*batch_size:]
y_tr, y_val = y_train_padded[:257*batch_size], y_valid_padded[-31*batch_size:]
#X_test = X_test_padded[:31*batch_size]
#y_test= y_test_padded[-31*batch_size:]
#y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
#y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)


#X_tr, X_val = X_train_padded, X_valid_padded
#y_tr, y_val = y_train_padded, y_valid_padded
X_test, y_test = X_test_padded,  y_test_padded
#y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
#y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
#y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

batch_size = 16


# use: with tf.device('/cpu:0') to run with cpu
with tf.device('/GPU:1'):
    RNN_model_adam_elmo = make_model_ELMO(
          hidden_activation='relu',   
          output_activation='softmax',
          dropout_rate=0.56,
          SENT_LENGTH = SENT_LENGTH,
          EMBEDDING_DIM = 1024,
          classes_no = 18)

np.random.seed(1402) # Define the seed for numpy to have reproducible experiments.
set_random_seed(1981) # Define the seed for Tensorflow to have reproducible experiments.
#with tf.Session() as sess:
#     sess.run(tf.local_variables_initializer())
#K.set_session(tf.Session())
#K.get_session().run(tf.local_variables_initializer())

lr_reducer = keras.callbacks.ReduceLROnPlateau(factor = 0.25, patience = 2, min_lr = 1e-7, verbose = 1)
check_pointer = keras.callbacks.ModelCheckpoint(model_file_name, verbose = 1, save_best_only = True)
early_stopper = keras.callbacks.EarlyStopping(patience = 5) # Change 4 to 8 in the final run    
#csv_logger = keras.callbacks.CSVLogger(log_file_name)

keras.backend.get_session().run(tf.global_variables_initializer())
# use: with tf.device('/cpu:0') to run with cpu
with tf.device('/GPU:1'):
    RNN_hs_model_elmo = RNN_model_adam_elmo.fit(
        x = np.array(X_tr),
        y = my_one_hot_encoding(y_tr,18),
        validation_data = (np.array(X_val), my_one_hot_encoding(y_val,18)),
        epochs=20,
        shuffle = True,
        verbose=1,
        batch_size=batch_size,
        callbacks = [early_stopper, lr_reducer,check_pointer]
        )  

    
def one_hot_to_class(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

#*****************************************************
# Load Model if needed for testing instead of training
#*****************************************************

#weights = np.ones((18,))
#weights[0] = 0.05 # PADs Minimum value for all instances to be TP
#RNN_model_adam_elmo = load_model(MODELS_PATH + r'\2LSTM1MLPd003_elmo_model.h1', 
#                            custom_objects= {'loss': weighted_categorical_crossentropy(weights),
#                                             'precision':precision,
#                                              'recall':recall,
#                                               'f1':f1,
#                                               'accuracy':accuracy,
#                                               'my_class_accuracy':my_class_accuracy,
#                                               'elmo':elmo,'tf':tf,'batch_size':batch_size,
#                                               'SENT_LENGTH':SENT_LENGTH})


RNN_eval_adam = RNN_model_adam_elmo.evaluate(np.array(X_test)[0:496], my_one_hot_encoding(y_test,18)[0:496], batch_size=batch_size, verbose=1)
print("Train Loss     : {0:.5f}".format(RNN_hs_model_elmo.history['loss'][-1]))
print("Validation Loss: {0:.5f}".format(RNN_hs_model_elmo.history['val_loss'][-1]))
print("Test Loss      : {0:.5f}".format(RNN_eval_adam[0]))
print("---")

print("Train Accuracy     : {0:.5f}".format(RNN_hs_model_elmo.history['accuracy'][-1]))
print("Validation Accuracy: {0:.5f}".format(RNN_hs_model_elmo.history['val_accuracy'][-1]))

print("Test categorical_cross_entropy:{0:.5f}".format(RNN_eval_adam[0]))
print("Test precision      : {0:.5f}".format(RNN_eval_adam[1]))
print("Test recall      : {0:.5f}".format(RNN_eval_adam[2]))
print("Test f1      : {0:.5f}".format(RNN_eval_adam[3]))
print("Test accuracy        : {0:.5f}".format(RNN_eval_adam[4]))
print("Test accuracy w/o PAD: {0:.5f}".format(RNN_eval_adam[5]))

plot_history(hs={'RNN': RNN_hs_model_elmo}, epochs=10, metric='loss')
plot_history(hs={'RNN': RNN_hs_model_elmo}, epochs=10, metric='accuracy')

#sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

# Predict and return one-hot targets to labels
predictions = RNN_model_adam_elmo.predict(np.array(X_test)[0:496],batch_size=batch_size,verbose=1)
y_predict_class = one_hot_to_class(predictions, {i: t for t, i in pos2index.items()})
y_test_class = one_hot_to_class(my_one_hot_encoding(y_test,18), {i: t for t, i in pos2index.items()})


#----- Analysis including PADS ------
#Flattent predict and test for passing below for comparison
y_predict_class_flat = list(itertools.chain(*y_predict_class))
y_test_class_flat = list(itertools.chain(*y_test_class[0:496]))

# Make them label encoders to use them for the rest
# Add also <PAD>
lbls = set(y_train_tags)
lbls.add('<PAD>')
le = preprocessing.LabelEncoder()
le.fit(list(lbls))
y_predict_class_flat_lbl = le.transform(y_predict_class_flat)
y_test_class_flat_lbl = le.transform(y_test_class_flat)

classes=18

# Print Classification Report
report_model = classification_report(y_predict_class_flat_lbl,y_test_class_flat_lbl,
                                     target_names = le.classes_,digits = 4)
print(report_model) 

# Construct Confusion Matrix and print it
cm1 = confusion_matrix(y_test_class_flat_lbl,y_predict_class_flat_lbl)
df_cm = pd.DataFrame(cm1, index = [i for i in np.linspace(0,classes-1,classes,dtype=int)],
                  columns = [i for i in np.linspace(0,classes-1,classes,dtype=int)])
plt.figure(figsize = (10,18))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)


# Strip PADS
y_predict_NoPADs, y_test_NoPADs = [],[]
for i in range(len(y_predict_class_flat)):
    if not ((y_predict_class_flat[i]=='<PAD>') and (y_test_class_flat[i]=='<PAD>')):
        y_predict_NoPADs.append(y_predict_class_flat[i])
        y_test_NoPADs.append(y_test_class_flat[i])
len(y_predict_NoPADs)
len(y_test_NoPADs)
'<PAD>' in y_test_NoPADs
'<PAD>' in y_predict_NoPADs
        
le1 = preprocessing.LabelEncoder()
le1.fit(list(set(y_train_tags)))
y_predict_class_flat_lbl = le1.transform(y_predict_NoPADs)
y_test_class_flat_lbl = le1.transform(y_test_NoPADs)

classes=len(le1.classes_)

# Print Classification Report
report_model = classification_report(y_predict_class_flat_lbl,y_test_class_flat_lbl,
                                     target_names = le1.classes_,digits = 4)
print(report_model) 

# Construct Confusion Matrix and print it
cm1 = confusion_matrix(y_test_class_flat_lbl,y_predict_class_flat_lbl)
df_cm = pd.DataFrame(cm1, index = [i for i in np.linspace(0,classes-1,classes,dtype=int)],
                  columns = [i for i in np.linspace(0,classes-1,classes,dtype=int)])
plt.figure(figsize = (10,18))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',xticklabels=le1.classes_,yticklabels=le1.classes_)






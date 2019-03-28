# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:09:59 2019

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
from keras.layers import  Bidirectional, Dropout, TimeDistributed, BatchNormalization, Masking
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers import LeakyReLU
from keras.layers.merge import add, concatenate
from keras.models import load_model
from tensorflow import set_random_seed # Used for reproducible experiments
from tensorflow import keras

#from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

model_file_name = os.path.join(MODELS_PATH, 'temp_model.h1')
log_file_name = os.path.join(MODELS_PATH, 'temp.log')

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

def clean_up(model):
    K.clear_session()
    del model
    gc.collect()  
    
def make_model (MAX_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,embedding_matrix, classes_no,batch_size,
                hidden_activation='relu',output_activation='softmax',dropout_rate=0.5):
#    GRU_SIZE = 130
    LSTM_SIZE = 150
    DENSE = 100
    N_CLASSES = classes_no
#    my_mask=np.zeros(N_CLASSES)
#    my_mask[-1] = 1
#    masked_categorical_crossentropy = get_loss(my_mask) #ignores pad, bad results not used
    # Define the input layer.
    input_layer = Input(shape = (MAX_SEQUENCE_LENGTH,),name = 'Input')
    masked_input = Masking(mask_value=0)(input_layer)
    embed_layer = Embedding(MAX_WORDS+1, EMBEDDING_DIM, weights=[embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH,mask_zero=False, trainable=False)(masked_input)
    # LSTM
    LSTM1 = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, 
                               recurrent_dropout = 0.15, dropout = dropout_rate,stateful=False))(embed_layer)
    LSTM2 = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True,
                               recurrent_dropout = 0.15, dropout = dropout_rate,stateful=False))(LSTM1)
#    LSTM3 = Bidirectional(LSTM(250, return_sequences=True,recurrent_dropout = 0.2, dropout = dropout_rate))(LSTM2)


    # GRU
#    gru_l1 = Bidirectional(GRU(GRU_SIZE, return_sequences=True, recurrent_dropout = 0.25,dropout = dropout_rate))(norm_embed)
#    gru_l2 = Bidirectional(GRU(GRU_SIZE, return_sequences=True, recurrent_dropout = 0.25,dropout = dropout_rate))(gru_l1)
#    gru_l3 = Bidirectional(GRU(GRU_SIZE, return_sequences=True, recurrent_dropout = 0.25,dropout = dropout_rate))(gru_L2)

    # Dense Layer (MLP)
    hidden_layer1 = Dense(units=DENSE)(LSTM2)
#    hidden_layer1 = Dense(units=DENSE)(gru_l2_drop_out)
    hidden_layer1 = LeakyReLU()(hidden_layer1)
    hidden_layer1_dropout=Dropout(dropout_rate)(hidden_layer1)

    # Define the output layer.    
    output = TimeDistributed(Dense(units=N_CLASSES,kernel_initializer='uniform',
                                   activation=output_activation,name='Output'))(hidden_layer1_dropout)
    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    weights = np.ones((N_CLASSES,))
    weights[0] = 0.05 # PADs Minimum value for all instances to be TP
    weights[1] = 0.5 # ADJ
    weights[8] = 1 # NOUN
    weights[16] = 0.5 # VERB
#   'categorical_crossentropy' ---- masked_categorical_crossentropy --- weighted_categorical_crossentropy(weights)
    model.compile(loss=weighted_categorical_crossentropy(weights),
                  optimizer=Adam(lr=0.008),
                  metrics=[precision, recall, f1, accuracy,my_class_accuracy])
    print('Model Summary')
    print('-------------')
    model.summary() # Print a description of the model.
    return model


#fasttext
######################################################################
idx = 0
vocab = {}

with open(FASSTEX_FILE, 'r', encoding="utf-8", newline='\n',errors='ignore') as f:
    for l in f:
        line = l.rstrip().split(' ')
        if idx == 0:
            vocab_size = int(line[0]) + 1
            dim = int(line[1])
            vecs = np.zeros(vocab_size*dim).reshape(vocab_size,dim)
            vocab["<PAD>"] = 0
            idx = 1
        else:
            vocab[line[0]] = idx
            emb = np.array(line[1:]).astype(np.float)
            if (emb.shape[0] == dim):
                vecs[idx,:] = emb
                idx+=1
            else:
                continue

    pickle.dump(vocab,open("fasttext_voc",'wb'))
    np.save(FASTEX_OUTPUT,vecs)

fasttext_embed = np.load(FASTEX_OUTPUT)
fasttext_word_to_index = pickle.load(open("fasttext_voc", 'rb'))


def sentence_indexed(dataset,word2indexset,tag2indexset):
  """
  Get the pyconll dataset and make a list of the sentences and the 
  TAGS with indexes instead of words.
  """   
  indexed_sentences = []
  indexed_tags = []
  for sentence in dataset:      
     s_int = []
     y_int = []
     for j in range(len(sentence)):
         try:
             s_int.append(word2indexset[sentence[j].form])
         except KeyError:
             s_int.append(word2indexset['<OOV>']) 
         try:
             y_int.append(tag2indexset[sentence[j].upos])
         except KeyError:
             y_int.append(tag2indexset['<OOV>'])              
     indexed_sentences.append(s_int)
     indexed_tags.append(y_int)
  return indexed_sentences, indexed_tags


def data_extract(dataset):
  x_set=[]
  y_set=[]   
  for sentence in dataset:
     for j in range(len(sentence)):
        x_set.append(sentence[j].form)
        y_set.append(sentence[j].upos)
  return x_set, y_set

def my_one_hot_encoding(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def one_hot_to_class(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences


train = pyconll.load_from_file(UD_ENGLISH_TRAIN)
validation =  pyconll.load_from_file(UD_ENGLISH_VALIDATION)
test =  pyconll.load_from_file(UD_ENGLISH_TEST)

x_train_words, y_train_tags = data_extract(train)
x_valid_words, y_valid_tags = data_extract(validation)
x_test_words, y_test_tags = data_extract(test)

word2index = {w: i + 2 for i, w in enumerate(list(set(x_train_words)))}
word2index['<PAD>'] = 0  # The special value used for padding
word2index['<OOV>'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(set(y_train_tags)))}
tag2index['<PAD>'] = 0  # The special value used to padding


x_train_seq, train_tags_y = sentence_indexed(train,word2index,tag2index)
x_val_seq, val_tags_y = sentence_indexed(validation,word2index,tag2index)
x_test_seq, test_tags_y = sentence_indexed(test,word2index,tag2index)


MAX_LENGTH = len(max(x_train_seq, key=len))
MAX_LENGTH = 89

print(MAX_LENGTH)

x_train_seq = pad_sequences(x_train_seq, maxlen=MAX_LENGTH, padding='post')
x_val_seq = pad_sequences(x_val_seq, maxlen=MAX_LENGTH, padding='post')
x_test_seq = pad_sequences(x_test_seq, maxlen=MAX_LENGTH, padding='post')

train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
val_tags_y = pad_sequences(val_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

cat_train_tags_y = my_one_hot_encoding(train_tags_y, len(tag2index))
cat_val_tags_y = my_one_hot_encoding(val_tags_y, len(tag2index))
cat_test_tags_y = my_one_hot_encoding(test_tags_y, len(tag2index))



MAX_WORDS = 5000
EMBEDDING_DIM = 300
# Define model's embedding matrix
word_set = set(x_train_words)
MAX_WORDS = len(word_set) + 1

embedding_matrix = np.zeros((MAX_WORDS+1, EMBEDDING_DIM))
i = 0
for word in word_set:
    if i > MAX_WORDS:
            continue
    try:
        embedding_vector = fasttext_embed[fasttext_word_to_index[word],:]
        embedding_matrix[i] = embedding_vector
        i+=1
    except:
        pass
#print(i)

batch_size = 128
epochs = 40

RNN_model_adam = make_model(
    hidden_activation='relu',
    output_activation='softmax',
    dropout_rate=0.55,
    MAX_WORDS = MAX_WORDS,
    EMBEDDING_DIM = EMBEDDING_DIM,
    MAX_SEQUENCE_LENGTH = MAX_LENGTH ,
    embedding_matrix = embedding_matrix,
    classes_no = 18,
    batch_size = batch_size
    )


# Keras Callbacks
np.random.seed(1402) # Define the seed for numpy to have reproducible experiments.
set_random_seed(1981) # Define the seed for Tensorflow to have reproducible experiments.
#with tf.Session() as sess:
#     sess.run(tf.local_variables_initializer())
#K.set_session(tf.Session())
#K.get_session().run(tf.local_variables_initializer())

lr_reducer = keras.callbacks.ReduceLROnPlateau(factor = 0.25, patience = 2, min_lr = 1e-6, verbose = 1)
check_pointer = keras.callbacks.ModelCheckpoint(model_file_name, verbose = 1, save_best_only = True)
early_stopper = keras.callbacks.EarlyStopping(patience = 5) # Change 4 to 8 in the final run    
csv_logger = keras.callbacks.CSVLogger(log_file_name)

keras.backend.get_session().run(tf.global_variables_initializer())
RNN_hs_model = RNN_model_adam.fit(
    x = x_train_seq,
    y = my_one_hot_encoding(train_tags_y, len(tag2index)),
    validation_data = (x_val_seq,cat_val_tags_y),
    epochs=epochs,
    shuffle = True,
    verbose=1,
    batch_size=batch_size,
    callbacks = [early_stopper, check_pointer, lr_reducer,  csv_logger]
    )



#*****************************************************
# Load Model if needed for testing instead of training
#*****************************************************

#weights = np.ones((18,))
#weights[0] = 0.05 # PADs Minimum value for all instances to be TP
#weights[1] = 0.5 # ADJ
#weights[8] = 1 # NOUN
#weights[16] = 0.5 # VERB
#RNN_model_adam = load_model(MODELS_PATH+'\BestModel.h1', 
#                            custom_objects= {'loss': weighted_categorical_crossentropy(weights),
#                                             'precision':precision,
#                                              'recall':recall,
#                                               'f1':f1,
#                                               'accuracy':accuracy,
#                                               'my_class_accuracy':my_class_accuracy})


RNN_eval_adam = RNN_model_adam.evaluate(x_test_seq, cat_test_tags_y, verbose=1)


# DO NOT USE UNLESS NEEDED clean_up(model=RNN_model_adam)


print("Train Loss     : {0:.5f}".format(RNN_hs_model.history['loss'][-1]))
print("Validation Loss: {0:.5f}".format(RNN_hs_model.history['val_loss'][-1]))
print("Test Loss      : {0:.5f}".format(RNN_eval_adam[0]))
print("---")

print("Train Accuracy     : {0:.5f}".format(RNN_hs_model.history['accuracy'][-1]))
print("Validation Accuracy: {0:.5f}".format(RNN_hs_model.history['val_accuracy'][-1]))

print("Test categorical_cross_entropy:{0:.5f}".format(RNN_eval_adam[0]))
print("Test precision      : {0:.5f}".format(RNN_eval_adam[1]))
print("Test recall      : {0:.5f}".format(RNN_eval_adam[2]))
print("Test f1      : {0:.5f}".format(RNN_eval_adam[3]))
print("Test accuracy        : {0:.5f}".format(RNN_eval_adam[4]))
print("Test accuracy w/o PAD: {0:.5f}".format(RNN_eval_adam[5]))

plot_history(hs={'RNN': RNN_hs_model}, epochs=epochs, metric='loss')
plot_history(hs={'RNN': RNN_hs_model}, epochs=epochs, metric='accuracy')
#---------------------------------------------------------
# TO DO MORE WORK FOR RESULTS
# UNDER CONSTRUCTION

# Predict and return one-hot targets to labels
predictions = RNN_model_adam.predict(x_test_seq)
y_predict_class = one_hot_to_class(predictions, {i: t for t, i in tag2index.items()})
y_test_class = one_hot_to_class(cat_test_tags_y, {i: t for t, i in tag2index.items()})


#----- Analysis including PADS ------
#Flattent predict and test for passing below for comparison
y_predict_class_flat = list(itertools.chain(*y_predict_class))
y_test_class_flat = list(itertools.chain(*y_test_class))

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

# ----- Analysis Excluding PADS ------

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


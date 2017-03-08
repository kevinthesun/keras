'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

import profiler
import multi_gpu

#Result dictionary
global ret_dict
ret_dict = dict()

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model = multi_gpu.make_model(model, optimizer='adam',
                                                loss='binary_crossentropy', metrics=['accuracy'])

print('Train...')
with profiler.Profiler(ret_dict):
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=4,
              validation_data=[X_test, y_test])

ret_dict["training_time"] = str(ret_dict["training_time"]) + ' sec'
ret_dict["training_accuracy"] = model.evaluate(X_train, y_train, verbose=0)[1]
ret_dict["test_accuracy"] = model.evaluate(X_test, y_test, verbose=0)[1]

from keras.layers.containers import Graph
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten, GaussianNoise
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.backend
import numpy as np
import h5py
from skimage.transform import rotate
import pandas as pd

nb_filters = 32

batch_size = 100

model = Sequential()
model.add(Dropout(0.25, batch_input_shape = (batch_size,1,28,28)))
model.add(GaussianNoise(0.2))
model.add(Convolution2D(nb_filters, 7, 7, border_mode = 'same', activation = 'relu', subsample=(2,2)))
model.add(Convolution2D(nb_filters, 1, 1, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(nb_filters, 3, 3, border_mode = 'same', activation = 'relu'))

def incept(input_shape, nm):
    inception = Graph()
    input_nm = nm + '_input'
    inception.add_input(name = input_nm, input_shape = input_shape)
    inception.add_node(Convolution2D(nb_filters, 1, 1,activation = 'relu', border_mode='same'), input = input_nm, name = nm + '11')
    inception.add_node(Convolution2D(nb_filters, 1, 1, activation = 'relu', border_mode='same'), input = input_nm, name = nm + '1133')
    inception.add_node(Convolution2D(nb_filters, 1, 1, activation = 'relu', border_mode='same'), input = input_nm, name = nm + '1155')
    inception.add_node(Convolution2D(nb_filters, 1, 1, activation = 'relu', border_mode='same'), input = input_nm, name = nm + '1177')
    inception.add_node(MaxPooling2D(pool_size = (3,3), strides = (1,1), border_mode='same'), input = input_nm, name = nm + 'max11')
    inception.add_node(Convolution2D(nb_filters, 3, 3, activation = 'relu', border_mode='same'), input = nm + '1133', name = nm + '33')
    inception.add_node(Convolution2D(nb_filters, 5, 5, activation = 'relu', border_mode='same'), input = nm + '1155', name = nm + '55')
    inception.add_node(Convolution2D(nb_filters, 7, 7, activation = 'relu', border_mode='same'), input = nm + '1177', name = nm + '77')
    inception.add_node(Convolution2D(nb_filters, 1, 1, activation ='relu', border_mode='same'), input = nm + 'max11', name = nm + '11out')
    inception.add_output(name = nm + "_out",
                         inputs = [nm + '11', nm + '33', nm + '55', nm + '77', nm + '11out'],
                         concat_axis = 1)
    return inception

model.add(incept(model.output_shape[1:4], "i1"))
model.add(incept(model.output_shape[1:4], "i2"))

model.add(Convolution2D(nb_filters, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

f = h5py.File("mnist.hdf5", 'r')
train_data = f['train'][:,:,:].reshape((42000, 1, 28, 28)).astype(keras.backend.floatx()) / 256
train_labels = f['train_labels']
labels = np_utils.to_categorical(train_labels, 10)
checkpointer = ModelCheckpoint(filepath= "./enhanced_{epoch:02d}_{val_acc:05f}_.hdf5",
                               monitor = 'val_acc', verbose = 1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')

class ValidationIterator(object):
    def __init__(self, test_data, labels):
        self.test_data = test_data
        self.labels = labels
        self.test_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next()

    def next(self):
        len = batch_size
        indices = np.mod(np.arange(self.test_idx, self.test_idx + len), self.test_data.shape[0])
        self.test_idx = self.test_idx + len
        return (self.test_data[indices,:,:,:], self.labels[indices,:])

class MnistEnhancer(object):
    def __init__(self, train, tags, valsplit):
        testidx = np.random.choice(train.shape[0], size = train.shape[0] * valsplit, replace = False)
        trainidx = np.ones(train.shape[0],dtype = np.bool)
        trainidx[testidx] = 0
        self.train_data = train[trainidx,:,:,:]
        self.test_data = train[testidx,:,:,:]
        self.train_labels = tags[trainidx,:]
        self.test_labels = tags[testidx,:]
        self.train_sz = self.train_data.shape[0]
        self.epoch()

    def getValidation(self):
        return ValidationIterator(self.test_data, self.test_labels)

    def epoch(self):
        sequence = np.random.choice(self.train_sz, self.train_sz, replace = False)
        self.this_train_data = self.train_data[sequence,:,:,:].copy()
        self.this_train_labels = self.train_labels[sequence,:].copy()
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        len = batch_size
        indices = np.mod(np.arange(self.idx, self.idx + len), self.train_sz)
        thisImage = self.this_train_data[indices, :, :, :]
        for idx in range(len):
            rotimage = rotate(thisImage[idx,0,:,:], angle = np.random.normal(loc = 0, scale = 20))
            thisImage[idx,0,:,:] = rotimage
        thisLabel = self.this_train_labels[indices,:]
        self.idx = self.idx + batch_size
        if (self.idx > self.train_sz):
            self.epoch()
        return (thisImage, thisLabel)

gen = MnistEnhancer(train_data, labels, 0.2)
validata = gen.getValidation()

model.fit_generator(gen, samples_per_epoch= 128 * 262, nb_epoch=1000,
                    callbacks = [checkpointer, earlystopper], nb_worker = 1,
                    validation_data = validata, show_accuracy=True, nb_val_samples = 8400)

def maketest(model, nm):
    test_data = f['test'][:,:,:].reshape((28000, 1, 28, 28)).astype(keras.backend.floatx()) / 256
    test_predictions = model.predict(test_data, batch_size = batch_size)
    test_predictions = np.argmax(test_predictions, axis = 1)
    outframe = pd.DataFrame({'ImageId' : np.arange(28000) + 1,
                             'Label' : test_predictions})
    outframe.to_csv(nm, index = False)
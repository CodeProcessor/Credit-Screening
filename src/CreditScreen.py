'''
Created on 7/22/20

@author: dulanj
'''
import logging
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CreditScreen():
    def __init__(self):
        # create logger
        logging.basicConfig(level=logging.DEBUG)
        self.df = self.load_data()
        self.clean_data()
        self.convert_data()
        self.model = self.create_model2()
        self.train_with_kfold()

    def load_data(self):
        df = pd.read_csv('../data/crx.data')
        logging.debug(df.head())
        print(df['label'].values[0])
        return df

    def clean_data(self):
        self.df.replace('?', np.nan, inplace=True)
        self.df['label'].replace('+', 1, inplace=True)
        self.df['label'].replace('-', 0, inplace=True)
        self.df.dropna(how='any', inplace=True)

    def convert_data(self):
        self.df = pd.get_dummies(self.df, columns=['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'])
        for col in self.df.columns:
            if col == 'label':
                continue
            self.df[col].astype(float)
        logging.info(self.df.head())
        logging.info(self.df.shape)

    def create_model(self):
        # input = tf.keras.Input(shape=(46,1))
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, input_shape=(46,1), activation=tf.nn.relu)
        x = tf.keras.layers.MaxPool1D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(30, activation=tf.nn.relu)(input)
        output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)

        model = tf.keras.Model(inputs=input, outputs=output)
        logging.info(model.summary())
        
        # Compile model
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_model2(optimizer='rmsprop', init='glorot_uniform'):
        model = Sequential()
        model.add(Conv1D(filters=8,
                         kernel_size=2,
                         input_shape=(46, 1),
                         kernel_initializer=init,
                         activation='relu'
                         ))
        model.add(MaxPooling1D())

        model.add(Conv1D(8, 2, activation='relu'))
        model.add(MaxPooling1D())

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(units=8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        # opt = Keras.optimizers.SGD(lr=0.01, momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

        return model

    def train_with_kfold(self):
        n_folds = 5

        train_data = self.df.drop(['label'],axis=1).values
        train_label = self.df['label'].values

        kfold = KFold(n_folds, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(train_data):
            # select rows for train and test
            trainX, trainY, testX, testY = train_data[train_ix], train_label[train_ix], \
                                           train_data[test_ix], train_label[test_ix]

            print(trainX.shape)
            print(testX.shape)

            # fit model
            trainX = np.asarray(trainX, dtype=float)
            testX = np.asarray(testX, dtype=float)

            # trainX = trainX.reshape(trainX.shape[0], 46, 1)
            # testX = testX.reshape(testX.shape[0], 46, 1)
            trainX = np.expand_dims(trainX, axis=2)
            testX = np.expand_dims(testX, axis=2)
            print(trainX.shape)
            print(testX.shape)

            trainY = np.asarray(trainY)
            testY = np.asarray(testY)
            # trainY = trainY.reshape(trainY.shape[0], 1)
            # testY = testY.reshape(testY.shape[0], 1)
            # trainY = np.expand_dims(trainY, axis=1)
            # testY = np.expand_dims(testY, axis=1)


            print(trainY.shape)
            print(testY.shape)

            history = self.model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
            # evaluate model
            _, acc = self.model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))

    def main(self):
        pass


if __name__ == "__main__":
    obj = CreditScreen()
    obj.main()

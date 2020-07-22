'''
Created on 7/22/20

@author: dulanj
'''
import logging
import pandas as pd
import tensorflow as tf


class CreditScreen():
    def __init__(self):
        # create logger
        logging.basicConfig(level=logging.DEBUG)
        self.df = self.load_data()
        self.clean_data()
        self.model = self.create_model()

    def load_data(self):
        df = pd.read_csv('../data/crx.data')
        logging.debug(df.head())
        return df

    def clean_data(self):
        self.df.dropna(inplace=True)

    def create_model(self):
        input = tf.keras.Input(shape=(15,1))
        x = tf.keras.layers.Dense(30, activation=tf.nn.relu)(input)
        output = tf.keras.layers.Dense(2,activation=tf.nn.softmax)(x)

        model = tf.keras.Model(inputs=input, outputs=output)
        logging.info(model.summary())
        
        #Compile model
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def main(self):
        pass


if __name__ == "__main__":
    obj = CreditScreen()
    obj.main()
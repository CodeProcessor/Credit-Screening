'''
Created on 7/22/20

@author: dulanj
'''
import logging
import pandas as pd


class CreditScreen():
    def __init__(self):
        # create logger
        logging.basicConfig(level=logging.DEBUG)
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv('../data/crx.data')
        logging.debug(df.head())
        return df

    def main(self):
        pass


if __name__ == "__main__":
    obj = CreditScreen()
    obj.main()
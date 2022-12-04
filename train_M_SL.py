import numpy as np
import pandas as pd
from utils import *
from model_class import *

if __name__ == '__main__':
    import pandas as pd
    train_data = np.asarray(pd.read_csv('data/data1_training_data.csv'))
    clf_list = [RandomPick, SameClass, DecisionTree, RandomForest]
    # clf_list = [RandomPick, SameClass]
    X_train, y_train = train_data[:,:-1], train_data[:,-1]
    train_SL(clf_list, X_train, y_train, path="trained_models/M/")



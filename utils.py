from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os
from glob import glob
from collections import Counter

def split_data(data, train_fraction, random_state=2022):
    rng = np.random.default_rng(random_state)
    shuffle_indices = rng.permutation(data.shape[0])
    train_size = int(train_fraction*data.shape[0])
    train_idx, test_idx = shuffle_indices[:train_size], shuffle_indices[train_size:]
    if isinstance(data, np.ndarray):
        train_data, test_data = data[train_idx, :], data[test_idx, :]
    elif isinstance(data, pd.DataFrame):
        train_data, test_data = data.iloc[train_idx, :], data.iloc[test_idx, :]
    else:
        print("Error: data type is allowed only np.array or pd.DataFrame")
    return train_data, test_data

def save(model, path):
    with open(path,"wb") as f:
        pickle.dump(model, f)

def load(path):
    with open(path,"rb") as f:
        model = pickle.load(f)
    return model

def train_SL(clf_list, X, y, path):
    for standardize in [False, True]:  
        for clf_class in clf_list:
            clf = clf_class().cross_val_fit(X, y, standardize)
            # must run cross_val_fit firstly, otherwise name_ won't create
            clf.save_results(os.path.join(path, f"{clf.name_}.pkl"))

def show_task_result(path, train_data, test_data):
    X_train, y_train = train_data[:,:-1], train_data[:,-1]
    X_test, y_test = test_data[:,:-1], test_data[:,-1]
    print("training data : ", dict(sorted(Counter(y_train).items())))
    print("test data : ", dict(sorted(Counter(y_test).items())))

    best_score, best_name = 0, ""
    for file in sorted(glob(os.path.join(path, '*.pkl'))):
        model = load(file)
        name = model.name_
        if "SameClass" in model.name_ or "RandomPick" in model.name_:
            pass
        else:
            model.print_best_param()
        score = model.print_accuracy(X_test, y_test)
        if score > best_score:
            best_score = score
            best_name = name
    print(f"***{best_name} wins***")



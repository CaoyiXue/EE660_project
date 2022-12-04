from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
from copy import deepcopy
import utils


class BaseClass(object):
    def __init__(self):
        super().__init__()

    def cross_val_fit(self, X, y, standardize=False, name=""):
        self.standardize = standardize
        self.name_ = name
        if self.standardize:
            self.name_ = "standardized_" + self.name_
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)

        self.get_best_param(X, y)
        self.__init__(**self.best_param)
        self.fit(X, y)

        return self

    def print_best_param(self):
        try:
            print(f"{self.name_} best parameter: {self.best_param}")
            return self.best_param
        except:
            print("Error: run get_best_param or cross_val_fit first")

    def print_accuracy(self, X, y):
        try:
            if self.standardize and hasattr(self, 'scaler'):
                X = self.scaler.transform(X)
            acc = accuracy_score(y, self.predict(X))*100
            print(f"{self.name_} accuracy: {acc:.2f}%")
            return acc
        except:
            print("Error: run get_best_param or cross_val_fit first")

    def save_results(self, path):
        try:
            print(f"save {self.name_} with best parameter")
            utils.save(self, path)
        except:
            print("Error: run get_best_param or cross_val_fit first")


class RandomPick(BaseClass):
    '''unifromly random pick one class'''

    def __init__(self, random_state=2022):
        super().__init__()
        self.classes = None
        self.random_state = random_state

    def fit(self, X, y):
        self.classes = np.unique(y)
        return self

    def predict(self, X):
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self.classes, X.shape[0])

    def get_best_param(self, X, y, pre_param=None):
        self.best_param = {}
        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        name = self.__class__.__name__
        return super().cross_val_fit(X, y, standardize, name)


class SameClass(BaseClass):
    '''select the second class'''

    def __init__(self):
        super().__init__()
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes[1])

    def get_best_param(self, X, y, pre_param=None):
        self.best_param = {}
        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        name = self.__class__.__name__
        return super().cross_val_fit(X, y, standardize, name)


class DecisionTree(DecisionTreeClassifier, BaseClass):
    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=2022):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state)

    def get_best_param(self, X, y, pre_param={}):
        best_score = 0
        param = deepcopy(pre_param)
        best_param = deepcopy(param)
        for criterion in ["gini", "entropy"]:
            for max_depth in range(5, 25):
                for min_samples_split in range(2, 12, 2):
                    param['criterion'] = criterion
                    param['max_depth'] = max_depth
                    param['min_samples_split'] = min_samples_split
                    param['min_samples_leaf'] = min_samples_split//2
                    self.__init__(**param)
                    score = np.mean(cross_val_score(
                        self, X, y, cv=5, n_jobs=4))
                    if score > best_score:
                        best_score = score
                        best_param = deepcopy(param)

        self.best_param = deepcopy(best_param)
        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        name = self.__class__.__name__
        return super().cross_val_fit(X, y, standardize, name)


class RandomForest(RandomForestClassifier, BaseClass):
    def __init__(self,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="sqrt",
                 n_jobs=4,
                 random_state=2022,
                 max_samples=None):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            max_samples=max_samples
        )

    def get_best_param(self, X, y, pre_param={}):
        best_score = 0
        param = deepcopy(pre_param)
        best_param = deepcopy(param)
        for n_estimators in range(300, 500, 50):
            for max_samples in np.arange(0.8, 1.0, 0.1):
                for max_features in np.arange(0.8, 1.0, 0.1):
                    param['n_estimators'] = n_estimators
                    param['max_samples'] = max_samples
                    param['max_features'] = max_features
                    super().__init__(**param)
                    score = np.mean(cross_val_score(
                        self, X, y, cv=5, n_jobs=4))
                    if score > best_score:
                        best_score = score
                        best_param = deepcopy(param)

        self.best_param = deepcopy(best_param)
        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        self.standardize = standardize
        self.name_ = self.__class__.__name__
        if self.standardize:
            self.name_ = "standardized_" + self.name_
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)

        self.get_best_param(X, y, DecisionTree().get_best_param(X, y))
        self.__init__(**self.best_param)
        self.fit(X, y)

        return self


class LogisticRegression(LogisticRegression, BaseClass):
    def __init__(self,
                 penalty="l2",
                 C=1,
                 random_state=2022,
                 solver="liblinear",
                 max_iter=1000):
        super().__init__(
            penalty=penalty,
            C=C,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter)

    def get_best_param(self, X, y, pre_param={}):
        best_score = 0
        param = deepcopy(pre_param)
        best_param = deepcopy(param)
        for penalty in ['l1', 'l2']:
            for log2C in np.linspace(-10, 10, num=41):
                param['penalty'] = penalty
                param['C'] = 2**log2C
                self.__init__(**param)
                score = np.mean(cross_val_score(self, X, y, cv=5, n_jobs=4))
                if score > best_score:
                    best_score = score
                    best_param = deepcopy(param)
        self.best_param = deepcopy(best_param)

        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        name = self.__class__.__name__
        return super().cross_val_fit(X, y, standardize, name)


class SVM_rbf(SVC, BaseClass):
    def __init__(self,
                 C=1,
                 kernel="rbf",
                 gamma="scale"):
        super().__init__(
            C=C,
            kernel=kernel,
            gamma=gamma,
            cache_size=1024)

    def get_best_param(self, X, y, pre_param={}):
        best_score = 0
        param = deepcopy(pre_param)
        best_param = deepcopy(param)
        for C in np.logspace(-1, 3, 5):
            for gamma in np.logspace(-1, 3, 5):
                param['C'] = C
                param['gamma'] = gamma
                self.__init__(**param)
                score = np.mean(cross_val_score(self, X, y, cv=5, n_jobs=4))
                if score > best_score:
                    best_score = score
                    best_param = deepcopy(param)
        self.best_param = deepcopy(best_param)

        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        name = self.__class__.__name__
        return super().cross_val_fit(X, y, standardize, name)


class AdaBoost(AdaBoostClassifier, BaseClass):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1,
                 algorithm="SAMME",
                 random_state=2022):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state)

    def get_best_param(self, X, y, pre_param={}):
        best_score = 0
        param = deepcopy(pre_param)
        best_param = deepcopy(param)
        for max_depth in [1, 2]:
            for n_estimators in np.arange(200, 300, 10):
                param['base_estimator'] = DecisionTreeClassifier(max_depth=max_depth)
                param['n_estimators'] = n_estimators
                self.__init__(**param)
                score = np.mean(cross_val_score(self, X, y, cv=5, n_jobs=4))
                if score > best_score:
                    best_score = score
                    best_param = deepcopy(param)
        self.best_param = deepcopy(best_param)
        return self.best_param

    def cross_val_fit(self, X, y, standardize=False):
        name = self.__class__.__name__
        return super().cross_val_fit(X, y, standardize, name)

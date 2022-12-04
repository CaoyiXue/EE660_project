from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SA_ee660:
    def __init__(self, estimator, n_components, random_state=2022):
        # n_components: feature dimension after PCA
        self.estimator_ = estimator
        self.n_components = n_components
        self.sign_mtx = np.eye(n_components)
        self.random_state = random_state

    def fit(self, Xs, Ys, Xt, Xtl=None, Ytl=None):
        # training of SA
        # Xs: source domain feature matrix, [Ns, D_feat]
        # Ys: labels in source domain
        # Xt: target domain feature matrix, [Nt, D_feat]
        # Xtl, Ytl: labeled target data. Do sign flipping if they are not None.

        self.pca_src_ = PCA(self.n_components)
        self.pca_src_.fit(Xs)
        self.pca_tgt_ = PCA(self.n_components)
        self.pca_tgt_.fit(Xt)

        self.M_ = self.pca_src_.components_ @ self.pca_tgt_.components_.T

        Xs_tf = self.transform(Xs, domain="src")
        self.estimator_.fit(Xs_tf, Ys)

        if Xtl is not None and Ytl is not None:
            bestacc = self.score(Xtl, Ytl)
            for i in range(self.n_components):
                self.sign_mtx[i, i] = -1
                acc = self.score(Xtl, Ytl)
                if acc > bestacc:
                    bestacc = acc
                else:
                    self.sign_mtx[i, i] = 1

        return self

    def predict(self, X):
        X_tf = self.transform(X, domain="tgt")
        if hasattr(self.estimator_, "predict"):
            predict_output = self.estimator_.predict(X_tf)
        else:
            raise ValueError(
                "Estimator does not implement predict method")
        return predict_output
        
    def score(self, X, y):
        # test trained SA on testing set X and y
        # return the testing accuracy, value in [0,1]
        X_tf = self.transform(X, domain="tgt")
        if hasattr(self.estimator_, "predict"):
            predict_output = self.estimator_.predict(X_tf)
            score = accuracy_score(predict_output, y)
        else:
            raise ValueError(
                "Estimator does not implement predict method")
        return score

    def transform(self, X, domain="tgt"):
        if domain in ["tgt"]:
            return self.pca_tgt_.transform(X) @ self.sign_mtx
        elif domain in ["src"]:
            return self.pca_src_.transform(X) @ self.M_
        else:
            raise ValueError(
                "`domain `argument should be `tgt` or `src`, got, %s" % domain)

def accuracy_percent(model, X, y):
    return accuracy_score(y, model.predict(X))*100

def experiment_SA(estimator, data_S, data_T, train_T_fraction=0.2,
                  sign_flip_fraction=0.05, n_components=None,
                  standardized=True, sign_flip=True, random_state=2022):

    data_S, data_T = np.asarray(data_S).copy(), np.asarray(data_T).copy()
    train_T, test_T = utils.split_data(data_T, train_T_fraction, random_state)
    Xs, Ys = data_S[:, :-1], data_S[:, -1]
    n_components = Xs.shape[1] if n_components is None else n_components
    Xt, Yt = train_T[:, :-1], train_T[:, -1]
    Xt_test, Yt_test = test_T[:, :-1], test_T[:, -1]
    if standardized:
        scaler = StandardScaler().fit(Xs)
        Xs = scaler.transform(Xs)
        scaler.fit(Xt)
        Xt = scaler.transform(Xt)
        Xt_test = scaler.transform(Xt_test)

      # select Ntl points with labels, for sign flipping.
    # Xt, Yt are feature matrix and labels for target domain (after standardization if there is)
    rng = np.random.default_rng(random_state)
    random_indices = rng.permutation(int(sign_flip_fraction*Xt.shape[0]))
    if sign_flip and len(random_indices) != 0:
        Xtl, Ytl = Xt[random_indices], Yt[random_indices]
        Xs, Ys = np.concatenate(
            (Xs, Xtl), axis=0), np.concatenate((Ys, Ytl), axis=0)
    else:
        Xtl, Ytl = None, None

    # SA
    sa_model = SA_ee660(estimator, n_components, random_state=random_state)
    sa_model.fit(Xs, Ys, Xt, Xtl, Ytl)
    sa_acc = accuracy_percent(sa_model, Xt_test, Yt_test)

    #Supervised Learning
    estimator.fit(Xs, Ys)
    sl_acc = accuracy_percent(estimator, Xt_test, Yt_test)

    return sa_acc, sl_acc

def decide_n_components(model, data_S, data_T):
    best_acc, best_n = 0, 1
    for n_components in range(1,12):
        sa_acc, sl_acc= experiment_SA(model, data_S, data_T,
                        train_T_fraction=0.1, sign_flip_fraction=0,
                        standardized=True, sign_flip=True)
        if sa_acc > best_acc:
            best_acc = sa_acc
            best_n = n_components
    return best_n

def get_vary_result(model, data_S, data_T):
    train_T_fraction_range = np.arange(0.9, 0, -0.1)
    sign_flip_fraction_range = np.arange(0, 1, 0.1)

    final_res = []
    for ttf in train_T_fraction_range:
        one_res = []
        for sff in sign_flip_fraction_range:
            sa_acc, sl_acc= experiment_SA(model, data_S, data_T,
                        train_T_fraction=ttf, sign_flip_fraction=sff,
                        standardized=True, sign_flip=True)
            one_res.append(sa_acc-sl_acc)
        final_res.append(one_res)

    return final_res

def plot_result(model_list, data_S, data_T, save_path="figs/SA_result.png"):
    fig = plt.figure(figsize=(13, 5))
    num = len(model_list)
    rows, cols = 1, num
    for i, model in enumerate(model_list):
        res = get_vary_result(model, data_S, data_T)
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(f"{model.__class__.__name__}")
        if i == 0:
            ax.set_ylabel("target train fraction")
            ax.set_yticks(np.arange(0,9), np.round(np.arange(0.9,0,-0.1),1))
        else:
            ax.set_yticks([])
        ax.set_xlabel("sign flip fraction")
        ax.set_xticks(np.arange(0, 10), np.round(np.arange(0, 1, 0.1),1))
        im = ax.imshow(res)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    fig.savefig(save_path)

if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import matplotlib.pyplot as plt

    data_S = pd.read_csv('data/data3_source_data.csv')
    data_T = pd.read_csv('data/data3_target_data.csv')
    knn = KNeighborsClassifier(n_neighbors=5)
    param = {'criterion': 'entropy',
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5}
    CART = DecisionTreeClassifier(**param)
    param = {'criterion': 'entropy',
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'n_jobs': -1,
                'n_estimators': 50,
                'max_samples': 0.9,
                'max_features': 0.8}
    RF = RandomForestClassifier(**param)
    model_list = [knn, CART, RF]
    plot_result(model_list, data_S, data_T)

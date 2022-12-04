import pandas as pd
def mutil2binary(data):
    if isinstance(data, pd.DataFrame):
        # good 0, 1
        data.loc[data['class']==1, 'class'] = 0 
        # not good 2, 3
        data.loc[(data['class']==2) | (data['class']==3), 'class'] = 1
    else:
        print("Error: data type is only allowed pd.DataFrame with 'class' column name")
    return data
    
if __name__ == '__main__':
    # split dataset
    print("starting to create dataset")
    import numpy as np
    from collections import Counter
    from sklearn import preprocessing
    from sklearn.cluster import KMeans 
    import utils

    df = pd.read_csv('data/bodyPerformance.csv')
    
    # transform gender F, M to 0, 1
    le = preprocessing.LabelEncoder().fit(df['gender'])
    df['gender'] = le.transform(df['gender'])
    print(f"gender {le.inverse_transform([0,1])}=>[0 1]")
    # transform A, B, C, D to 0, 1, 2, 3
    le = preprocessing.LabelEncoder().fit(df['class'])
    df['class'] = le.transform(df['class'])
    print(f"class {le.inverse_transform([0,1,2,3])}=>[0 1 2 3]")
    df.to_csv("data/new_data.csv", index=False)

    # random split
    train_fraction = 0.7 # split train and validation as 7:3
    train_df, test_df = utils.split_data(df, train_fraction, random_state=2022)
    train_df.to_csv('data/data1_training_data.csv', index=False)
    test_df.to_csv('data/data1_test_data.csv', index=False)
    print("Muti-class Classification data:")
    print("training data : ", dict(sorted(Counter(train_df.iloc[:,-1]).items())))
    print("test data : ", dict(sorted(Counter(test_df.iloc[:,-1]).items())))

    train_df = mutil2binary(train_df)
    test_df = mutil2binary(test_df)
    train_df.to_csv('data/data2_training_data.csv', index=False)
    test_df.to_csv('data/data2_test_data.csv', index=False)
    print("Binary Classification data:")
    print("training data : ", dict(sorted(Counter(train_df.iloc[:,-1]).items())))
    print("test data : ", dict(sorted(Counter(test_df.iloc[:,-1]).items())))

    ## K-means to create transfer learning data
    X_normal = preprocessing.StandardScaler().fit_transform(df.iloc[:,:-1])
    kmeans = KMeans(n_clusters=2, random_state=2022).fit(X_normal)
    num0, num1 = (kmeans.labels_==0).sum(), (kmeans.labels_==1).sum()
    if num0 > num1:
        data_S = df.loc[kmeans.labels_==0]
        data_T = df.loc[kmeans.labels_==1]    
    else:
        data_S = df.loc[kmeans.labels_==1]
        data_T = df.loc[kmeans.labels_==0] 

    data_S.to_csv('data/data3_source_data.csv', index=False)
    data_T.to_csv('data/data3_target_data.csv', index=False)
    print("Subspace Alignment Experiment data:")
    print("source data : ", dict(sorted(Counter(data_S.iloc[:,-1]).items())))
    print("target data : ", dict(sorted(Counter(data_T.iloc[:,-1]).items())))

    # # binary 
    # data_S = mutil2binary(data_S)
    # data_T = mutil2binary(data_T)
    # data_S.to_csv('data/data4_source_data.csv')
    # data_T.to_csv('data/data4_target_data.csv')
    # print("TrAdaboost and SA Experiment data:")
    # print("source data : ", dict(sorted(Counter(data_S.iloc[:,-1]).items())))
    # print("target data : ", dict(sorted(Counter(data_T.iloc[:,-1]).items())))



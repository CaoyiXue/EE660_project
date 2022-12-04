if __name__ == '__main__':
    from utils import *
    import numpy as np
    import pandas as pd
    from model_class import *
    import matplotlib.pyplot as plt
    import PIL 

    print("Muti-class Classification result:")
    path = "trained_models/M/"
    train_data = np.asarray(pd.read_csv('data/data1_training_data.csv'))
    test_data = np.asarray(pd.read_csv('data/data1_test_data.csv'))
    show_task_result(path, train_data, test_data)

    print()

    print("Binary Classification result:")
    path = "trained_models/B/"
    train_data = np.asarray(pd.read_csv('data/data2_training_data.csv'))
    test_data = np.asarray(pd.read_csv('data/data2_test_data.csv'))
    show_task_result(path, train_data, test_data)

    print()
    print("SA result is a figure")
    im = plt.imread('figs/SA_result.png')
    plt.imshow(im)
    plt.show()




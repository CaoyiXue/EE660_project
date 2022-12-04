# EE660 Project
Before run main.sh, please run the following code in your shell to avoid unknown error due to different enviroment.

```shell
conda create --name ee660_caoyixue --file requirements.txt
conda active ee660_caoyixue
```

main.sh create same version python environment and run show\_results.py \
show\_results.py show results.\
model\_class.py contains all of my model class for AutoML system.\
train\_B\_SL.py train binary classification.\
train\_M\_SL.py train multi-class classification. \
train\_SA.py train SA. \
trained\_models store model pkl files. "B" subfolder contains models of binary classification. "M" subfolder contains models of multi-class classification


```bash
EE660_project
├── README.md
├── create_data.py
├── data
│   ├── bodyPerformance.csv
│   ├── data1_test_data.csv
│   ├── data1_training_data.csv
│   ├── data2_test_data.csv
│   ├── data2_training_data.csv
│   ├── data3_source_data.csv
│   └── data3_target_data.csv
├── figs
│   └── SA_result.png
├── main.sh
├── model_class.py
├── requirements.txt
├── show_results.py
├── train_B_SL.py
├── train_M_SL.py
├── train_SA.py
├── trained_models
│   ├── B
│   │   ├── AdaBoost.pkl
│   │   ├── DecisionTree.pkl
│   │   ├── LogisticRegression.pkl
│   │   ├── RandomForest.pkl
│   │   ├── RandomPick.pkl
│   │   ├── SVM_rbf.pkl
│   │   ├── SameClass.pkl
│   │   ├── standardized_AdaBoost.pkl
│   │   ├── standardized_DecisionTree.pkl
│   │   ├── standardized_LogisticRegression.pkl
│   │   ├── standardized_RandomForest.pkl
│   │   ├── standardized_RandomPick.pkl
│   │   ├── standardized_SVM_rbf.pkl
│   │   └── standardized_SameClass.pkl
│   └── M
│       ├── DecisionTree.pkl
│       ├── RandomForest.pkl
│       ├── RandomPick.pkl
│       ├── SameClass.pkl
│       ├── standardized_DecisionTree.pkl
│       ├── standardized_RandomForest.pkl
│       ├── standardized_RandomPick.pkl
│       └── standardized_SameClass.pkl
└── utils.py
```

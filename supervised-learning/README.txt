# Supervised Learning

# Overview
5 models - Decision tree, XGB, neural network, SVM, KNN.  
Each model has their own notebook to tune hyper-param, train and plot learning curves etc. 

# Getting started
Create virtual python environment and install requirements from `../requirements.txt`.

# Dataset
- Bank churn data (`data/bank.csv`)
- Income data (`data/adult.data`)

# Notebooks
To specify the dataset you would like to run the model on, go to their respective notebook and update `get_train_test_ds()` method in the 2nd cell.  
Next, click run all and wait for training and analysis to be completed.  
Some extra plots can be found in `overall_performance.ipynb` but the values are all hard coded - get values from running the model notebooks.  
The bank churn data is quite heavy so running notebook and that dataset might take some time, especially for SVM model.  

Check out the notebooks in github repo `https://github.com/cjiefeng/7641-ml/tree/main/supervised-learning`.

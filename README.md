# Nested-CV-binary-classification
Function to perform binary classification, estimating the model accuracy by cross validation and automatically selecting model hyperparameters by nested cross validation.

In input the function requires:
- data =  the whole dataset, with the (binary) answer in the first column
- tipo = the model with which you want to make predictions. Here I put the RF, the SVM and the binomial GLM with elastic net penalty (default), but the function is easily customizable to add as many models (that are present in caret) as desired
- kfold = number of folds in the CV
- nest_kfold = number of folds in the nested CV
- conf_matr = whether, in addition to accuracy, you also wants the confusion matrix as output
- seme = the seed to be set to have replicable results

In output:
- the model accuracy
- the confusion matrix if desired (TRUE by default).

In addition, the function prints progress on the screen via a loading bar.



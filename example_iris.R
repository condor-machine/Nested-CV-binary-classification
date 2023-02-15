###############---------------###############

# example of the use of clbin_nestkfold on iris (with only the two classes 'versicolor' and 'virginica')

source('class_bin_nested_CV.R')

db <- data.frame(y=droplevels(as.factor(iris[iris$Species != 'setosa',5])),
                 iris[iris$Species != 'setosa',-5])

clbin_nestkfold(db) # binomial GLM with elasticnet as default
clbin_nestkfold(db,'rf') # Random Forests
clbin_nestkfold(db,'svm') # Support Vector Machines

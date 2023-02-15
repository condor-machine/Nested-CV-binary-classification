


###############---------------###############

# useful packages for the classification models.
# If in the clbin_nestkfold function you want to add
# more models, you must insert here the packages that contain
# the desired models.

lapply(c('caret','ranger','glmnet','e1071'), 
       FUN = function(x) { do.call("require", list(x)) })


clbin_nestkfold <- function(data, tipo = 'elasticnet', kfold = 10, nest_kfold = 10, conf_matr = TRUE, seme = 910, ...){
  # 'data' must have the answer as the first column
  tipo <- tolower(tipo)
  set.seed(seme)
  data <- data[sample(1:NROW(data)),]
  folds <- cut(seq(1,NROW(data)), breaks = kfold, labels = FALSE)
  prev <- NULL; names(data)[1] <- 'y'
  pb <- txtProgressBar(min = 0, max = kfold, style = 3)
  prev_mod <- function(dbtr, dbts, model, ...){
    
    if(model == 'elasticnet'){ 
      set.seed(seme)
      en_tr <- caret::train(y ~., data = dbtr, method = "glmnet", tuneLength = nest_kfold,
                            trControl = trainControl("cv", number = nest_kfold, allowParallel = F), 
                            family = 'binomial', ...)
      return(predict(en_tr,dbts[,-1])) }
    
    if(model == 'svm'){ 
      set.seed(seme)
      svm_tr <- caret::train(y ~., data = dbtr, method = "svmLinearWeights", tuneLength = nest_kfold,
                             trControl = trainControl("cv", number = nest_kfold), allowParallel = F, ...)
      return(predict(svm_tr,dbts[,-1])) }
    
    
    if(model == 'rf'){ 
      set.seed(seme)
      rf_tr <- caret::train(y ~., data = dbtr, method = "ranger", tuneLength = nest_kfold,
                            trControl = trainControl("oob", allowParallel = F), ...)
      return(predict(rf_tr,dbts[,-1])) }}
  
  for(k in 1:kfold){
    test_idx <- which(folds == k, arr.ind = TRUE)
    dbts <- data[test_idx, ]
    dbtr <- data[-test_idx, ]
    prev <- c(prev, prev_mod(dbtr, dbts, model = tipo, ...)) 
    setTxtProgressBar(pb, k) }
  prev <- as.factor(prev)
  levels(prev) <- levels(data[,1])
  conf_mat <- confusionMatrix(prev, data[,1])
  accur <- conf_mat$overall[1]
  close(pb)
  if(conf_matr){ return(list(conf_matr = conf_mat$table, accuracy = accur)) }
  else{ return(list(accuracy = accur)) }}

# if the number of predictors in the dataset is less than the number of folds of the nested CV (nest_kfold),
# when using RF a warning message is printed, saying that only the possible values (< nest_kfold)
# for the mtry hyperparameter will be tested.

###############---------------###############









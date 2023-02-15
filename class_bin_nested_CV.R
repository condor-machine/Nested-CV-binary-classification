


###############---------------###############

# pacchetti utili per i modelli di classificazione.
# Se nella funzione clbin_nestkfold si vogliono aggiungere
# altri modelli bisogna inserire qui i pacchetti che contengono
# i modelli desiderati.

lapply(c('caret','ranger','glmnet','e1071'), 
       FUN = function(x) { do.call("require", list(x)) })



# funzione per applicare un modello di classificazione binaria a proprio piacimento,
# con una stima dell'errore di previsione tramite cross validation
# e una selezione automatica degli iperparametri del modello
# tramite una nested cross validation.
# input: 
#     - data = tutto il dataset, con la risposta (binaria) in prima colonna
#     - tipo = modello con il quale si vogliono effettuare le previsioni.
#              qui ho messo la RF, l'SVM e il GLM binomiale con penalizzazione elastic net (default),
#              ma la funzione è facilmente customizzabile per
#              aggiungere tutti i modelli (presenti in caret) desiderati
#     - kfold = numero di fold nella CV
#     - nest_kfold = numero di fold nella nested CV
#     - conf_matr = se vuole che ritorni anche la confusion matrix oltre all'accuracy
#     - seme = il seed da settare per avere risultati replicabili
# output:
#     - accuracy
#     - confusion matrix se si vuole (TRUE di default)

# In aggiunta la funzione stampa a video i progressi tramite una barra di caricamento.


clbin_nestkfold <- function(data, tipo = 'elasticnet', kfold = 10, nest_kfold = 10, conf_matr = TRUE, seme = 910, ...){
  # 'data' deve avere la risposta come prima colonna
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

# se il numero di predittori del dataset è minore del numero di fold della nested CV (nest_kfold),
# quando si utilizza la RF viene stampato un messaggio di avviso
# che non sono stati utilizzati nest_kfold valori unici per l'iperparametro mtry,
# ma che invece vengono troncati al numero massimo possibile, ovvero 
# che verranno testati come valori per mtry solo quelli possibili (da uno al numero totale di predittori).


##########----------##########

# esempio di utilizzo su iris (con solo le due classi 'versicolor' e 'virginica')

# db <- data.frame(y=droplevels(as.factor(iris[iris$Species != 'setosa',5])),
#                  iris[iris$Species != 'setosa',-5])
# 
# clbin_nestkfold(db,'elasticnet')
# clbin_nestkfold(db,'rf')
# clbin_nestkfold(db,'svm')












# Knn-Classifier

To find KNN
```
./knn_classifier 10 ../Resources/digits_train.json ../Resources/digits_test.json
./knn_classifier 10 ../Resources/votes_train.json ../Resources/votes_test.json
```

To tune Hyperparameter
```
./hyperparam_tune 20 ../Resources/digits_train.json ../Resources/digits_val.json ../Resources/digits_test.json 
./hyperparam_tune 20 ../Resources/votes_train.json ../Resources/votes_val.json ../Resources/votes_test.json 
```

To get Learning Curve by running KNN on different slices of traning set
```
./learning_curve 10 ../Resources/digits_train.json ../Resources/digits_test.json 
./learning_curve  10 ../Resources/digits_train.json ../Resources/digits_test.json 
```

To get ROC curve by prediction TPR and FPR
```
./roc_curve 10 ../Resources/votes_train.json ../Resources/votes_test.json 

```

This folder consists of the results of our models on each dataset. Specifically, it contains:

``` 
\---{training dataset}                                      #e.g., dblp.v12.json
|   \---{model name}                                        #fnn (feedforward nn)
|   |    \---{model's running setting}                      #t30.s11.e12.l[100].lr0.1.b4096.e2
|             state-dic_model.f{fold#}.e{epoch#}.pt         #model state per epoch
|             state-dic_model.f{fold#}.pt                   #final model state
|             train_valid_loss.json                         #loss of the model at each epoch for train and valid sets
|             fold{fold#}.png                               #figure for loss of train vs. valid over epochs
|             f{fold#}.{test,train,valid}.pred              #the predictions of trained model of fold_i on test, train (itself), valid
|             f{fold#}.{test,train,valid}.pred.eval.csv     #the evaluation of trained model of fold_i per instances of test, train (itself), valid
|             f{fold#}.{test,train,valid}.pred.eval.mean.csv#the mean of evaluation of trained model of fold_i on all instances of test, train (itself), valid
```
It is worth noting that we report the evaluation of trained models of each fold on the test set.
Also, we report the evaluation of trained models of each fold on their own training set (should overfit) and validation sets.
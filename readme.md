# ReadMe (FOR OUR USE, NOT FOR LAB MACHINES) (REAL)



## Find Tree Algorithm




## Evaluation Metrics
The evaluation function will take as inputs the true data as well as data that our trained model predicts.
The first metric that is created is the confusion matrix, this matrix has prediction labels on the x axis and true labels on the y axis. It will then iterate over both the true and predicted data and fill out the confusion matrix.
Correct predictions lie along the diagonal of the confusion matrix. False Negatives occur when the model predicts the value to not be in class i but the true value is i. False Positives are when the model predicts it to be class i, but it is not class i. True Negatives are the rest...
We can then calculate accuracy, precision, recall and f1 scores with this data for each individual class.
Accuracy = (TP + TN) / (Whole data set) 
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Once all the scores for each class has been taken, an average across all classes is returned as the final evaluation metrics for our model.

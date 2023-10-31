import numpy as np

clean_txt = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_txt = np.loadtxt("./wifi_db/noisy_dataset.txt")

def confusion_matrix(true_data, predicted_data):
    matrix = np.zeros((4,4), dtype="int")
    for i in range(len(true_data)):
        num_true = true_data[i] - 1
        num_predicted = predicted_data[i] - 1
        # predicted labels are on the top, actual labels will be on the side.
        # for example if the model predicts the data to be in room 3 and it is actually in room 3, then matrix[2][2] will have a 1.
        # True positives are along the diagonal.
        # Need to extract what the values from both the data.
        # E.g. num_true, num_pred: then matrix[num_pred][num_true] += 1
        matrix[num_true][num_predicted] += 1
    # print(matrix)
    return matrix

def evaluation_metrics(true_data, predicted_data, matrix):
    classes = len(matrix[0])
    accuracy_scores = np.zeros(classes)
    precision_scores = np.zeros(classes)
    recall_scores = np.zeros(classes)
    f1_scores = np.zeros(classes)


    # Recall for multi-class is measured as sum of true positives across all classes divided by sum of true positives + false negatives across all classes.
    # Could compute scores on a class basis and then take the average of all of them?
    for i in range(len(matrix[0])):
        # Obtaining values for how our model scores on test data.
        True_Positives = matrix[i][i] # True positives are along the diagonal
        False_Positives = sum(matrix[:, i]) - True_Positives # Predicted to be class i but is not class i.
        False_Negatives = sum(matrix[i, :]) - True_Positives # Is class i but predicted not to be class i.
        True_Negatives = np.sum(matrix) - True_Positives - False_Positives - False_Negatives # True Negatives are classes that are correctly predicted not to be i

        Accuracy = (True_Positives + True_Negatives) / (True_Negatives + True_Positives + False_Positives + False_Negatives)
        Precision = True_Positives / (True_Positives + False_Positives)
        Recall = True_Positives / (True_Positives + False_Negatives)
        F1 = 2 * ((Precision * Recall) / (Precision + Recall))

        accuracy_scores[i] = Accuracy
        precision_scores[i] = Precision
        recall_scores[i] = Recall
        f1_scores[i] = F1

    average_accuracy = round(np.mean(accuracy_scores), 3)
    average_precision = round(np.mean(precision_scores), 3)
    average_recall = round(np.mean(recall_scores), 3)
    average_f1 = round(np.mean(f1_scores), 3)

    return(average_accuracy, average_precision, average_recall, average_f1)


true_data = [1, 1, 1, 4, 4, 1, 1, 3, 3, 2, 2, 4, 1, 3, 4, 1, 1, 1, 2, 2]
predicted_data = [1, 2, 1, 3, 4, 1, 1, 3, 4, 3, 1, 3, 2, 4, 1, 2, 3, 1, 2, 2]
conf_matrix = confusion_matrix(true_data=true_data, predicted_data=predicted_data)
print(conf_matrix)
accuracy, precision, recall, f1 = evaluation_metrics(true_data, predicted_data, conf_matrix)
print(accuracy, precision, recall, f1)
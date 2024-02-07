import numpy as np
import sklearn.metrics


def get_metrics(y_pred, y_true):
    '''
    Given true and predicted labels, return accuracy, precision, recall, f1

    TODO: add support for y_true to contain labels from multiple raters, and 
    report per-rater metrics + return inter-rater agreement metric(s)
    '''
    acc = np.mean(y_true == y_pred)
    acc_t2 = np.mean(np.abs(y_true - y_pred) <= 1)  # accuracy, allowing for labels within 1 of each other
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)

    metrics = {
        'acc': acc,
        'acc_t2': acc_t2,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_mat': conf_mat
    }

    return metrics
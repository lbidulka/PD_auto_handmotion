import numpy as np
import sklearn.metrics


def get_metrics(y_pred, y_true, task):
    '''
    Given true and predicted labels, return accuracy, precision, recall, f1

    TODO: add support for y_true to contain labels from multiple raters, and 
    report per-rater metrics + return inter-rater agreement metric(s)
    '''
    metrics = {}
    for i in range(y_true.shape[1]):
        y_rater = y_true[:, i]
        acc = np.mean(y_rater == y_pred)
        acc_t2 = np.mean(np.abs(y_rater - y_pred) <= 1) # accuracy, allowing for labels within 1 of each other
        precision = np.sum((y_rater == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_rater == 1) & (y_pred == 1)) / np.sum(y_rater == 1)
        f1 = sklearn.metrics.f1_score(y_rater, y_pred, average='weighted')
        conf_mat = sklearn.metrics.confusion_matrix(y_rater, y_pred)

        metrics[i] = {
            'acc': acc,
            'acc_t2': acc_t2,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_mat': conf_mat
        }

    # accuracy, allowing for pred to match either set of labels
    acc_inter = np.mean((y_pred == y_true[:,0]) | (y_pred == y_true[:,1]))
    acc_t2_inter= np.mean((np.abs(y_true - y_pred.reshape(-1,1)) <= 1).any(axis=1))

    metrics['inter_rater'] = {
        'acc': acc_inter,
        'acc_t2': acc_t2_inter
    }

    # Binary classification
    if task == 'binclass':
        # remove acc_t2, as it's not relevant for binclass
        for key in metrics.keys():
            metrics[key].pop('acc_t2')

    return metrics
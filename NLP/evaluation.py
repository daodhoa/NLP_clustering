import numpy as np
def my_confusion_matrix(y_pred, y_true):
    N = np.unique(y_true).shape[0] # number of classes
    cm = np.zeros((N, N), dtype=int)
    for n in range(y_true.shape[0]):
        cm[y_pred[n], y_true[n]] += 1
    return cm

def purity_score(y_pred, y_true):
    cm = my_confusion_matrix(y_pred, y_true)
    return np.sum(np.amax(cm, axis=1)) / np.sum(cm)

def purity_score2(y_pred, y_true):
    cm = my_confusion_matrix(y_pred, y_true)
    max = np.amax(cm, axis=1)
    sum = np.sum(cm, axis=1)
    tong =0
    for i in range(0, 20):
        tong += max[i]/sum[i]
    return tong/20


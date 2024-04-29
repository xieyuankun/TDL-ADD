import numpy as np
from sklearn.metrics import (auc, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

if __name__ == "__main__":
    label = np.load('./scores/final_label.npy')
    pred = np.load('./scores/final_pred.npy')

    print(pred,'pred')
    print(label,'label')
    EER = calculate_eer(label,pred)

    a = np.array([0.5] * len(label))
    result = (a + pred).astype(int)
    precision, recall, f1_score, support = precision_recall_fscore_support(label,result, average="binary", beta=1.0)

    print(EER,'EER')
    print(precision, 'precision')
    print(recall,'recall')
    print(f1_score,'f1_score')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix, precision_score, recall_score, \
    accuracy_score, f1_score

import numpy as np
import torch
import transformers


def get_transfer_by_threshold(predictions, th):
    real_scores = predictions.get("real", [])
    sampled_scores = predictions.get("sampled", predictions.get("samples", []))

    y_true = [0] * len(real_scores) + [1] * len(sampled_scores)
    y_pred_scores = list(real_scores) + list(sampled_scores)
    y_pred = [1 if s >= th else 0 for s in y_pred_scores]

    tp = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(y_pred, y_true) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(y_pred, y_true) if p == 0 and y == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'fpr': float(fpr)
    }
def calc_stats(real_scores, sampled_scores):
    y = [0] * len(real_scores) + [1] * len(sampled_scores)
    scores = real_scores + sampled_scores

    ths = sorted(set(scores))
    ths = [ths[0] - 1e-12] + ths + [ths[-1] + 1e-12]

    def metrics(t):
        pred = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for p, yy in zip(pred, y) if p == 1 and yy == 1)
        fp = sum(1 for p, yy in zip(pred, y) if p == 1 and yy == 0)
        fn = sum(1 for p, yy in zip(pred, y) if p == 0 and yy == 1)
        tn = sum(1 for p, yy in zip(pred, y) if p == 0 and yy == 0)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return p, r, f1, fpr

    best = None
    for t in ths:
        p, r, f1, fpr = metrics(t)
        row = (f1, p, r, t, fpr)
        if best is None or f1 > best[0] or (abs(f1 - best[0]) < 1e-12 and t < best[3]):
            best = row

    cand = []
    for t in ths:
        p, r, f1, fpr = metrics(t)
        if fpr <= 0.01 + 1e-12:
            cand.append((r, f1, p, t, fpr))

    if not cand:
        raise ValueError("No threshold satisfies FPR <= 0.01")

    sel = max(cand, key=lambda x: (x[0], x[1], x[2], -x[3]))
    return {
        "threshold": best[3],
        "f1": best[0],
        "precision": best[1],
        "recall": best[2],
        "fpr": best[4],
    },{
        "threshold": sel[3],
        "f1": sel[1],
        "precision": sel[2],
        "recall": sel[0],
        "fpr": sel[4],
    }

def get_roc_by_threshold(real_preds, sample_preds, threshold):
    real_labels = [0]* len(real_preds) +[1]* len(sample_preds) 
    predicted_probs = real_preds + sample_preds
    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    predictions = [1 if prob >= threshold else 0 for prob in predicted_probs] 
    roc_auc = roc_auc_score(real_labels, predictions)
    conf_matrix = confusion_matrix(real_labels, predictions) 
    precision = precision_score(real_labels, predictions) 
    recall = recall_score(real_labels, predictions) 
    f1 = f1_score(real_labels,predictions)
    accuracy = accuracy_score(real_labels, predictions)

    
    return (round(roc_auc * 100, 2), round(pr_auc * 100, 2), float(threshold), conf_matrix.tolist(), round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2), round(accuracy * 100, 2), round(tpr_at_fpr_0_01 * 100, 2))
    
def get_roc_metrics_rep(real_preds, sample_preds):
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)

    predictions = [1 if prob >= optimal_threshold else 0 for prob in predicted_probs]
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)
    # Interpolate the TPR at FPR = 0.01%, this is a fixed point in roc curve
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    print(np.interp(0.95, tpr, fpr))

    return (round(roc_auc * 100, 2), round(pr_auc * 100, 2), float(optimal_threshold), conf_matrix.tolist(), round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2), round(accuracy * 100, 2), round(tpr_at_fpr_0_01 * 100, 2))



def get_classification_thresholds(real_preds, sample_preds, target_fpr=0.001):
    """
    Calculate optimal classification thresholds using different strategies.

    Args:
        real_preds (list or array): Predictions for real data (lower scores preferred).
        sample_preds (list or array): Predictions for sample data (higher scores preferred).
        target_fpr (float): Target false positive rate for fixed FPR threshold.

    Returns:
        dict: Thresholds for 'youden' (Youden's J), 'f1' (max F1), 'fixed_fpr'.
    """
    y_true = np.concatenate([np.ones_like(real_preds), np.zeros_like(sample_preds)])
    y_pred_prob = np.concatenate([real_preds, sample_preds])

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)

    youden_j = tpr - fpr
    best_idx_youden = np.argmax(youden_j)
    thresh_youden = thresholds_roc[best_idx_youden]

    _, _, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = [f1_score(y_true, (y_pred_prob >= t).astype(int)) for t in thresholds_pr]
    best_idx_f1 = np.argmax(f1_scores)
    thresh_f1 = thresholds_pr[best_idx_f1]

    if np.max(fpr) < target_fpr:
        thresh_fixed_fpr = thresholds_roc[-1]
    elif np.min(fpr) > target_fpr:
        thresh_fixed_fpr = thresholds_roc[0]
    else:
        idx_fixed_fpr = np.where(fpr <= target_fpr)[0][-1]
        thresh_fixed_fpr = thresholds_roc[idx_fixed_fpr]

    return {
        'youden': thresh_youden,
        'f1': thresh_f1,
        'fixed_fpr': thresh_fixed_fpr
    }


def get_all_metrics(real_preds, sample_preds):
    """
    Compute comprehensive classification metrics including ROC AUC and optimal threshold.

    Args:
        real_preds (list or array): Predictions for real data.
        sample_preds (list or array): Predictions for sample data.

    Returns:
        tuple: (roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01, tpr_at_fpr_0_05)
    """
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    # Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions = [1 if prob >= optimal_threshold else 0 for prob in predicted_probs]
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    tpr_at_fpr_0_05 = np.interp(0.05 / 100, fpr, tpr)

    return float(roc_auc), float(optimal_threshold), conf_matrix.tolist(), float(
        precision), float(recall), float(f1), float(accuracy), float(tpr_at_fpr_0_01), float(tpr_at_fpr_0_05)


def get_transfer_metrics(real_preds, sample_preds, threshold):
    """
    Compute metrics using a fixed threshold (e.g., for transfer learning).

    Args:
        real_preds (list or array): Predictions for real data.
        sample_preds (list or array): Predictions for sample data.
        threshold (float): Fixed threshold for classification.

    Returns:
        tuple: (threshold, conf_matrix, precision, recall, f1, accuracy)
    """
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    predictions = [1 if prob >= threshold else 0 for prob in predicted_probs]
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)

    return float(threshold), conf_matrix.tolist(), float(
        precision), float(recall), float(f1), float(accuracy)





def get_roc_metrics(real_preds, sample_preds):
    """
    Get ROC curve data.

    Args:
        real_preds (list or array): Predictions for real data.
        sample_preds (list or array): Predictions for sample data.

    Returns:
        tuple: (fpr_list, tpr_list, roc_auc)
    """
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    """
    Get Precision-Recall curve data.

    Args:
        real_preds (list or array): Predictions for real data.
        sample_preds (list or array): Predictions for sample data.

    Returns:
        tuple: (precision_list, recall_list, pr_auc)
    """
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)




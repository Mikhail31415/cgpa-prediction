import torch
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score, confusion_matrix)
import pandas as pd


def evaluate_model(y_true, y_pred, metrics_log=None, model_name=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"  Metrics{f' for {model_name}' if model_name else ''}:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision (w):  {prec:.4f}")
    print(f"  Recall (w):     {rec:.4f}")
    print(f"  Precision (per class): ", [f"{p:.4f}" for p in prec_per_class])
    print(f"  Recall (per class): ", [f"{p:.4f}" for p in rec_per_class])
    print(f"  F1 (macro):     {f1:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    print("\nConfusion Matrix:")
    print(cm_df)

    if metrics_log is not None:
        metrics_log.append({
            'model': model_name or 'Unnamed',
            'accuracy': acc,
            'precision_w': prec,
            'recall_w': rec,
            'precision_per_class': prec_per_class,
            'recall_per_class': rec_per_class,
            'f1_macro': f1,
            'f1_weighted': f1_weighted
        })

    return f1_weighted


def shuffle_dataset(x_dict, y):
    n = y.size(0)
    idx = torch.randperm(n)

    def shuffle_item(item):
        if isinstance(item, dict):
            return {k: shuffle_item(v) for k, v in item.items()}
        elif isinstance(item, torch.Tensor):
            return item[idx]
        else:
            raise TypeError(f"Unsupported type: {type(item)}")

    x_shuffled = shuffle_item(x_dict)
    y_shuffled = y[idx]
    return x_shuffled, y_shuffled


def move_to_device(x_dict, device):
    def move_item(item):
        if isinstance(item, dict):
            return {k: move_item(v) for k, v in item.items()}
        elif isinstance(item, torch.Tensor):
            return item.to(device)
        else:
            raise TypeError(f"Unsupported type: {type(item)}")

    return move_item(x_dict)


def evaluate_model_d(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score (macro)': f1, 'F1-score (weighted)': f1_weighted}

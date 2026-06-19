import torch
import warnings

def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

def iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)

def dice_score(tp, fp, fn, tn):
    return (2 * tp) / (2 * tp + fp + fn)

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn, beta=1):
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score

def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    elif zero_division == "ignore":
        return x[~nans]
    
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x

metric_functions = {
    "accuracy": accuracy,
    "iou": iou_score,
    "dice": dice_score,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
}

def custom_metric(tp, fp, fn, tn, metric_fn, reduction="micro", zero_division="warn"):
    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn)

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()
        
    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn)
        score = _handle_zero_division(score, zero_division)
        
    return score

def compute_metrics(results, metrics, classes, stage="train"):
    tp = torch.cat([x["tp"] for x in results])
    fp = torch.cat([x["fp"] for x in results])
    fn = torch.cat([x["fn"] for x in results])
    tn = torch.cat([x["tn"] for x in results])

    results = {}

    for metric in metrics:
        metric_fn = metric_functions[metric]

        score_global = custom_metric(tp, fp, fn, tn, metric_fn, reduction="micro")
        results[f"{metric}_{stage}"] = score_global.item() if torch.is_tensor(score_global) else score_global

        score_none = custom_metric(tp, fp, fn, tn, metric_fn, reduction="none")
        score_per_class = score_none.mean(dim=0)  
        
        if len(classes) == 2: 
            class_name = [c for c in classes if c != "background"][0]
            results[f"{metric}_{stage}_class_{class_name}"] = score_per_class.item()
        else:
            for class_idx, class_score in enumerate(score_per_class):
                class_name = classes[class_idx]
                results[f"{metric}_{stage}_class_{class_name}"] = class_score.item()
        
    return results
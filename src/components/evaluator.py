import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def make_compute_metrics(average: str = "weighted"):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average=average)
        return {"accuracy": acc, "f1": f1}
    return compute_metrics

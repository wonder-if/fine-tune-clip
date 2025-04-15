import numpy as np
from transformers import EvalPrediction
from evaluate import load

accuracy_metric = load("accuracy")


def compute_accuracy(eval_pred: EvalPrediction) -> dict:
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # 计算总体准确率
    overall_accuracy = accuracy_metric.compute(
        predictions=predictions, references=labels
    )

    # 返回结果
    return {"overall_accuracy": overall_accuracy}


def compute_per_class_accuracy(eval_pred: EvalPrediction, label_names: list) -> dict:
    """
    计算总体准确率和每个类别的准确率
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # 计算总体准确率
    overall_accuracy = accuracy_metric.compute(
        predictions=predictions, references=labels
    )
    # 计算每个类别的准确率
    class_metrics = {}
    for i, label_name in enumerate(label_names):
        class_labels = labels[labels == i]
        class_predictions = predictions[labels == i]
        if len(class_labels) > 0:
            class_accuracy = accuracy_metric.compute(
                predictions=class_predictions, references=class_labels
            )
            class_metrics[f"{label_name}_accuracy"] = class_accuracy

    # 返回结果
    return {"overall_accuracy": overall_accuracy, **class_metrics}


def compute_metrics(eval_pred: EvalPrediction, label_names: list) -> dict:
    """
    计算总体准确率和每个类别的准确率
    """
    return compute_per_class_metrics(eval_pred, label_names)

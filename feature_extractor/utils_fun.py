import sklearn.metrics


def get_evaluation_value_each_epoch(labels_true, labels_pred, class_names):
    report = sklearn.metrics.classification_report(labels_true, labels_pred, target_names=class_names, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']
    acc = sklearn.metrics.accuracy_score(labels_true, labels_pred)

    return acc, precision, recall, f1_score





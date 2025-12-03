from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas detalladas usando scikit-learn.
    y_true: Lista o array de etiquetas reales
    y_pred: Lista o array de predicciones
    """
    # Métricas Globales
    accuracy = accuracy_score(y_true, y_pred)
    
    # Métricas por clase y promedios
    # average='weighted' tiene en cuenta el desbalance de clases para el score general
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Métricas por cada clase individual (Para el reporte detallado)
    precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'per_class': {
            'bajista_0': {'p': precision_cls[0], 'r': recall_cls[0], 'f1': f1_cls[0]},
            'alcista_1': {'p': precision_cls[1], 'r': recall_cls[1], 'f1': f1_cls[1]},
            'neutral_2': {'p': precision_cls[2], 'r': recall_cls[2], 'f1': f1_cls[2]},
        }
    }
    
    return metrics
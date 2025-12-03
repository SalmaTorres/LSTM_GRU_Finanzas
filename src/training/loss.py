import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
import os
import sys

# Ajuste para importar config y data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import DATA_PATH
import src.config as config

def get_class_weights(data_path=DATA_PATH):
    """
    Calcula los pesos inversos de frecuencia para manejar el desbalance de clases.
    Fórmula: Peso = Total_Muestras / (Numero_Clases * Frecuencia_Clase)
    """
    try:
        df = pd.read_csv(data_path)
        labels = df['label'].tolist()
        counts = Counter(labels)
        
        n_samples = len(labels)
        n_classes = len(counts)
        
        # Inicializamos pesos para clases 0, 1, 2
        weights = []
        for i in range(n_classes):
            count = counts.get(i, 0)
            if count > 0:
                weight = n_samples / (n_classes * count)
            else:
                weight = 1.0 # Fallback
            weights.append(weight)
            
        print(f"Pesos de Clase Calculados: {weights}")
        return torch.tensor(weights, dtype=torch.float)
        
    except Exception as e:
        print(f"Error calculando pesos: {e}. Usando pesos iguales.")
        return torch.ones(3)

def get_loss_function(device):
    """
    Retorna la función de pérdida CrossEntropyLoss ponderada.
    """
    weights = get_class_weights()
    weights = weights.to(device)
    
    # CrossEntropyLoss ya incluye Softmax internamente
    criterion = nn.CrossEntropyLoss(weight=weights)
    return criterion
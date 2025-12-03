import torch
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import DataLoader

# --- Configuración de Rutas ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, DATA_PATH
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier
from src.models.ensemble import EnsembleModel
import src.config as config

# Directorios de salida
RESULTS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..', 'results'))
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_model(filename, model_type, use_attention, bidirectional, hidden_dim, n_layers):
    """Carga un modelo específico desde checkpoints."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(CHECKPOINT_DIR, filename)
    
    if not os.path.exists(path):
        print(f"Error: No se encontró el modelo en {path}")
        return None

    # Parámetros comunes
    params = {
        'vocab_size': len(WORD_TO_INDEX),
        'embedding_dim': config.EMBEDDING_DIM,
        'hidden_dim': hidden_dim,
        'output_dim': config.OUTPUT_DIM,
        'n_layers': n_layers,
        'dropout': config.DROPOUT,
        'use_attention': use_attention,
        'pretrained_embeddings': None # No necesario para inferencia
    }
    
    if model_type == 'lstm':
        model = LSTMClassifier(**params, bidirectional=bidirectional)
    elif model_type == 'gru':
        model = GRUClassifier(**params, bidirectional=bidirectional)
    
    # Cargar pesos
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Modelo cargado: {filename}")
    return model

def generate_detailed_report(model, dataset, output_filename):
    """
    Genera un CSV detallado con: Texto, Longitud, Real vs Predicción.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)
    
    results = []
    print(f"Generando reporte: {output_filename}...")
    
    with torch.no_grad():
        # Iteramos sobre el loader y mantenemos un índice global para recuperar el texto
        global_idx = 0
        
        for texts, lengths, labels in loader:
            texts = texts.to(device)
            labels_np = labels.cpu().numpy()
            
            # Inferencia
            preds, _ = model(texts, lengths.cpu())
            preds_cls = torch.argmax(preds, dim=1).cpu().numpy()
            probs = torch.softmax(preds, dim=1).cpu().numpy()
            
            # Guardar datos
            for i in range(len(labels)):
                # Recuperar texto original desde el dataset usando el índice global
                original_text = dataset.clean_texts[global_idx]
                
                is_correct = (preds_cls[i] == labels_np[i])
                
                results.append({
                    'original_text': original_text,
                    'length': lengths[i].item(),
                    'label_true': labels_np[i],
                    'label_pred': preds_cls[i],
                    'confidence': probs[i][preds_cls[i]],
                    'is_correct': is_correct,
                    'error_type': "None" if is_correct else f"True_{labels_np[i]}_Pred_{preds_cls[i]}"
                })
                global_idx += 1
                
    df = pd.DataFrame(results)
    save_path = os.path.join(REPORTS_DIR, output_filename)
    df.to_csv(save_path, index=False)
    print(f"Reporte guardado en: {save_path}")
    
    # Imprimir métricas rápidas
    acc = df['is_correct'].mean()
    print(f"   -> Accuracy del Reporte: {acc:.4f}")
    return df

def run_ensemble_test():
    """
    Carga el Mejor LSTM y el Mejor GRU para hacer un Ensemble.
    (Puntos Extra)
    """
    print("\n--- INICIANDO TEST DE ENSEMBLE (LSTM + GRU) ---")
    
    # 1. Definir los nombres de los archivos (AJUSTA ESTO SI TUS NOMBRES SON DISTINTOS)
    # Basado en tus logs anteriores:
    lstm_name = "best_lstm_bi_attn.pth"  # Tu ganador
    gru_name = "best_gru_uni_attn.pth"   # Tu mejor GRU
    
    # 2. Cargar modelos (Asegúrate que hidden_dim y n_layers coincidan con engine.py)
    # Según tus logs anteriores, usaste hidden=256 (estándar) y layers=2 (por defecto)
    lstm_model = load_model(lstm_name, 'lstm', use_attention=True, bidirectional=True, hidden_dim=256, n_layers=2)
    gru_model = load_model(gru_name, 'gru', use_attention=True, bidirectional=False, hidden_dim=256, n_layers=2)
    
    if lstm_model is None or gru_model is None:
        print("No se puede correr el Ensemble porque faltan modelos.")
        return

    # 3. Crear Ensemble
    ensemble = EnsembleModel(lstm_model, gru_model)
    ensemble.eval()
    
    # 4. Probar
    dataset = FinancialTweetDataset(DATA_PATH, WORD_TO_INDEX)
    generate_detailed_report(ensemble, dataset, "ensemble_results.csv")

if __name__ == "__main__":
    # 1. Generar reporte detallado de tu MEJOR MODELO INDIVIDUAL (Para análisis de errores)
    print("\n--- ANALIZANDO MEJOR MODELO INDIVIDUAL ---")
    best_model_name = "best_lstm_bi_attn.pth" # Tu ganador
    model = load_model(best_model_name, 'lstm', use_attention=True, bidirectional=True, hidden_dim=256, n_layers=2)
    
    if model:
        dataset = FinancialTweetDataset(DATA_PATH, WORD_TO_INDEX)
        generate_detailed_report(model, dataset, "best_model_detailed_analysis.csv")
    
    # 2. Correr Ensemble
    run_ensemble_test()
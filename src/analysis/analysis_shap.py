import shap
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# --- 1. Configuración de Rutas ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, DATA_PATH
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier
import src.config as config

# Directorios de salida
RESULTS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..', 'results'))
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(PLOTS_DIR, exist_ok=True)

def analyze_with_shap(model_filename, model_type='lstm', bidirectional=True, hidden_dim=256, n_layers=2):
    print(f"\n--- INICIANDO ANÁLISIS SHAP PARA: {model_filename} ---")
    
    # --- Evitar conflicto de hilos ---
    torch.set_num_threads(1)
    device = torch.device('cpu') 
    
    # 1. Cargar Datos 
    # Reducimos batch_size para consumir menos memoria
    dataset = FinancialTweetDataset(csv_file=DATA_PATH, vocab=WORD_TO_INDEX)
    background_loader = torch.utils.data.DataLoader(dataset, batch_size=20, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=collate_fn, shuffle=True)
    
    background_batch, _, _ = next(iter(background_loader))
    test_batch, _, _ = next(iter(test_loader))

    # --- FIX 2: Alineación de Dimensiones ---
    len_bg = background_batch.shape[1]
    len_test = test_batch.shape[1]
    max_len = max(len_bg, len_test)

    if len_bg < max_len:
        padding = torch.zeros((background_batch.shape[0], max_len - len_bg), dtype=torch.long)
        background_batch = torch.cat([background_batch, padding], dim=1)

    if len_test < max_len:
        padding = torch.zeros((test_batch.shape[0], max_len - len_test), dtype=torch.long)
        test_batch = torch.cat([test_batch, padding], dim=1)
        
    print(f"Dimensiones alineadas a: {max_len} palabras.")

    # En lugar de usar todos los puntos, usamos un resumen estadístico (K-Means o Sample)
    # Esto evita que la matriz de KernelExplainer explote en memoria.
    background_summary = shap.sample(background_batch.numpy(), 10) # Usamos solo 10 representantes
    print("Datos de fondo resumidos para estabilidad.")
    
    # 2. Reconstruir Modelo
    params = {
        'vocab_size': len(WORD_TO_INDEX),
        'embedding_dim': config.EMBEDDING_DIM,
        'hidden_dim': hidden_dim,
        'output_dim': config.OUTPUT_DIM,
        'n_layers': n_layers,
        'dropout': 0, 
        'use_attention': True,
        'pretrained_embeddings': None
    }
    
    if model_type == 'lstm':
        model = LSTMClassifier(**params, bidirectional=bidirectional)
    else:
        model = GRUClassifier(**params, bidirectional=bidirectional)
        
    path = os.path.join(CHECKPOINT_DIR, model_filename)
    if not os.path.exists(path):
        print(f"Modelo no encontrado: {path}")
        return

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Modelo cargado.")

    # 3. Wrapper Seguro
    def model_wrapper(x):
        # x llega como numpy array, a veces float por SHAP
        # Lo forzamos a int y a tensor contiguo en memoria
        x_int = np.round(x).astype(np.int64)
        tensor_x = torch.as_tensor(x_int, dtype=torch.long, device=device)
        
        # Calcular lengths
        lengths = (tensor_x != 0).sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)
        
        with torch.no_grad():
            output, _ = model(tensor_x, lengths)
        
        # Retornamos numpy limpio
        return output.detach().numpy()

    # 4. Calcular SHAP Values
    print("Calculando SHAP (Modo Seguro)...")
    
    try:
        # Usamos el resumen en lugar del batch completo
        explainer = shap.KernelExplainer(model_wrapper, background_summary)
        shap_values = explainer.shap_values(test_batch.numpy(), nsamples=100) # nsamples bajo para velocidad
        
        print("Generando gráfico...")
        plt.figure()
        shap.summary_plot(shap_values, test_batch.numpy(), show=False)
        
        save_path = os.path.join(PLOTS_DIR, 'shap_summary.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Gráfico SHAP guardado en: {save_path}")
        
    except Exception as e:
        print(f"Error durante el cálculo SHAP: {e}")
        print("   Intenta reducir aún más 'batch_size' en el script.")

if __name__ == "__main__":
    # Ajusta al nombre de tu mejor modelo
    model_name = 'best_lstm_bi_attn.pth' 
    analyze_with_shap(model_name, model_type='lstm', bidirectional=True, hidden_dim=256, n_layers=2)
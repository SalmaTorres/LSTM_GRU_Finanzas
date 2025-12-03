import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# --- 1. Configuración de Rutas ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

from src.data.dataset import WORD_TO_INDEX
from src.data.preprocess import clean_and_preprocess, text_to_sequence
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier
import src.config as config

# Directorios de salida
RESULTS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..', 'results'))
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 2. Función para Cargar el Modelo ---
def load_model_for_viz(model_filename, hidden_dim=256, n_layers=2):
    """
    Carga un modelo entrenado para visualización.
    Intenta deducir la configuración del nombre del archivo, 
    pero usa hidden_dim y n_layers por defecto del 'Torneo'.
    """
    device = torch.device('cpu') # Visualización es mejor en CPU
    path = os.path.join(CHECKPOINT_DIR, model_filename)
    
    if not os.path.exists(path):
        print(f"Error: No se encontró el modelo en {path}")
        print("   Asegúrate de haber corrido engine.py primero.")
        return None

    # Deducir configuración del nombre (si seguiste mi engine.py)
    model_type = 'lstm' if 'lstm' in model_filename else 'gru'
    bidirectional = 'bi' in model_filename
    use_attention = 'attn' in model_filename # Si dice 'noattn', esto fallaría, pero queremos visualizar atención así que asumimos que cargamos uno con atención.
    
    print(f"Cargando {model_type.upper()} (Bi={bidirectional}, Attn={use_attention})...")

    # Inicializar arquitectura
    params = {
        'vocab_size': len(WORD_TO_INDEX),
        'embedding_dim': config.EMBEDDING_DIM,
        'hidden_dim': hidden_dim,
        'output_dim': config.OUTPUT_DIM,
        'n_layers': n_layers,
        'dropout': 0, # Dropout apagado en inferencia
        'use_attention': True, # Forzamos True porque este script es PARA visualizar atención
        'pretrained_embeddings': None
    }
    
    if model_type == 'lstm':
        model = LSTMClassifier(**params, bidirectional=bidirectional)
    else:
        model = GRUClassifier(**params, bidirectional=bidirectional)
        
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"Error cargando pesos: {e}")
        return None

# --- 3. Función Principal de Visualización ---
def predict_and_visualize(text, model, save_name=None):
    """
    Toma un texto crudo, predice y genera el heatmap de atención.
    """
    # A. Preprocesamiento
    clean_text = clean_and_preprocess(text)
    seq = text_to_sequence(clean_text, WORD_TO_INDEX)
    
    # Manejo de secuencia vacía
    if len(seq) == 0:
        print(f"El texto '{text}' quedó vacío tras la limpieza.")
        return

    # Convertir a tensor
    tensor_seq = torch.tensor([seq], dtype=torch.long)
    length = torch.tensor([len(seq)], dtype=torch.long)
    
    # B. Inferencia
    with torch.no_grad():
        preds, attn_weights = model(tensor_seq, length)
        probs = torch.softmax(preds, dim=1)
        pred_class = torch.argmax(probs).item()
    
    # Etiquetas
    classes = ['Bajista', 'Alcista', 'Neutral']
    confidence = probs[0][pred_class].item()
    
    print(f"\nTexto Original: {text}")
    print(f"Texto Limpio:   {clean_text}")
    print(f"Predicción:     {classes[pred_class]} ({confidence:.2%})")
    
    # C. Generar Heatmap
    if attn_weights is not None:
        words = clean_text.split()
        # attn_weights shape: [1, seq_len] -> [seq_len]
        weights = attn_weights[0].numpy()
        
        # Crear figura
        plt.figure(figsize=(10, 3))
        sns.heatmap([weights], xticklabels=words, yticklabels=['Importancia'], 
                    cmap='YlOrRd', cbar=True, annot=True, fmt=".2f",
                    linewidths=0.5, square=False)
        
        title = f"Pred: {classes[pred_class]} | '{text[:30]}...'"
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Guardar o Mostrar
        if save_name:
            save_path = os.path.join(PLOTS_DIR, save_name)
            plt.savefig(save_path)
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()
        
        plt.close()
    else:
        print("Este modelo no devolvió pesos de atención (¿cargaste un modelo 'noattn'?)")

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # Asegúrate de poner aquí el nombre EXACTO del archivo generado por engine.py
    # Basado en tus logs anteriores, tu ganador fue este:
    BEST_MODEL_FILENAME = "best_lstm_bi_attn.pth" 
    
    # Cargar el modelo una sola vez
    model = load_model_for_viz(BEST_MODEL_FILENAME, hidden_dim=256, n_layers=2)
    
    if model:
        print("\n--- GENERANDO VISUALIZACIONES PARA REPORTE ---")
        
        # CASO 1: SARCASMO (Difícil)
        # "Genial trabajo Fed" suele ser positivo, pero aquí es sarcasmo bajista
        predict_and_visualize(
            "Great job Fed, another rate hike is exactly what we needed!", 
            model, 
            save_name="attn_sarcasm.png"
        )
        
        # CASO 2: NEGACIÓN (Contexto)
        # "Not" invierte el significado de "crash"
        predict_and_visualize(
            "The market is definitely not going to crash today.", 
            model, 
            save_name="attn_negation.png"
        )
        
        # CASO 3: JERGA FINANCIERA (Alcista)
        # Debe enfocarse en "longing" y "bullish"
        predict_and_visualize(
            "I am longing BTC because the chart looks super bullish.", 
            model, 
            save_name="attn_jargon.png"
        )
        
        # CASO 4: INCERTIDUMBRE (Neutral)
        predict_and_visualize(
            "Watching the charts, price is moving sideways.", 
            model, 
            save_name="attn_neutral.png"
        )
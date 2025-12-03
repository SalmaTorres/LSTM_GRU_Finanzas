import torch
import torch.nn.functional as F
import os
import sys

# --- Configuración de Rutas ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from src.data.dataset import WORD_TO_INDEX, PAD_INDEX
from src.data.preprocess import clean_and_preprocess, text_to_sequence
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier
import src.config as config

# --- CONFIGURA AQUÍ EL NOMBRE DE TU MEJOR MODELO ---
# (Revisa la carpeta results/checkpoints para ver el nombre exacto)
BEST_MODEL_NAME = "best_lstm_bi_attn.pth" 

def load_inference_model():
    print("Cargando el cerebro del modelo...")
    device = torch.device('cpu') # Para inferencia de un solo texto, CPU es instantáneo
    
    path = os.path.join(current_dir, '..', 'results', 'checkpoints', BEST_MODEL_NAME)
    if not os.path.exists(path):
        print(f"Error: No encuentro el modelo en {path}")
        return None

    # Deducir configuración del nombre del archivo
    model_type = 'lstm' if 'lstm' in BEST_MODEL_NAME else 'gru'
    bidirectional = 'bi' in BEST_MODEL_NAME
    # Asumimos configuración del torneo (ajústalo si ganaste con otros hiperparámetros)
    hidden_dim = 256 
    n_layers = 2
    
    # Inicializar arquitectura
    params = {
        'vocab_size': len(WORD_TO_INDEX),
        'embedding_dim': config.EMBEDDING_DIM,
        'hidden_dim': hidden_dim,
        'output_dim': config.OUTPUT_DIM,
        'n_layers': n_layers,
        'dropout': 0, # Sin dropout en inferencia
        'use_attention': True,
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
        print("¡Modelo cargado y listo!")
        return model
    except Exception as e:
        print(f"El modelo no coincide con la arquitectura: {e}")
        return None

def predict_live(model):
    print("\n" + "="*50)
    print("SISTEMA DE ANÁLISIS DE SENTIMIENTO FINANCIERO")
    print(" Escribe 'salir' para terminar.")
    print("="*50)
    
    classes = ['Bajista (Bearish)', 'Alcista (Bullish)', 'Neutral']
    
    while True:
        raw_text = input("\nEscribe un tweet financiero: ")
        
        if raw_text.lower() in ['salir', 'exit', 'quit']:
            print("¡Hasta luego!")
            break
            
        if not raw_text.strip():
            continue

        # 1. Preprocesar
        clean_text = clean_and_preprocess(raw_text)
        print(f"   (Limpio: '{clean_text}')")
        
        # 2. Convertir a índices
        seq = text_to_sequence(clean_text, WORD_TO_INDEX)
        if len(seq) == 0:
            print("El texto no contiene palabras conocidas.")
            continue
            
        tensor_seq = torch.tensor([seq], dtype=torch.long)
        length = torch.tensor([len(seq)], dtype=torch.long)
        
        # 3. Predecir
        with torch.no_grad():
            preds, attn = model(tensor_seq, length)
            probs = F.softmax(preds, dim=1).squeeze()
            
        # 4. Mostrar Resultados
        best_class = torch.argmax(probs).item()
        confidence = probs[best_class].item()
        
        print(f"\n   RESULTADO: {classes[best_class]}")
        print(f"   CONFIANZA: {confidence:.2%}")
        
        # Barra de progreso visual
        print("\n   Distribución:")
        for i, class_name in enumerate(classes):
            bar_len = int(probs[i] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {class_name.split()[0]:<10} {bar} {probs[i]:.4f}")

if __name__ == "__main__":
    model = load_inference_model()
    if model:
        predict_live(model)
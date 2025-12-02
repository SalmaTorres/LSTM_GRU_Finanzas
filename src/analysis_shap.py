import shap
import torch
import numpy as np
import sys
import os

# Ajuste de path para importar módulos hermanos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, DATA_PATH
from src.models.lstm_model import LSTMClassifier
import src.config as config

def analyze_with_shap(model_path):
    print("--- Iniciando Análisis SHAP ---")
    
    # Cargar datos de fondo y prueba
    dataset = FinancialTweetDataset(csv_file=DATA_PATH, vocab=WORD_TO_INDEX)
    background_loader = torch.utils.data.DataLoader(dataset, batch_size=50, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=collate_fn, shuffle=True)
    
    background_batch, _, _ = next(iter(background_loader))
    test_batch, _, _ = next(iter(test_loader))
    
    # Reconstruir modelo
    model = LSTMClassifier(
        vocab_size=len(WORD_TO_INDEX),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        n_layers=config.NUM_LAYERS,
        bidirectional=True,
        dropout=config.DROPOUT
    )
    
    # Cargar pesos entrenados
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Modelo cargado.")
    else:
        print("⚠️ Advertencia: No se encontró el modelo, usando pesos aleatorios para prueba.")

    model.eval()

    # Wrapper para SHAP
    def model_wrapper(x):
        lengths = (torch.tensor(x) != 0).sum(dim=1).cpu()
        with torch.no_grad():
            output, _ = model(torch.tensor(x, dtype=torch.long), lengths)
        return output.numpy()

    # Calcular SHAP
    explainer = shap.KernelExplainer(model_wrapper, background_batch.numpy())
    shap_values = explainer.shap_values(test_batch.numpy())
    
    # Visualizar
    shap.summary_plot(shap_values, test_batch.numpy())

if __name__ == "__main__":
    # Cambia esto por la ruta de tu modelo guardado cuando entrenes
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_lstm_model.pth')
    analyze_with_shap(model_path)
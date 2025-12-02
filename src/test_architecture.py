import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# Ajuste de path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, DATA_PATH
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier
import src.config as config
from src.models.ensemble import EnsembleModel

def run_sanity_checks():
    print("\n=======================================================")
    print(" VALIDACIÓN DE ARQUITECTURAS PARA EXPERIMENTOS")
    print("=======================================================")

    # 1. Cargar Datos
    try:
        vocab_size = len(WORD_TO_INDEX)
        dataset = FinancialTweetDataset(csv_file=DATA_PATH, vocab=WORD_TO_INDEX)
        loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
        batch_text, batch_lengths, _ = next(iter(loader))
        print(f"✅ Datos cargados. Batch: {batch_text.shape}")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    # -------------------------------------------------------------
    # PRUEBA A: LSTM AVANZADA (Bidireccional + Atención + Word2Vec)
    # -------------------------------------------------------------
    print("\n[A] Probando: LSTM Bidireccional + Atención + Word2Vec Simulado")
    try:
        # Simulamos una matriz de Word2Vec
        dummy_w2v = np.random.rand(vocab_size, config.EMBEDDING_DIM)
        
        model_a = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=2,                 # Deep RNN
            bidirectional=True,         # Bidireccional
            dropout=0.5,
            use_attention=True,         # CON Atención
            pretrained_embeddings=dummy_w2v
        )
        
        preds, attn = model_a(batch_text, batch_lengths)
        print(f"  -> Forward exitoso.")
        print(f"  -> Predicciones: {preds.shape} (Esperado [16, 3])")
        print(f"  -> Pesos Atención: {attn.shape} (Esperado [16, {batch_text.shape[1]}])")
        print("  ✅ PRUEBA A SUPERADA")
    except Exception as e:
        print(f"  ❌ FALLO PRUEBA A: {e}")

    # -------------------------------------------------------------
    # PRUEBA B: GRU (Unidireccional + Atención)
    # -------------------------------------------------------------
    print("\n[B] Probando: GRU Con Atención")
    try:
        model_b = GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=2,
            bidirectional=False,        # Unidireccional
            dropout=0.5,
            use_attention=True,
            pretrained_embeddings=None  # Sin W2V
        )
        preds, _ = model_b(batch_text, batch_lengths)
        print("  ✅ PRUEBA B SUPERADA")
    except Exception as e:
        print(f"  ❌ FALLO PRUEBA B: {e}")

    # -------------------------------------------------------------
    # PRUEBA C: LSTM SIMPLE (Sin Atención - Para Exp 8.1)
    # -------------------------------------------------------------
    print("\n[C] Probando: LSTM Sin Atención")
    try:
        model_c = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=1,
            bidirectional=False,
            dropout=0.0,
            use_attention=False,         # SIN Atención
            pretrained_embeddings=None
        )
        preds, attn = model_c(batch_text, batch_lengths)
        
        if attn is None:
            print(f"  -> Atención desactivada correctamente (es None).")
            print("  ✅ PRUEBA C SUPERADA")
        else:
            print("  ❌ FALLO PRUEBA C: Atención no debería devolver valores.")
            
    except Exception as e:
        print(f"  ❌ FALLO PRUEBA C: {e}")

    # -------------------------------------------------------------
    # PRUEBA D: GRU SIN ATENCIÓN
    # -------------------------------------------------------------
    print("\n[D] Probando: GRU Sin Atención")
    try:
        model_d = GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=1,
            bidirectional=False,
            dropout=0.0,
            use_attention=False,
            pretrained_embeddings=None
        )
        preds, _ = model_d(batch_text, batch_lengths)
        print("  ✅ PRUEBA D SUPERADA")
    except Exception as e:
        print(f"  ❌ FALLO PRUEBA D: {e}")

    # -------------------------------------------------------------
    # PRUEBA E: ENSEMBLE (LSTM + GRU) - Puntos Extra
    # -------------------------------------------------------------
    print("\n[E] Probando: Ensemble (LSTM + GRU) Puntos Extra")
    try:
        # 1. Instanciamos dos modelos "tontos" (sin entrenar)
        # Nota: En la vida real, cargarías pesos entrenados aquí con load_state_dict
        dummy_lstm = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=2,
            bidirectional=True,
            dropout=0.5
        )
        
        dummy_gru = GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=2,
            bidirectional=False, # GRU puede ser unidireccional para variar
            dropout=0.5
        )
        
        # 2. Creamos el Ensemble
        ensemble_model = EnsembleModel(dummy_lstm, dummy_gru)
        
        # 3. Verificamos congelamiento de gradientes
        # (Si funciona, requires_grad debe ser False)
        param_ejemplo = next(ensemble_model.lstm.parameters())
        if param_ejemplo.requires_grad == False:
            print("  -> ✅ Los sub-modelos se congelaron correctamente.")
        else:
            print("  ❌ ERROR: Los sub-modelos siguen entrenables.")

        # 4. Forward Pass del Ensemble
        ensemble_probs = ensemble_model(batch_text, batch_lengths)
        
        print(f"  -> Ensemble Output Shape: {ensemble_probs.shape} (Esperado [16, 3])")
        
        # Verificamos que sean probabilidades (suman 1)
        suma_probs = ensemble_probs[0].sum().item()
        print(f"  -> Suma de probabilidades (ejemplo): {suma_probs:.4f} (Debe ser aprox 1.0)")
        
        if 0.99 < suma_probs < 1.01:
             print("  ✅ PRUEBA E SUPERADA: El Ensemble promedia probabilidades correctamente.")
        else:
             print("  ⚠️ ALERTA: La salida no parece una distribución de probabilidad (Softmax).")

    except Exception as e:
        print(f"  ❌ FALLO PRUEBA E: {e}")
        import traceback
        traceback.print_exc()

    print("\n=======================================================")
    print(" CONCLUSIÓN: Tus modelos están listos para la batalla")
    print(" Ahora puedes crear 'train.py' para encontrar el mejor.")
    print("=======================================================")

if __name__ == "__main__":
    run_sanity_checks()

    
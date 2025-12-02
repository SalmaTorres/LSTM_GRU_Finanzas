# --- Hiperparámetros del Modelo ---
EMBEDDING_DIM = 300      # 100, 200, 300 (Recomendado 300 para finanzas)
HIDDEN_DIM = 256         # 128, 256, 512 (Memoria de la red)
NUM_LAYERS = 2           # >1 para cumplir "Deep RNN" (ej. 2 o 3)
DROPOUT = 0.5            # 0.3 a 0.5 para regularización (evitar overfitting)
OUTPUT_DIM = 3           # Clases: 0 (Bajista), 1 (Alcista), 2 (Neutral)

# --- Entrenamiento ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

# --- Rutas ---
MODEL_SAVE_PATH = 'models/'
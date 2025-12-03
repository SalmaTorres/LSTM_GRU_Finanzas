import os

# --- Hiperparámetros del Modelo ---
EMBEDDING_DIM = 300      # 100, 200, 300 (Recomendado 300 para finanzas)
HIDDEN_DIM = 256         # 128, 256, 512 (Memoria de la red)
NUM_LAYERS = 2           # >1 para cumplir "Deep RNN" (ej. 2 o 3)
DROPOUT = 0.5            # 0.3 a 0.5 para regularización (evitar overfitting)
OUTPUT_DIM = 3           # Clases: 0 (Bajista), 1 (Alcista), 2 (Neutral)

# --- Entrenamiento ---
BATCH_SIZE = 32          # 32 o 64 suelen ser estables
LEARNING_RATE = 0.001    # Punto de partida estándar para Adam
EPOCHS = 15              # Con Early Stopping, pon esto alto (ej. 15 o 20)

# --- Rutas ---
MODEL_SAVE_PATH = 'models/'

base_dir = os.path.dirname(os.path.abspath(__file__)) # Raíz del proyecto
RESULTS_DIR = os.path.join(base_dir, 'results')

# Rutas específicas
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'checkpoints')
LOGS_PATH = os.path.join(RESULTS_DIR, 'logs')
PLOTS_PATH = os.path.join(RESULTS_DIR, 'plots')

# Asegurar que existan
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

import pandas as pd
import sys
import os
import torch
from torch.utils.data import DataLoader

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Ajuste de PATH para importar desde la raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

# RUTA DEL ARCHIVO DE DATOS (Relativa a la raíz del proyecto)
DATA_PATH = 'data/sent_train.csv' 

try:
    # Importaciones Absolutas
    from src.data.preprocess import clean_and_preprocess, create_vocabulary
    from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, PAD_INDEX, DATA_PATH as DATA_PATH_ABS 
except ImportError as e:
    print(f"Error de Importación: {e}")
    print("Asegúrate de que los archivos están en 'src/data/' y que el script se ejecuta desde la raíz o con el path correcto.")
    sys.exit(1)

# --- 2. PREPARACIÓN: Carga y Creación de Vocabulario ---

print("--- 1. Preparación: Carga y Creación de Vocabulario ---")

try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except FileNotFoundError:
    print(f"Error: Archivo de datos no encontrado en {DATA_PATH}")
    sys.exit(1)

df['cleaned_text'] = df['text'].apply(clean_and_preprocess)
word_to_index, _, vocab_size = create_vocabulary(df['cleaned_text'].tolist())

print("Preparación completada: Vocabulario construido.")
print("--------------------------------------------------")

# --- 3. PRUEBA DE LIMPIEZA Y EXPANSIÓN DE JERGA (Criterio 2) ---

print("--- 2. Prueba de Limpieza y Expansión de Jerga ---") 

test_text = "I'm feeling FOMO and HODL my $BTC. ATH is next! Check out https://t.co/url @user #crypto"
cleaned = clean_and_preprocess(test_text)

print(f"Texto Original: {test_text}")
print(f"Texto Limpio:   {cleaned}")

# Resultado esperado tras expansión de I'm -> i am, y protección de <TICKER>
expected_start = "i am feeling fear of missing out and hold on for dear life my <TICKER> all time high is next check out crypto"

if cleaned == expected_start:
    print(" Prueba de Limpieza Avanzada (Contracciones/Jerga/Tickers): ¡Correcta!")
else:
    print(f" Prueba Fallida. El resultado fue: {cleaned}")
    print(f" Prueba Fallida. Se esperaba: {expected_start}")
    
print("--------------------------------------------------")

# --- 4. PRUEBA DE DATASET Y DATALOADER (Manejo de Secuencias Variables) ---

print("--- 3. Prueba de Dataset y DataLoader (collate_fn) ---")

dataset = FinancialTweetDataset(csv_file=DATA_PATH_ABS, vocab=WORD_TO_INDEX) 
data_loader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=False, 
    collate_fn=collate_fn
)

for padded_batch, lengths, labels in data_loader:
    # Verificación de Ordenamiento (CRÍTICO)
    is_descending = all(lengths[i] >= lengths[i+1] for i in range(len(lengths) - 1))
    
    # Verificación de Padding (CRÍTICO)
    # Se espera que el padding use el índice 0 (<PAD>)
    last_seq_len = lengths[-1].item()
    padding_ok = torch.all(padded_batch[-1, last_seq_len:] == PAD_INDEX)

    print(f"Longitudes del Batch (deben ser descendentes): {lengths.tolist()}")
    
    if is_descending and padding_ok:
        print(" Prueba de collate_fn: ¡Correcta! Ordenamiento y Padding Dinámico funcionan.")
    else:
        print(" Prueba de collate_fn: ¡FALLIDA! Revisa ordenamiento o el índice de padding.")
        
    break # Solo probamos el primer batch

print("--------------------------------------------------")
print("Pruebas de Preprocesamiento Finalizadas. Criterio 2: ¡Completo!")
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import sys

# Ajuste temporal para la importación local si es necesario.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importa funciones de la Fase 1
from preprocess import clean_and_preprocess, text_to_sequence, create_vocabulary


# ==============================================================================
# 1. Inicialización y Vocabulario Estático
# ==============================================================================

# RUTA DEL ARCHIVO DE DATOS (Relativa a la ubicación del script)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sent_train.csv')

def get_vocab(data_path):
    """Carga los datos, los limpia y construye el vocabulario estático (simulación de carga)."""
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except FileNotFoundError:
        # Fallback simple si falla la ruta relativa compleja
        if os.path.exists('data/sent_train.csv'):
            df = pd.read_csv('data/sent_train.csv', encoding='utf-8')
        else:
            raise FileNotFoundError(f"No se encuentra el archivo en {data_path}")
        
    df['cleaned_text'] = df['text'].apply(clean_and_preprocess)
    # Crear vocabulario
    word_to_index, _, vocab_size = create_vocabulary(df['cleaned_text'].tolist())
    return word_to_index, vocab_size

# Cargar el vocabulario una vez (Constante para la arquitectura)
try:
    WORD_TO_INDEX, VOCAB_SIZE = get_vocab(DATA_PATH)
except Exception as e:
    print(f"Advertencia cargando vocabulario: {e}")
    WORD_TO_INDEX = {'<PAD>': 0, '<UNK>': 1}
    VOCAB_SIZE = 2

PAD_INDEX = WORD_TO_INDEX.get('<PAD>', 0)


# ==============================================================================
# 2. Clase FinancialTweetDataset (Fase 2, Paso 3)
# ==============================================================================

class FinancialTweetDataset(Dataset):
    """
    Clase PyTorch Dataset que convierte tweets en secuencias de índices numéricos.
    """
    def __init__(self, csv_file, vocab):
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
             # Intento de ruta alternativa si se ejecuta desde raíz
             if os.path.exists('data/sent_train.csv'):
                 df = pd.read_csv('data/sent_train.csv')
             else:
                 raise

        self.vocab = vocab

        # 1. Limpiar textos
        print("Dataset: Limpiando textos...")
        df['cleaned_text'] = df['text'].apply(clean_and_preprocess)
        
        # 2. FILTRADO CRÍTICO: Eliminar textos que quedaron vacíos
        # Si 'cleaned_text' es solo espacios o vacío, su longitud split() será 0.
        initial_len = len(df)
        df = df[df['cleaned_text'].str.strip().astype(bool)]
        final_len = len(df)
        
        if initial_len != final_len:
            print(f"Se eliminaron {initial_len - final_len} tweets vacíos o corruptos.")
        
        self.clean_texts = df['cleaned_text'].tolist()
        self.labels = df['label'].tolist()
        
    def __len__(self):
        return len(self.clean_texts)

    def __getitem__(self, idx):
        # Obtener texto limpio y etiqueta
        text = self.clean_texts[idx]
        label = self.labels[idx]
        
        # Convertir a secuencia de índices numéricos (usando UNK si es necesario)
        sequence = text_to_sequence(text, self.vocab)
        
        # Salvaguarda final: si por alguna razón sigue vacío, poner <UNK>
        if len(sequence) == 0:
            sequence = [self.vocab.get('<UNK>', 1)]

        # Devolver la secuencia como un tensor y la etiqueta
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 3. Función collate_fn (Fase 2, Paso 4: Manejo de Secuencias Variables)
# ==============================================================================

def collate_fn(batch):
    """
    Función de colación personalizada para el DataLoader.
    CRÍTICA para RNNs: Ordena las secuencias y aplica padding dinámico.
    """
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    
    # 1. Ordenar el Batch por longitud descendente (Requisito de pack_padded_sequence)
    sorted_batch = sorted(zip(lengths, sequences, labels), key=lambda x: x[0], reverse=True)
    
    # Re-obtener los elementos ordenados
    sorted_lengths = torch.tensor([item[0] for item in sorted_batch], dtype=torch.long)
    sorted_sequences = [item[1] for item in sorted_batch]
    sorted_labels = torch.stack([item[2] for item in sorted_batch])
    
    # 2. Realizar el padding dinámico (solo hasta la longitud máxima de este batch)
    max_len = sorted_lengths[0].item()
    # Usamos PAD_INDEX (0) para rellenar
    padded_sequences = torch.full((len(sorted_sequences), max_len), PAD_INDEX, dtype=torch.long)
    
    for i, seq in enumerate(sorted_sequences):
        padded_sequences[i, :len(seq)] = seq 
        
    # 3. Retorno: Las longitudes son fundamentales para la siguiente fase (engine.py)
    return padded_sequences, sorted_lengths, sorted_labels

# ==============================================================================
# 4. Código de Prueba
# ==============================================================================
if __name__ == '__main__':
    print("Módulo de Dataset cargado.")
    # (El código de prueba funcional se encuentra en 'prueba.py')
import pandas as pd
import os
import sys
import torch
import numpy as np
from gensim.models import Word2Vec

# Ajuste de rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from src.data.dataset import DATA_PATH, WORD_TO_INDEX, PAD_INDEX
from src.data.preprocess import clean_and_preprocess
import src.config as config

def train_and_save_word2vec():
    print("--- Entrenando Word2Vec desde cero en tus datos ---")
    
    # 1. Cargar Datos
    df = pd.read_csv(DATA_PATH)
    # Asegurar que aplicamos la misma limpieza que en el dataset
    print("Limpiando textos...")
    cleaned_texts = [clean_and_preprocess(t).split() for t in df['text']]
    
    # 2. Configurar y Entrenar Word2Vec
    # vector_size debe coincidir con EMBEDDING_DIM de tu config.py
    print(f"Entrenando modelo Word2Vec (Dim={config.EMBEDDING_DIM})...")
    w2v_model = Word2Vec(
        sentences=cleaned_texts,
        vector_size=config.EMBEDDING_DIM,
        window=5,       # Contexto (palabras vecinas)
        min_count=1,    # Incluir todas las palabras
        workers=4,
        sg=1,           # 1=Skip-Gram (mejor para datasets pequeños/jerga), 0=CBOW
        epochs=10
    )
    
    # 3. Crear Matriz de Pesos para PyTorch
    # Necesitamos una matriz de forma [VOCAB_SIZE, EMBEDDING_DIM]
    # donde la fila 'i' corresponde al vector de la palabra con índice 'i'.
    
    vocab_size = len(WORD_TO_INDEX)
    embedding_matrix = np.zeros((vocab_size, config.EMBEDDING_DIM))
    
    hits = 0
    misses = 0
    
    for word, index in WORD_TO_INDEX.items():
        if index == PAD_INDEX:
            continue # Dejar en ceros (padding)
            
        if word in w2v_model.wv:
            embedding_matrix[index] = w2v_model.wv[word]
            hits += 1
        else:
            # Si la palabra no está en Word2Vec (raro si entrenamos con los mismos datos),
            # inicializamos aleatoriamente
            embedding_matrix[index] = np.random.normal(scale=0.6, size=(config.EMBEDDING_DIM, ))
            misses += 1
            
    print(f"Word2Vec completado. Palabras encontradas: {hits}, Faltantes: {misses}")
    
    # 4. Guardar Matriz
    output_path = os.path.join(os.path.dirname(DATA_PATH), 'word2vec_matrix.npy')
    np.save(output_path, embedding_matrix)
    print(f"Matriz de embeddings guardada en: {output_path}")
    return output_path

if __name__ == '__main__':
    train_and_save_word2vec()
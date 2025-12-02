import re
from collections import Counter
import pandas as pd
import os

# ==============================================================================
# 1. Definiciones de Reglas y Jerga (Criterio 2: Preprocesamiento Avanzado)
# ==============================================================================

# 1.1. Expansión de Jerga Financiera (Slang)
JARGON_DICT = {
    'fomo': 'fear of missing out',
    'hodl': 'hold on for dear life',
    'fud': 'fear uncertainty doubt',
    'ath': 'all time high',
    'bearish': 'bearish', # Se mantienen como palabras, pero se listan
    'bullish': 'bullish',
    'lol': 'laughing out loud', # Abreviaturas comunes
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
}

# 1.2. Expansión de Contracción (Manejo de I'm, isn't, etc.)
CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "can not",
    "wouldn't": "would not",
    "couldn't": "could not",
}

def expand_jargon(text):
    """Expande la jerga financiera y abreviaturas."""
    words = text.split()
    expanded_words = [JARGON_DICT.get(word.lower(), word) for word in words]
    return ' '.join(expanded_words)

# ==============================================================================
# 2. Funciones de Limpieza
# ==============================================================================

def clean_and_preprocess(text):
    """
    Implementa la limpieza robusta: expansión de contracciones, tickers y ruido.
    """
    # 1. Normalización a minúsculas
    text = text.lower()

    # 2. Expansión de Contracción (maneja I'm -> i am)
    for contraction, expansion in CONTRACTIONS.items():
        # Usa \b para asegurar que solo se reemplaza la palabra completa
        text = re.sub(r'\b{}\b'.format(re.escape(contraction)), expansion, text)

    # 3. Expansión de Jerga (maneja FOMO -> fear of missing out)
    text = expand_jargon(text)

    # 4. Manejo de Tickers: Sustituir $TICKER por el token especial <TICKER>
    text = re.sub(r'\$[a-zA-Z]{1,5}', '<TICKER>', text) 
    
    # 5. Limpieza de Ruido de Redes Sociales
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URLs
    text = re.sub(r'@\w+', '', text)                  # Handles (@usuario)
    text = re.sub(r'#', '', text)                     # Eliminar símbolo #
    # Eliminar Emojis (simplificado)
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F7FF]+', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 6. Remover Puntuación y Caracteres Especiales
    # CRÍTICO: Reemplazamos puntuación con un espacio, pero EXCLUIMOS < y >
    # para PROTECCIÓN del token <TICKER>.
    text = re.sub(r'[^\w\s\<\>]', ' ', text)
    
    # 7. Remover espacios múltiples y trim final
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ==============================================================================
# 3. Creación de Vocabulario (Criterio 2: Prerrequisito de Embeddings)
# ==============================================================================

def create_vocabulary(texts):
    """
    Crea el vocabulario estático a partir de los textos limpios del dataset.
    Debe incluir los tokens especiales <PAD> y <UNK>.
    """
    # Tokens Especiales
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    word_counts = Counter(all_words)
    
    # 1. Asignar índices (estático y fijo)
    word_to_index = {
        PAD_TOKEN: 0, # Índice 0 para padding
        UNK_TOKEN: 1  # Índice 1 para palabras desconocidas
    }
    
    # 2. Asignar índices al resto de las palabras del dataset
    for word, count in word_counts.items():
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
            
    vocab_size = len(word_to_index)
    
    index_to_word = {i: w for w, i in word_to_index.items()}
    
    return word_to_index, index_to_word, vocab_size

def text_to_sequence(text, word_to_index):
    """Convierte un texto limpio a una secuencia de índices numéricos."""
    UNK_INDEX = word_to_index.get('<UNK>', 1)
    sequence = [word_to_index.get(word, UNK_INDEX) for word in text.split()]
    return sequence

# ==============================================================================
# 4. Código de Prueba (Se ejecuta solo si es el archivo principal)
# ==============================================================================
if __name__ == '__main__':
    # Esta sección debería ser probada usando 'prueba.py'
    print("Módulo de preprocesamiento cargado.")
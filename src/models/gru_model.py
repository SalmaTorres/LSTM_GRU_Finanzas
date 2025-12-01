import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")


# =============================================================================
# 1. DATASET SIMULADO - Reseñas de películas (positivas/negativas)
# =============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convertir texto a índices numéricos
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]

        # Padding o truncamiento
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# =============================================================================
# 2. MODELO GRU - Explicación detallada
# =============================================================================

class GRUClassifier(nn.Module):
    """
    GRU (Gated Recurrent Unit) - Versión simplificada de LSTM
    Componentes principales:
    - Reset Gate (r_t): Decide qué información olvidar del estado anterior
    - Update Gate (z_t): Decide qué información mantener del estado anterior y nueva
    - Estado oculto (h_t): Representación comprimida de la secuencia
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(GRUClassifier, self).__init__()

        # Capa de embedding: convierte palabras en vectores densos
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Capa GRU - ¡EL CORAZÓN DEL MODELO!
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )

        """
        Parámetros de GRU:
        - input_size: dimensión de los embeddings
        - hidden_size: tamaño del estado oculto (memoria de la red)
        - num_layers: número de capas GRU apiladas
        - batch_first: True = (batch, seq, features) en lugar de (seq, batch, features)
        - dropout: regularización entre capas (solo si n_layers > 1)
        """

        # Capas fully connected para clasificación
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """
        Forward pass paso a paso:

        text shape: (batch_size, seq_length)
        """
        # 1. Embedding: palabras -> vectores
        embedded = self.embedding(text)  # (batch, seq, embedding_dim)

        # 2. GRU: procesamiento secuencial
        gru_output, hidden = self.gru(embedded)
        """
        gru_output: (batch, seq, hidden_dim) - salidas en cada paso de tiempo
        hidden: (num_layers, batch, hidden_dim) - estado oculto final

        La GRU procesa la secuencia manteniendo un estado interno que
        captura dependencias a largo plazo de manera más eficiente que RNN simple
        """

        # 3. Tomamos el último estado oculto como representación de la secuencia
        last_hidden = hidden[-1]  # (batch, hidden_dim) - última capa

        # 4. Clasificación
        output = self.fc(self.dropout(last_hidden))  # (batch, output_dim)

        return output


# =============================================================================
# 3. PREPARACIÓN DE DATOS
# =============================================================================

# Datos de ejemplo - reseñas de películas
texts = [
    "esta película es excelente me encantó",
    "odio esta película es terrible",
    "muy buena actuación y guión",
    "pérdida de tiempo aburrida",
    "recomendada para todos",
    "no vale la pena verla",
    "increíble cinematografía",
    "horrible dirección y actuación",
    "me gustó mucho la trama",
    "peor película del año"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positivo, 0: negativo


# Construcción del vocabulario
def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


vocab = build_vocab(texts)
vocab_size = len(vocab)
print(f"Tamaño del vocabulario: {vocab_size}")
print(f"Vocabulario: {vocab}")

# Parámetros
max_length = 10
embedding_dim = 100
hidden_dim = 128
output_dim = 2  # positivo/negativo
n_layers = 2
dropout = 0.3
batch_size = 2
learning_rate = 0.001
epochs = 50

# Crear datasets y dataloaders
dataset = TextDataset(texts, labels, vocab, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO
# =============================================================================

def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping para evitar exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Estadísticas
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Época [{epoch + 1}/{epochs}], Pérdida: {avg_loss:.4f}, Exactitud: {accuracy:.2f}%')

    return losses, accuracies


# =============================================================================
# 5. EVALUACIÓN Y PREDICCIÓN
# =============================================================================

def predict_sentiment(model, text, vocab, max_length):
    model.eval()

    # Preprocesar texto
    words = text.split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]

    # Padding
    if len(indices) < max_length:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    # Convertir a tensor
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()

    sentiment = "POSITIVO" if predicted_class == 1 else "NEGATIVO"
    confidence = probabilities[0][predicted_class].item()

    return sentiment, confidence


# =============================================================================
# 6. EJECUCIÓN COMPLETA
# =============================================================================

def main():
    # Crear modelo
    model = GRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    print("\n" + "=" * 50)
    print("ARQUITECTURA DEL MODELO GRU")
    print("=" * 50)
    print(f"Embedding dim: {embedding_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"GRU layers: {n_layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    print("\nINICIANDO ENTRENAMIENTO...")
    losses, accuracies = train_model(model, dataloader, criterion, optimizer, epochs)

    # Visualizar resultados del entrenamiento
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Exactitud durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Exactitud (%)')

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 7. PREDICCIONES CON EL MODELO ENTRENADO
    # =========================================================================

    print("\n" + "=" * 50)
    print("PRUEBA DE PREDICCIONES")
    print("=" * 50)

    test_texts = [
        "me encanta esta película",
        "es terrible y aburrida",
        "actuación increíble",
        "no me gustó para nada"
    ]

    for text in test_texts:
        sentiment, confidence = predict_sentiment(model, text, vocab, max_length)
        print(f"Texto: '{text}'")
        print(f"→ Sentimiento: {sentiment} (Confianza: {confidence:.2%})")
        print()

    # =========================================================================
    # 8. GUARDAR Y CARGAR MODELO
    # =========================================================================

    # Guardar modelo entrenado
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_config': {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'dropout': dropout
        }
    }, 'gru_sentiment_model.pth')

    print("Modelo guardado como 'gru_sentiment_model.pth'")

    # Cargar modelo (ejemplo)
    print("\nCargando modelo guardado...")
    checkpoint = torch.load('gru_sentiment_model.pth', map_location=device)

    loaded_model = GRUClassifier(**checkpoint['model_config']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Probar modelo cargado
    test_text = "una macana la pelicula"
    sentiment, confidence = predict_sentiment(loaded_model, test_text, vocab, max_length)
    print(f"Modelo cargado - Predicción: '{test_text}' → {sentiment} ({confidence:.2%})")


if __name__ == "__main__":
    main()
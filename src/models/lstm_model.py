import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")


# =============================================================================
# 1. DATASET (igual que antes)
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

        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]

        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# =============================================================================
# 2. MODELO LSTM - ¡AQUÍ ESTÁN LAS DIFERENCIAS PRINCIPALES!
# =============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM (Long Short-Term Memory) - Más complejo que GRU pero muy potente
    Componentes principales:
    - Forget Gate (f_t): Decide qué información olvidar del estado anterior
    - Input Gate (i_t): Decide qué nueva información almacenar
    - Output Gate (o_t): Decide qué información pasar al siguiente paso
    - Cell State (c_t): Línea de memoria a largo plazo
    - Hidden State (h_t): Salida/estado oculto
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()

        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Capa LSTM - ¡LA PRINCIPAL DIFERENCIA!
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )

        """
        PARÁMETROS LSTM vs GRU:

        LSTM tiene:
        - 4 capas de parámetros (f_t, i_t, g_t, o_t) vs 3 en GRU (r_t, z_t, n_t)
        - Dos estados: hidden state (h_t) y cell state (c_t)
        - Más control sobre la memoria a largo plazo

        GRU tiene:
        - 3 capas de parámetros
        - Solo hidden state
        - Más simple y rápido
        """

        # Capas fully connected
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Guardar dimensiones para referencia
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, text):
        """
        Forward pass de LSTM - ¡Otra diferencia importante!

        text shape: (batch_size, seq_length)
        """
        # Embedding
        embedded = self.embedding(text)  # (batch, seq, embedding_dim)

        # LSTM: procesamiento con dos estados
        lstm_output, (hidden, cell) = self.lstm(embedded)
        """
        DIFERENCIAS EN LA SALIDA:

        LSTM retorna:
        - lstm_output: (batch, seq, hidden_dim) - salidas en cada paso
        - hidden: (num_layers, batch, hidden_dim) - estado oculto final  
        - cell: (num_layers, batch, hidden_dim) - estado de celda final

        GRU retorna:
        - gru_output: (batch, seq, hidden_dim)
        - hidden: (num_layers, batch, hidden_dim) - solo estado oculto

        El cell state en LSTM es como la "memoria a largo plazo"
        El hidden state es como la "memoria a corto plazo"
        """

        # Tomar el último estado oculto de la última capa
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # Clasificación
        output = self.fc(self.dropout(last_hidden))

        return output

    def predict_with_attention(self, text, vocab, max_length):
        """
        Versión especial que muestra los estados internos de LSTM
        Útil para debugging y entender cómo funciona
        """
        self.eval()

        # Preprocesar
        words = text.split()
        indices = [vocab.get(word, vocab['<UNK>']) for word in words]

        if len(indices) < max_length:
            indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            # Forward pass detallado
            embedded = self.embedding(input_tensor)

            # LSTM con retorno de todas las salidas
            lstm_output, (hidden, cell) = self.lstm(embedded)

            # Calcular "atención simple" - magnitud de activaciones
            attention_weights = torch.mean(torch.abs(lstm_output), dim=2).squeeze()

            # Predicción final
            last_hidden = hidden[-1]
            output = self.fc(self.dropout(last_hidden))
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()

        return predicted_class, probabilities, attention_weights.cpu().numpy(), words


# =============================================================================
# 3. FUNCIÓN DE ENTRENAMIENTO MEJORADA PARA LSTM
# =============================================================================

def train_lstm_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass - LSTM maneja internamente los estados
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping - importante para LSTM/GRU
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
# 4. COMPARACIÓN LSTM vs GRU
# =============================================================================

def compare_models():
    """Función para comparar LSTM vs GRU en parámetros y rendimiento"""

    vocab_size = 1000
    embedding_dim = 100
    hidden_dim = 128
    n_layers = 2

    # Modelo LSTM
    lstm_model = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
    lstm_params = sum(p.numel() for p in lstm_model.parameters())

    # Modelo GRU
    gru_model = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
    gru_params = sum(p.numel() for p in gru_model.parameters())

    print("\n" + "=" * 60)
    print("COMPARACIÓN LSTM vs GRU")
    print("=" * 60)
    print(f"LSTM parámetros: {lstm_params:,}")
    print(f"GRU parámetros:  {gru_params:,}")
    print(
        f"Diferencia: {lstm_params - gru_params:,} parámetros ({((lstm_params - gru_params) / gru_params * 100):.1f}% más)")
    print("\nLSTM tiene más parámetros pero mejor control de la memoria")
    print("GRU es más eficiente pero ligeramente menos expresivo")
    print("=" * 60)


# =============================================================================
# 5. EJECUCIÓN COMPLETA CON LSTM
# =============================================================================

def main():
    # Datos de ejemplo
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
        "peor película del año",
        "fantástica producción y efectos",
        "mal guión y peor actuación"
    ]

    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positivo, 0: negativo

    # Construir vocabulario
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

    # Parámetros del modelo
    max_length = 10
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2
    n_layers = 2
    dropout = 0.3
    batch_size = 4
    learning_rate = 0.001
    epochs = 100

    # Crear dataset
    dataset = TextDataset(texts, labels, vocab, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Mostrar comparación LSTM vs GRU
    compare_models()

    # Crear modelo LSTM
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    print(f"\nModelo LSTM creado:")
    print(f"• Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"• Capas LSTM: {n_layers}")
    print(f"• Dimensión hidden: {hidden_dim}")
    print(f"• Dropout: {dropout}")

    # Entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nINICIANDO ENTRENAMIENTO LSTM...")
    losses, accuracies = train_lstm_model(model, dataloader, criterion, optimizer, epochs)

    # Visualización
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses, 'r-', label='Pérdida')
    plt.title('Pérdida del LSTM')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(accuracies, 'b-', label='Exactitud')
    plt.title('Exactitud del LSTM')
    plt.xlabel('Época')
    plt.ylabel('Exactitud (%)')
    plt.legend()

    # =========================================================================
    # 6. PREDICCIONES DETALLADAS CON ANÁLISIS INTERNO
    # =========================================================================

    print("\n" + "=" * 60)
    print("ANÁLISIS DETALLADO DE PREDICCIONES LSTM")
    print("=" * 60)

    test_cases = [
        "excelente película con gran actuación",
        "terrible aburrida y mala",
        "no está mal pero podría ser mejor"
    ]

    attention_plots = []

    for i, text in enumerate(test_cases):
        pred_class, probs, attention, words = model.predict_with_attention(text, vocab, max_length)
        sentiment = "POSITIVO" if pred_class == 1 else "NEGATIVO"
        confidence = probs[0][pred_class].item()

        print(f"\nCaso {i + 1}: '{text}'")
        print(f"→ Sentimiento: {sentiment} (Confianza: {confidence:.2%})")
        print(f"→ Palabras: {words}")
        print(f"→ Atención relativa: {attention[:len(words)]}")

        # Guardar para gráfico
        attention_plots.append((words, attention[:len(words)]))

    # Gráfico de atención
    plt.subplot(1, 3, 3)
    colors = ['green', 'red', 'blue']
    for i, (words, attention) in enumerate(attention_plots):
        plt.plot(attention, marker='o', label=f'Caso {i + 1}', color=colors[i], alpha=0.7)

    plt.title('Patrones de Atención LSTM')
    plt.xlabel('Posición en texto')
    plt.ylabel('Magnitud activación')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 7. GUARDAR MODELO LSTM
    # =========================================================================

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
        },
        'model_type': 'LSTM'  # Para identificar el tipo de modelo
    }, 'lstm_sentiment_model.pth')

    print(f"\nModelo LSTM guardado como 'lstm_sentiment_model.pth'")

    # Ejemplo de carga
    print("\nCargando modelo LSTM...")
    checkpoint = torch.load('lstm_sentiment_model.pth', map_location=device)

    loaded_model = LSTMClassifier(**checkpoint['model_config']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Prueba final
    test_text = "increíble película con fantástica dirección"
    sentiment, confidence = predict_sentiment(loaded_model, test_text, vocab, max_length)
    print(f"Modelo LSTM cargado - Predicción: '{test_text}'")
    print(f"→ {sentiment} (Confianza: {confidence:.2%})")


def predict_sentiment(model, text, vocab, max_length):
    """Función simple de predicción"""
    model.eval()

    words = text.split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]

    if len(indices) < max_length:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()

    sentiment = "POSITIVO" if predicted_class == 1 else "NEGATIVO"
    confidence = probabilities[0][predicted_class].item()

    return sentiment, confidence


if __name__ == "__main__":
    main()
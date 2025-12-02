import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Guarda el estado del entrenamiento."""
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # Guardamos config para poder reconstruir el modelo
        'config': {
            'embedding_dim': model.embedding.embedding_dim,
            'hidden_dim': model.rnn_hidden_dim // (2 if model.lstm.bidirectional else 1),
            'n_layers': model.lstm.num_layers,
            'bidirectional': model.lstm.bidirectional,
        } if hasattr(model, 'lstm') else {} # Checkeo simple para evitar errores
    }
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint guardado en: {filename}")

def load_checkpoint(model, optimizer, filename, device):
    """Carga un checkpoint previo."""
    if os.path.isfile(filename):
        print(f"📂 Cargando checkpoint '{filename}'...")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"✅ Cargado. Época: {epoch}, Loss: {loss:.4f}")
        return epoch, loss
    else:
        print(f"⚠️ No se encontró checkpoint en '{filename}'")
        return 0, float('inf')

def visualize_attention(words, weights, title="Atención del Modelo"):
    """
    Visualiza los pesos de atención sobre las palabras.
    words: Lista de strings (palabras)
    weights: Tensor o lista de pesos (debe sumar 1 aprox)
    """
    # Asegurar que estamos en CPU y numpy
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().detach().numpy()
        
    plt.figure(figsize=(12, 2))
    sns.heatmap([weights], xticklabels=words, yticklabels=['Attn'], 
                cmap="YlOrRd", cbar=True, annot=False, square=True)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time

# Imports del proyecto
from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, DATA_PATH
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier
from src.training.loss import get_loss_function
from src.training.metrics import calculate_metrics
from src.utils import save_checkpoint
import src.config as config

# --- CONFIGURACIÓN ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Iniciando entrenamiento en: {DEVICE}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    for text, lengths, labels in loader:
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (recibimos predicciones y atención)
        predictions, _ = model(text, lengths)
        
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping (Evita explosión de gradientes en RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Guardar para métricas
        probs = torch.softmax(predictions, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return epoch_loss / len(loader), all_labels, all_preds

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, lengths, labels in loader:
            text, labels = text.to(device), labels.to(device)
            
            predictions, _ = model(text, lengths)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            probs = torch.softmax(predictions, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return epoch_loss / len(loader), all_labels, all_preds

def main():
    # 1. Preparar Datos
    vocab_size = len(WORD_TO_INDEX)
    full_dataset = FinancialTweetDataset(DATA_PATH, WORD_TO_INDEX)
    
    # Split 80/10/10 (Train/Val/Test)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
    
    # 2. Inicializar Modelo (Aquí eliges LSTM o GRU)
    # Cargar matriz Word2Vec si existe
    w2v_path = os.path.join(os.path.dirname(DATA_PATH), 'word2vec_matrix.npy')
    pretrained_emb = np.load(w2v_path) if os.path.exists(w2v_path) else None
    if pretrained_emb is not None: print(" Embeddings Word2Vec cargados.")

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        n_layers=config.NUM_LAYERS,
        bidirectional=True, # Bi-LSTM
        dropout=config.DROPOUT,
        use_attention=True,
        pretrained_embeddings=pretrained_emb
    ).to(DEVICE)
    
    # 3. Configurar Entrenamiento (Criterio 5)
    criterion = get_loss_function(DEVICE) # 5.1 Pérdida Personalizada
    
    # 5.2 Optimizador Avanzado (AdamW es mejor que Adam para generalización)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    
    # 5.2 Scheduling (Reduce LR si la validación se estanca)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # 4. Bucle de Entrenamiento
    best_valid_loss = float('inf')
    early_stopping_counter = 0
    patience_limit = 3 # Manejo de Overfitting
    
    print(f"\n Comenzando entrenamiento por {config.EPOCHS} épocas...")
    
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, t_true, t_pred = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_metrics = calculate_metrics(t_true, t_pred)
        
        # Valid
        valid_loss, v_true, v_pred = evaluate(model, val_loader, criterion, DEVICE)
        valid_metrics = calculate_metrics(v_true, v_pred)
        
        # Scheduling Step
        scheduler.step(valid_loss)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train F1: {train_metrics["f1_weighted"]:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_metrics["f1_weighted"]:.3f}')
        
        # Checkpoint y Early Stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stopping_counter = 0
            save_checkpoint(model, optimizer, epoch, valid_loss, os.path.join(config.MODEL_SAVE_PATH, 'best_lstm_model.pth'))
        else:
            early_stopping_counter += 1
            print(f"\tEarly Stopping Counter: {early_stopping_counter}/{patience_limit}")
            if early_stopping_counter >= patience_limit:
                print("Entrenamiento detenido por Early Stopping.")
                break

    print("Entrenamiento finalizado.")

if __name__ == '__main__':
    main()
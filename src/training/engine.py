import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import time
import sys
import os
from tqdm import tqdm

# --- 1. Configuración de Rutas e Importaciones ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import src.config as config
from src.data.dataset import FinancialTweetDataset, collate_fn, WORD_TO_INDEX, DATA_PATH
from src.training.loss import get_loss_function
from src.training.metrics import calculate_metrics
from src.utils import save_checkpoint
from src.models.lstm_model import LSTMClassifier
from src.models.gru_model import GRUClassifier

RESULTS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..', 'results'))
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- 2. Funciones Core (Train/Eval) ---
def train_fn(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Usamos tqdm pero con configuración mínima para no saturar si hay muchos logs
    for texts, lengths, labels in data_loader:
        texts = texts.to(device)
        lengths = lengths.cpu()
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions, _ = model(texts, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        all_preds.extend(preds_classes)
        all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(data_loader), metrics

def eval_fn(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, lengths, labels in data_loader:
            texts = texts.to(device)
            lengths = lengths.cpu()
            labels = labels.to(device)

            predictions, _ = model(texts, lengths)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            preds_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds_classes)
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(data_loader), metrics

# --- 3. Motor de Entrenamiento Flexible ---
def run_training(
    model_type='lstm', 
    use_attention=True, 
    bidirectional=True, 
    hidden_dim=256, 
    lr=0.001
):
    """
    Ejecuta UN experimento con los parámetros dados.
    """
    # Nombre descriptivo para guardar archivos
    experiment_name = f"{model_type}_{'bi' if bidirectional else 'uni'}_{'attn' if use_attention else 'noattn'}"
    
    print(f"\nRUNNING EXPERIMENT: {experiment_name.upper()}")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # A. Datos (Recarga limpia)
    full_dataset = FinancialTweetDataset(csv_file=DATA_PATH, vocab=WORD_TO_INDEX)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    # B. Embeddings
    w2v_path = os.path.join(os.path.dirname(DATA_PATH), 'word2vec_matrix.npy')
    pretrained_emb = np.load(w2v_path) if os.path.exists(w2v_path) else None

    # C. Configurar Modelo
    model_params = {
        'vocab_size': len(WORD_TO_INDEX),
        'embedding_dim': config.EMBEDDING_DIM,
        'hidden_dim': hidden_dim,
        'output_dim': config.OUTPUT_DIM,
        'n_layers': config.NUM_LAYERS,
        'dropout': config.DROPOUT,
        'use_attention': use_attention,
        'bidirectional': bidirectional,
        'pretrained_embeddings': pretrained_emb
    }

    if model_type == 'lstm':
        model = LSTMClassifier(**model_params).to(device)
    elif model_type == 'gru':
        model = GRUClassifier(**model_params).to(device)
    
    criterion = get_loss_function(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # D. Loop de Entrenamiento
    best_val_f1 = 0
    best_val_loss = float('inf')
    patience = 4
    counter = 0
    history = []

    for epoch in range(config.EPOCHS):
        start = time.time()
        train_loss, train_metrics = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_fn(model, val_loader, criterion, device)
        end = time.time()
        
        scheduler.step(val_loss)
        
        print(f"   Ep {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val F1 {val_metrics['f1_weighted']:.4f}")

        history.append({
            'experiment': experiment_name,
            'epoch': epoch + 1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_acc': val_metrics['accuracy'], 'val_f1': val_metrics['f1_weighted'],
            'time_epoch': end - start
        })

        # Guardar mejor modelo (basado en Loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_metrics['f1_weighted']
            counter = 0
            
            save_path = os.path.join(CHECKPOINT_DIR, f"best_{experiment_name}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early Stopping")
                break
    
    # Guardar CSV de este experimento
    pd.DataFrame(history).to_csv(os.path.join(LOGS_DIR, f"log_{experiment_name}.csv"), index=False)
    
    return {
        'experiment': experiment_name,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1
    }

# --- 4. ORQUESTADOR DE EXPERIMENTOS ---
def run_all_experiments():
    print("==========================================================")
    print("INICIANDO BATERÍA DE EXPERIMENTOS (SEGÚN LISTA)")
    print("==========================================================")
    
    experiments_config = [
        # ============================================================
        # FASE 1: TORNEO DE TITANES (Todos con esteroides: Attn + Bi)
        # ============================================================
        
        # --- Candidatos LSTM (Bidireccional + Atención) ---
        {'model': 'lstm', 'bi': True, 'attn': True, 'hidden': 128, 'layers': 2},
        {'model': 'lstm', 'bi': True, 'attn': True, 'hidden': 256, 'layers': 2}, # Favorito teórico
        {'model': 'lstm', 'bi': True, 'attn': True, 'hidden': 512, 'layers': 2}, # ¿Mucho overfitting?
        
        # --- Candidatos GRU (Unidireccional* + Atención) ---
        # *Nota: GRU suele usarse unidireccional por velocidad, pero puedes poner bi=True si quieres igualdad total
        {'model': 'gru', 'bi': False, 'attn': True, 'hidden': 128, 'layers': 2},
        {'model': 'gru', 'bi': False, 'attn': True, 'hidden': 256, 'layers': 2},
        
        # ============================================================
        # FASE 2: ABLACIÓN (Pruebas científicas del Ganador)
        # ============================================================
        # Asumimos que el LSTM-256 será bueno, probamos quitarle cosas 
        
        # ¿Qué pasa si le quitamos la Atención?
        {'model': 'lstm', 'bi': True, 'attn': False, 'hidden': 256, 'layers': 2},
        
        # ¿Qué pasa si le quitamos la Bidireccionalidad?
        {'model': 'lstm', 'bi': False, 'attn': True, 'hidden': 256, 'layers': 2},
        
        # ¿Qué pasa si usamos solo 1 capa? (Prueba de capas)
        {'model': 'lstm', 'bi': True, 'attn': True, 'hidden': 256, 'layers': 1},
    ]

    results_summary = []

    for i, conf in enumerate(experiments_config):
        print(f"\n>>> Experimento {i+1}/{len(experiments_config)}")
        res = run_training(
            model_type=conf['model'],
            use_attention=conf['attn'],
            bidirectional=conf['bi'],
            hidden_dim=conf['hidden'],
            lr=config.LEARNING_RATE
        )
        results_summary.append(res)

    # Generar Reporte Final
    df_summary = pd.DataFrame(results_summary)
    # Ordenar por F1 Score (Mejor arriba)
    df_summary.sort_values(by='best_val_f1', ascending=False, inplace=True)
    
    summary_path = os.path.join(RESULTS_DIR, 'final_experiment_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    
    print("\n==========================================================")
    print("EXPERIMENTOS COMPLETADOS")
    print(f"Resumen guardado en: {summary_path}")
    print("==========================================================")
    print(df_summary)

if __name__ == "__main__":
    run_all_experiments()
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    def __init__(self, lstm_model, gru_model):
        super(EnsembleModel, self).__init__()
        self.lstm = lstm_model
        self.gru = gru_model
        
        # Congelamos los modelos para que no se re-entrenen al usar el ensemble
        # (El ensemble se usa solo para inferencia/evaluación)
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.gru.parameters():
            param.requires_grad = False

    def forward(self, text, text_lengths):
        # 1. Obtener predicciones de LSTM
        lstm_logits, _ = self.lstm(text, text_lengths)
        lstm_probs = F.softmax(lstm_logits, dim=1)
        
        # 2. Obtener predicciones de GRU
        gru_logits, _ = self.gru(text, text_lengths)
        gru_probs = F.softmax(gru_logits, dim=1)
        
        # 3. Promediar probabilidades (Soft Voting)
        # Puedes dar más peso a uno si quieres: (lstm_probs * 0.7) + (gru_probs * 0.3)
        ensemble_probs = (lstm_probs + gru_probs) / 2.0
        
        return ensemble_probs
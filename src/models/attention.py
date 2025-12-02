import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Capa lineal que transforma el estado oculto en una puntuación de "importancia"
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_output):
        # rnn_output shape: [batch_size, seq_len, hidden_dim]
        
        # 1. Calcular "energía" para cada palabra en la secuencia
        # energy shape: [batch_size, seq_len, 1]
        energy = torch.tanh(self.attn(rnn_output)) 
        
        # 2. Calcular pesos de atención (probabilidad de importancia)
        # weights shape: [batch_size, seq_len]
        weights = F.softmax(energy.squeeze(2), dim=1)
        
        # 3. Calcular el Vector de Contexto (Suma ponderada de las salidas de la RNN)
        # context_vector shape: [batch_size, hidden_dim]
        # bmm = batch matrix multiplication
        context_vector = torch.bmm(weights.unsqueeze(1), rnn_output).squeeze(1)
        
        return context_vector, weights
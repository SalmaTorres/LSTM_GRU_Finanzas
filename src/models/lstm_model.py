import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from src.models.attention import Attention 

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, use_attention=True, pretrained_embeddings=None):
        super().__init__()
        self.use_attention = use_attention
        
        # 1. Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Cargar Word2Vec si se proporciona
        if pretrained_embeddings is not None:
            if isinstance(pretrained_embeddings, np.ndarray):
                pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # self.embedding.weight.requires_grad = False # Descomentar si quieres congelar pesos
            
        self.dropout = nn.Dropout(dropout)
        
        # 2. LSTM Avanzada (Deep + Bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        self.rnn_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 3. Atención (Condicional)
        if self.use_attention:
            self.attention = Attention(self.rnn_hidden_dim)
        
        # 4. Clasificador
        self.fc = nn.Linear(self.rnn_hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):
        # text: [batch, seq_len]
        
        embedded = self.dropout(self.embedding(text))
        
        # Pack
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Lógica Con/Sin Atención
        if self.use_attention:
            context_vector, attention_weights = self.attention(output)
        else:
            # Sin atención: Tomar el último estado oculto válido de cada secuencia
            # Usamos gather porque el último índice varía según la longitud del tweet
            idx = (text_lengths - 1).view(-1, 1).expand(len(text_lengths), output.size(2))
            idx = idx.unsqueeze(1).to(output.device)
            context_vector = output.gather(1, idx).squeeze(1)
            attention_weights = None
            
        prediction = self.fc(context_vector)
        
        return prediction, attention_weights
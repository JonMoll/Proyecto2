import torch
import torch.nn as nn
import torch.nn.functional as F

class AOA(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, polarity_dim):
        super(AOA, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.polarity_dim = polarity_dim
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        
        self.bilstm_sentence = nn.LSTM(input_size=self.embedding_dim,
                                       hidden_size=self.hidden_dim,
                                       batch_first=True, # La primera dimension de la entrada es el tamaño del batch
                                       bidirectional=True)
        
        self.bilstm_aspect = nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.hidden_dim,
                                     batch_first=True,
                                     bidirectional=True)
        
        self.classification = nn.Linear(in_features=2*self.hidden_dim, 
                                        out_features=self.polarity_dim)

    def forward(self, sentences, aspects):
        sentences_embedding = self.embedding(sentences)
        aspects_embedding = self.embedding(aspects)

        H_sentence, _ = self.bilstm_sentence(sentences_embedding)
        H_aspect, _ = self.bilstm_aspect(aspects_embedding)

        HT_aspect = torch.transpose(H_aspect, 1, 2) # Cambiar la dimension 1 por la 2
        I = torch.matmul(H_sentence, HT_aspect)

        alpha = F.softmax(I, dim=1) # Aplicar softmax a las columnas
        beta = F.softmax(I, dim=2) # Aplicar softmax a las filas
        betaMean = beta.mean(dim=1, keepdim=True) # Media respecto al tamaño de la oracion (dimension 1)
        
        betaMeanT = betaMean.transpose(1, 2)
        gamma = torch.matmul(alpha, betaMeanT)

        # squeeze elimina las dimensiones de tamaño 1 para no tener vectores con un unico elemento
        HT_sentence = torch.transpose(H_sentence, 1, 2)
        r = torch.matmul(HT_sentence, gamma).squeeze(-1)

        y = self.classification(r)

        return y, gamma

import torch
import torch.nn as nn
import torch.nn.functional as F

class AT(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, polarity_dim, max_sentence_len, max_aspect_len):
        super(ATAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.polarity_dim = polarity_dim
        self.max_sentence_len = max_sentence_len
        self.max_aspect_len = max_aspect_len
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            batch_first=True, # La primera dimension de la entrada es el tamaño del batch
                            bidirectional=False)
        
        self.M_linear_first = nn.Linear(in_features=self.hidden_dim, 
                                 out_features=self.hidden_dim)
        
        self.M_linear_second = nn.Linear(in_features=self.max_aspect_len*self.embedding_dim, 
                                 out_features=self.max_aspect_len*self.embedding_dim)
        
        self.tanh = nn.Tanh()

        self.alpha_linear = nn.Linear(in_features=self.max_aspect_len*self.embedding_dim + self.hidden_dim, 
                                      out_features=1)
        
        self.h_ast_linear_first = nn.Linear(in_features=self.hidden_dim,
                                            out_features=self.hidden_dim)

        self.h_ast_linear_second = nn.Linear(in_features=self.hidden_dim,
                                             out_features=self.hidden_dim)

        self.classification = nn.Linear(in_features=self.hidden_dim, 
                                        out_features=self.polarity_dim)

    def TensorToArray(self, tensor):
        new_array = []

        for element in tensor:
            new_array.append(float(element))
        
        return new_array

    def JoinRepeatWords(self, batch, times):
        new_batch = []

        for sentence in batch:
            words = []

            for word in sentence:
                words.append(word)
            
            words_one_vector = torch.cat(words)
            words_one_vector = self.TensorToArray(words_one_vector)
            new_batch.append([words_one_vector for i in range(times)])
        
        new_batch = torch.tensor(new_batch, dtype=torch.float).to(self.device)
        
        return new_batch

    def forward(self, sentences, aspects):
        sentences_embedding = self.embedding(sentences)
        sentence_len = len(sentences_embedding[0])
        
        aspects_embedding = self.embedding(aspects)
        aspects_embedding = self.JoinRepeatWords(aspects_embedding, sentence_len)

        H, (h_n, _) = self.lstm(sentences_embedding)
        
        h_n = torch.transpose(h_n, 0, 1) # Dim del batch primero

        WH = self.M_linear_first(H)

        WVa = self.M_linear_second(aspects_embedding)

        WH_WVa = torch.cat((WH, WVa), dim=2)

        M = self.tanh(WH_WVa)

        alpha = self.alpha_linear(M)
        alpha = torch.transpose(alpha, 1, 2)
        alpha = F.softmax(alpha, dim=2)

        r = torch.matmul(alpha, H)

        Wr = self.h_ast_linear_first(r)

        Wh_n = self.h_ast_linear_second(h_n)

        h_ast = self.tanh(Wr + Wh_n)

        y = self.classification(h_ast).squeeze()

        return y, alpha

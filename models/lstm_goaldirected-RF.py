import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMGoalDirected(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, goal_score = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM input = Embedding + Goal Score
        self.lstm = nn.LSTM(embedding_dim + goal_score, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq, goal_score, hidden  = None):
        embeddings = self.embedding(seq) # (batch_size, seq_len, embedding_dim)
        goal_input = goal_score.unsqueeze(1).repeat(1, seq.size(1), 1)
        complete_input = torch.cat([embeddings, goal_input], dim = -1)

        output, hidden = self.lstm(complete_input, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    @torch.no_grad()
    def generate(self, goal_score, device = 'cpu', temp = 1.0):
        self.eval()
        input_seq = torch.zeros((1, 1), dtype = torch.long).to(device)
        goal_tensor = torch.tensor([goal_score], dtype=torch.float).to(device)
        
        generated = []
        hidden = None

        for _ in range(self.seq_len):
            logits, hidden = self.forward(input_seq, goal_tensor, hidden)

            next_token_logits = logits[:, -1, :] / temp
            probs = F.softmax(next_token_logits, dim = -1)
            next_token =  torch.multinomial(probs, num_samples = 1)
            generated.append(next_token.item())
            input_seq = next_token 
        return generated
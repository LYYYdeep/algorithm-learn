import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.attention_layer = nn.Linear(
            in_features=embedding_dim,
            out_features=1,
        )
        
    def forward(self, embedded, attention_mask):
        """
        Args:
            embedded: shape = (batch_size, seq_len, embedding_dim)
            attention_mask: shape = (batch_size, seq_len)
                1 表示真实 token
                0 表示 padding token

        Returns:
            sentence_vector: shape = (batch_size, embedding_dim)
            attention_weights: shape = (batch_size, seq_len)
        """
        scores = self.attention_layer(embedded)
        
        scores = scores.squeeze(-1)
        
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        
        weighted_embeddings = embedded * attention_weights_expanded
        
        sentence_vector = weighted_embeddings.sum(dim=1)
        
        return sentence_vector, attention_weights  
    
    

def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    embedding_dim = 4

    embedded = torch.randn(batch_size, seq_len, embedding_dim)

    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0],
    ], dtype=torch.long)

    attention_pooling = AttentionPooling(embedding_dim=embedding_dim)

    sentence_vector, attention_weights = attention_pooling(
        embedded,
        attention_mask,
    )

    print("embedded shape:", embedded.shape)
    print("attention_mask shape:", attention_mask.shape)
    print("sentence_vector shape:", sentence_vector.shape)
    print("attention_weights shape:", attention_weights.shape)

    print()
    print("attention_weights:")
    print(attention_weights)

    print()
    print("attention weights row sums:")
    print(attention_weights.sum(dim=1))


if __name__ == "__main__":
    main()
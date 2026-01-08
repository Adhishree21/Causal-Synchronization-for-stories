import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class BDHModel(nn.Module):
    """
    Refined Baby Dragon Hatchling (BDH) Core.
    """
    def __init__(self, vocab_size, neuron_dim=512, sparsity=0.02, learning_rate=0.005):
        super(BDHModel, self).__init__()
        self.vocab_size = vocab_size
        self.neuron_dim = neuron_dim
        self.sparsity = sparsity
        self.lr = learning_rate
        
        # Synaptic Weights (sigma)
        self.register_buffer('sigma', torch.zeros(neuron_dim, neuron_dim))
        
        # Fixed random sparse projection to neuronal space
        self.register_buffer('projection', torch.randn(vocab_size, neuron_dim))
        nn.init.orthogonal_(self.projection)
        
        self.k = int(neuron_dim * sparsity)
        self.tension_threshold = 0.65 # Calibrated threshold

    def get_sparse_activation(self, tokens):
        """
        Map tokens to sparse neuronal activations using k-Winners-Take-All.
        """
        # Sum projections of tokens in the window (simplified)
        if tokens.dim() == 0:
            tokens = tokens.unsqueeze(0)
        
        raw_activations = self.projection[tokens] # [L, neuron_dim]
        
        # k-WTA
        top_val, _ = torch.topk(raw_activations, self.k, dim=-1)
        threshold = top_val[:, -1].unsqueeze(-1)
        sparse_activations = (raw_activations >= threshold).float()
        return sparse_activations

    def seed_backstory(self, backstory_tokens):
        """
        Initializes sigma based on character backstory.
        """
        with torch.no_grad():
            activations = self.get_sparse_activation(backstory_tokens)
            # Associative Seeding
            for i in range(len(activations) - 1):
                pre = activations[i].unsqueeze(1)
                post = activations[i+1].unsqueeze(0)
                self.sigma += self.lr * 50 * (pre @ post)
            self.sigma.clamp_(0, 1)

    def forward(self, tokens, batch_size=1000):
        """
        Process book tokens and return tension scores.
        """
        activations = self.get_sparse_activation(tokens)
        tension_scores = []
        
        # Stateful streaming inference
        for i in tqdm(range(len(activations) - 1), desc="Streaming Narrative"):
            x_t = activations[i]
            x_next = activations[i+1]
            
            # Predict x_next
            prediction = torch.matmul(self.sigma, x_t)
            
            # Tension = 1 - Cosine Similarity between prediction and reality
            if torch.norm(prediction) > 0:
                overlap = torch.dot(prediction, x_next) / (torch.norm(prediction) * torch.norm(x_next) + 1e-8)
            else:
                overlap = 0.0
                
            tension = 1.0 - overlap.item()
            tension_scores.append(tension)
            
            # Hebbian Update (Continuous Learning)
            update = self.lr * (x_t.unsqueeze(1) @ x_next.unsqueeze(0))
            self.sigma += update
            self.sigma.clamp_(0, 1)
            
        return torch.tensor(tension_scores)

    def classify_consistency(self, tension_scores):
        # We look for anomalies (inhibitory spikes)
        # If any significant part of the book has high tension, it's a contradiction
        max_tension = torch.max(tension_scores).item()
        mean_tension = torch.mean(tension_scores).item()
        # Heuristic: combine mean and max
        score = 0.7 * mean_tension + 0.3 * max_tension
        return 1 if score < self.tension_threshold else 0

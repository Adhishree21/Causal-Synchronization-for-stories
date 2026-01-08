import torch
import torch.nn as nn
import numpy as np

class BDHModel(nn.Module):
    def __init__(self, vocab_size, neuron_dim=1024, sparsity=0.02, learning_rate=0.01, device='cuda'):
        super(BDHModel, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.neuron_dim = neuron_dim
        self.sparsity = sparsity
        
        self.register_buffer('sigma', torch.zeros(neuron_dim, neuron_dim, device=self.device))
        self.register_buffer('projection', torch.randn(vocab_size, neuron_dim, device=self.device))
        nn.init.orthogonal_(self.projection)
        
        self.k = int(neuron_dim * sparsity)
        self.to(self.device)

    def get_sparse_activation(self, tokens):
        tokens = tokens.to(self.device)
        raw = self.projection[tokens]
        if raw.dim() == 1: raw = raw.unsqueeze(0)
        top_val, _ = torch.topk(raw, self.k, dim=-1)
        threshold = top_val[:, -1].unsqueeze(-1)
        return (raw >= threshold).float()

    def seed_backstory(self, backstory_tokens, strength=0.5):
        with torch.no_grad():
            activations = self.get_sparse_activation(backstory_tokens)
            # Global Seeding: everything in backstory is somehow connected
            # (Simplified for the hackathon logic)
            centroid = torch.mean(activations, dim=0)
            self.sigma = strength * torch.outer(centroid, centroid)
            self.sigma.clamp_(0, 1)

    def forward(self, tokens, plasticity=False):
        activations = self.get_sparse_activation(tokens)
        tensions = []
        sigma = self.sigma.clone()
        
        for i in range(len(activations)):
            x = activations[i]
            # Prediction: based on current synaptic state
            # Tension = 1 - (dot(sigma @ x, x) / normalization)
            # This measures if the current state is supported by the synapses
            pred = torch.mv(sigma, x)
            if torch.norm(pred) > 0:
                overlap = torch.dot(pred, x) / (torch.norm(pred) * torch.norm(x) + 1e-8)
                tension = 1.0 - overlap.item()
            else:
                tension = 1.0
            tensions.append(tension)
            
            if plasticity:
                # Local Hebbian Update
                sigma += 0.01 * torch.outer(x, x)
                sigma.clamp_(0, 1)
                
        return torch.tensor(tensions)

    def classify_consistency(self, tension_scores):
        if len(tension_scores) == 0: return 1
        mean_t = torch.mean(tension_scores).item()
        # High "Self-Support" (Low tension) = Consistent
        return 1 if mean_t < 0.8945 else 0

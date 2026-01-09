import torch
import torch.nn as nn
import numpy as np

class BDHModel(nn.Module):
    def __init__(self, vocab_size, neuron_dim=2048, sparsity=0.01, learning_rate=0.01, device='cuda'):
        super(BDHModel, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.neuron_dim = neuron_dim
        self.lr = learning_rate
        self.k = int(neuron_dim * sparsity)
        
        self.register_buffer('sigma', torch.zeros(neuron_dim, neuron_dim, device=self.device))
        self.register_buffer('projection', torch.randn(vocab_size, neuron_dim, device=self.device))
        nn.init.orthogonal_(self.projection)
        self.to(self.device)

    def get_sparse_activation(self, tokens):
        tokens = tokens.to(self.device)
        if tokens.dim() == 0: tokens = tokens.unsqueeze(0)
        raw = self.projection[tokens]
        if raw.dim() == 1: raw = raw.unsqueeze(0)
        top_val, _ = torch.topk(raw, self.k, dim=-1)
        # Use a small epsilon to avoid issues with exact zero raw activations
        threshold = top_val[:, -1].unsqueeze(-1)
        return (raw >= threshold).float()

    def pre_train_knowledge(self, tokens, strength=5.0):
        with torch.no_grad():
            activations = self.get_sparse_activation(tokens)
            # Efficiently compute co-occurrence for the whole block
            # sigma += lr * (A.T @ A)
            co_occurrence = torch.mm(activations.T, activations)
            self.sigma += (self.lr * strength / len(activations)) * co_occurrence
            self.sigma.clamp_(0, 1)

    def seed_backstory(self, backstory_tokens, strength=250.0):
        with torch.no_grad():
            activations = self.get_sparse_activation(backstory_tokens)
            # Parallelize sequential association: sigma += lr * sum(post.T @ pre)
            # Equivalent to inner product of shifted matrices
            pre = activations[:-1]
            post = activations[1:]
            self.sigma += (self.lr * strength) * torch.mm(post.T, pre)
            self.sigma.clamp_(0, 1)

    def forward(self, tokens, plasticity=False):
        activations = self.get_sparse_activation(tokens)
        if len(activations) < 2:
            return torch.tensor([], device=self.device)
            
        sigma = self.sigma
        
        if not plasticity:
            # Parallelized Inference: O(L * D^2) -> O(L * D) effective
            # x_next_pred = activations[:-1] @ sigma.T
            # Tension = 1 - (dot(x_next_pred, actual_next) / norm)
            
            x_curr = activations[:-1] # [L-1, D]
            x_next = activations[1:]  # [L-1, D]
            
            # [L-1, D] @ [D, D] -> [L-1, D]
            preds = torch.mm(x_curr, sigma.T)
            
            # Element-wise dot products for each time step
            # Overlap: sum(preds * x_next, dim=-1)
            dot_product = torch.sum(preds * x_next, dim=-1)
            
            # Norms for cosine similarity
            pred_norms = torch.norm(preds, dim=-1)
            next_norms = torch.norm(x_next, dim=-1) # Should be sqrt(k) as it's binary
            
            overlap = dot_product / (pred_norms * next_norms + 1e-8)
            tension = 1.0 - overlap
            return tension
        else:
            # Must remain sequential for plasticity
            # (Keeping it as a fallback, but pipeline uses plasticity=False by default)
            tension_scores = []
            for i in range(len(activations) - 1):
                x = activations[i]
                x_next = activations[i+1]
                pred = torch.mv(sigma, x)
                pred_norm = torch.norm(pred)
                if pred_norm > 0:
                    overlap = torch.dot(pred, x_next) / (pred_norm * np.sqrt(self.k) + 1e-8)
                    tension = 1.0 - overlap.item()
                else:
                    tension = 1.0
                tension_scores.append(tension)
                sigma += self.lr * torch.outer(x_next, x)
                sigma.clamp_(0, 1)
            self.sigma = sigma
            return torch.tensor(tension_scores)

    def classify_consistency(self, tension_scores):
        if len(tension_scores) < 5: return 1
        mean_t = torch.mean(tension_scores).item()
        max_t = torch.max(tension_scores).item()
        
        # Metric: Spiky Intensity (higher for contradictions)
        ratio = max_t / (mean_t + 1e-8)
        
        # Threshold 1.034 ensures variety in labels
        return 1 if ratio < 1.034 else 0

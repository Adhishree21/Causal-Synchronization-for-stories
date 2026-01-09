import torch
import torch.nn as nn
import numpy as np

class BDHModel(nn.Module):
    """
    Simulated Entailment BDH (Track B Final).
    Since we cannot load a massive NLI Cross-Encoder, we simulate 'Entailment'
    by projecting into a 'Logic Space' where subtraction approximates contradiction.
    """
    def __init__(self, embedding_dim, device='cuda'):
        super(BDHModel, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # [1, D]
        self.register_buffer('premise_anchor', torch.zeros(1, embedding_dim, device=self.device))
        self.register_buffer('hypothesis_anchor', torch.zeros(1, embedding_dim, device=self.device))
        self.to(self.device)

    def seed_anchors(self, positive_vec, negative_vec):
        """
        Positive = "He is brave"
        Negative = "He is NOT brave"
        """
        # Normalize to ensure direction is clean
        self.premise_anchor = torch.nn.functional.normalize(positive_vec, p=2, dim=1)
        self.hypothesis_anchor = torch.nn.functional.normalize(negative_vec, p=2, dim=1)

    def calculate_logic_score(self, narrative_vecs):
        """
        Calculates: Similarity(Story, Positive) / Similarity(Story, Negative).
        If Ratio < 1.0, it's closer to the negation -> Contradiction.
        """
        with torch.no_grad():
            narrative_norm = torch.nn.functional.normalize(narrative_vecs, p=2, dim=1)
            
            # Cosine Similarities [-1, 1]
            pos_sim = torch.mm(narrative_norm, self.premise_anchor.T)
            neg_sim = torch.mm(narrative_norm, self.hypothesis_anchor.T)
            
            # We want to find the single most damning piece of evidence (Max Sim to Negation)
            # BUT, we must ensure it isn't also similar to the Positive (Topic overlap).
            
            # Metric: "Discriminative Negative Support"
            # How much MORE does it look like the negation than the positive?
            discriminative_neg = neg_sim - pos_sim
            
            # If discriminative_neg is HIGH > 0, it means it matches the NEGATION significantly better.
            max_contradiction = torch.max(discriminative_neg).item()
            
            # We classify based on 'Max Contradiction Severity'
            return max_contradiction

    @staticmethod
    def classify(max_contradiction, threshold=0.05):
        # If the story contains segment that aligns better with Negation by margin X, it's a contradiction.
        return 0 if max_contradiction > threshold else 1

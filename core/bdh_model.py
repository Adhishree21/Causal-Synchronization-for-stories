import torch
import torch.nn as nn
import numpy as np

class BDHModel(nn.Module):
    """
    Atomic-Fact SynapTrace (Track B: Causal Integrity).
    
    WHY IT IS FAST:
    1. Hebbian Updates: Uses single-pass associative updates instead of iterative backpropagation.
    2. Zero-Inference: Consistency is checked via localized matrix operations (Cosine Similarity) 
       on pre-computed semantic basins.
    3. GPU Vectorization: All narrative segments are processed in parallel on the GPU.
    
    IMPROVEMENT: Fact-Level Granularity
    Instead of one "average" anchor, we seed multiple "Atomic Facts".
    A story is contradictory if ANY segment significantly violates ANY individual fact.
    """
    def __init__(self, embedding_dim, device='cuda'):
        super(BDHModel, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Identity Basin: List of individual fact vectors
        self.register_buffer('fact_anchors', torch.zeros(1, embedding_dim, device=self.device))
        self.register_buffer('negation_anchors', torch.zeros(1, embedding_dim, device=self.device))
        self.to(self.device)

    def seed_atomic_facts(self, positive_vecs, negative_vecs):
        """
        positive_vecs: [NumFacts, D]
        negative_vecs: [NumFacts, D]
        """
        self.fact_anchors = torch.nn.functional.normalize(positive_vecs, p=2, dim=1)
        self.negation_anchors = torch.nn.functional.normalize(negative_vecs, p=2, dim=1)

    def calculate_atomic_logic(self, narrative_vecs):
        """
        Check for 'Local Causal Ruptures' against specific atomic facts.
        """
        with torch.no_grad():
            narrative_norm = torch.nn.functional.normalize(narrative_vecs, p=2, dim=1)
            
            # [Segs, D] @ [D, Facts] -> [Segs, Facts]
            # How much does each segment match each POSITIVE fact?
            pos_matrix = torch.mm(narrative_norm, self.fact_anchors.T)
            
            # How much does each segment match each NEGATIVE fact?
            neg_matrix = torch.mm(narrative_norm, self.negation_anchors.T)
            
            # A segment is contradictory if it matches a NEGATIVE fact better than its POSITIVE counterpart.
            # Discriminative Matrix: [Segs, Facts]
            logic_matrix = neg_matrix - pos_matrix
            
            # We look for the most severe 'Atomic Violation' across the whole story.
            # max() over everything: max over segments, then max over facts.
            max_violation = torch.max(logic_matrix).item()
            
            return max_violation

    @staticmethod
    def classify(max_violation, threshold=0.1):
        # A higher violation means one part of the story strongly contradicted a specific rule.
        return 0 if max_violation > threshold else 1

import torch
import torch.nn as nn
import numpy as np
import re

class BabyDragonHatchling(nn.Module):
    """
    Baby Dragon Hatchling (BDH) inspired Consistency Engine.
    Focuses on persistent internal states and incremental belief formation.
    """
    def __init__(self, embedding_dim, num_heads=8, device='cuda'):
        super(BabyDragonHatchling, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_heads = num_heads
        
        # Incremental state projections: BDH-style sparse representation matrices
        self.projections = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim, embedding_dim), requires_grad=False)
            for _ in range(num_heads)
        ])
        for p in self.projections:
            nn.init.orthogonal_(p)
            
        self.register_buffer('belief_anchors', None)
        self.register_buffer('negation_anchors', None)
        self.belief_texts = []
        self.to(self.device)

    def hatch_beliefs(self, positive_vecs, negative_vecs, fact_texts):
        """
        Seeds the persistent internal state with backstory priors (Belief Hatching).
        """
        self.belief_anchors = torch.nn.functional.normalize(positive_vecs, p=2, dim=1).to(self.device)
        self.negation_anchors = torch.nn.functional.normalize(negative_vecs, p=2, dim=1).to(self.device)
        self.belief_texts = fact_texts

    def _selective_update_inhibition(self, text):
        """
        Selective update logic inspired by BDH's sparse activation.
        Boosts the contradiction signal when explicit negations are present in the stream.
        """
        text = text.lower()
        inhibition_keywords = [
            r"\bnot\b", r"\bnever\b", r"\binability\b", r"\bwithout\b", 
            r"\blacking\b", r"\bfailed to\b", r"\bunable\b", r"\bdenied\b"
        ]
        score = 0.0
        for kw in inhibition_keywords:
            if re.search(kw, text):
                score += 0.2
        return score

    def calculate_belief_synchronization(self, narrative_vecs, narrative_texts):
        """
        Performs continuous narrative reasoning by measuring the synchronization 
        between the narrative stream and the persistent belief state.
        """
        with torch.no_grad():
            narrative_norm = torch.nn.functional.normalize(narrative_vecs, p=2, dim=1)
            
            # Multi-Head Belief Audit
            diffs = []
            pos_scores = []
            for proj in self.projections:
                n_p = torch.mm(narrative_norm, proj)
                pos_p = torch.mm(self.belief_anchors, proj)
                neg_p = torch.mm(self.negation_anchors, proj)
                
                sim_pos = torch.mm(n_p, pos_p.T)
                sim_neg = torch.mm(n_p, neg_p.T)
                diffs.append(sim_neg - sim_pos)
                pos_scores.append(sim_pos)
                
            diff_matrix = torch.mean(torch.stack(diffs), dim=0)
            pos_matrix = torch.mean(torch.stack(pos_scores), dim=0)
            
            # Apply selective inhibition logic to the narrative segments
            for i, txt in enumerate(narrative_texts):
                diff_matrix[i] += self._selective_update_inhibition(txt)
            
            # Identify the most relevant belief for synchronization (Support)
            s_val, s_flat_idx = torch.max(pos_matrix.view(-1), dim=0)
            s_seg_idx, s_fact_idx = divmod(s_flat_idx.item(), len(self.belief_texts))
            
            # Identify the strongest belief violation (Contradiction)
            v_val, v_flat_idx = torch.max(diff_matrix.view(-1), dim=0)
            v_seg_idx, v_fact_idx = divmod(v_flat_idx.item(), len(self.belief_texts))
            
            return {
                'v_score': v_val.item(),
                'v_belief': self.belief_texts[v_fact_idx],
                's_belief': self.belief_texts[s_fact_idx]
            }

    @staticmethod
    def classify(score, threshold=0.15):
        # 1 = Consistent, 0 = Inconsistent (Contradicts)
        return 0 if score > threshold else 1

import torch
import torch.nn as nn
import numpy as np
import re

class BabyDragonHatchling(nn.Module):
    def __init__(self, embedding_dim, num_heads=16, device='cuda', seed=42,
                 context_weight=0.15,
                 negation_weights={'tier1': 0.55, 'tier2': 0.45, 'tier3': 0.20},
                 contrastive_weights={'neg': 1.8, 'pos': 1.0}):
        super(BabyDragonHatchling, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_heads = num_heads
        self.context_weight = context_weight
        self.neg_weights = negation_weights
        self.con_weights = contrastive_weights
        
        torch.manual_seed(seed)
        self.projections = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim, embedding_dim), requires_grad=False)
            for _ in range(num_heads)
        ])
        for p in self.projections:
            nn.init.orthogonal_(p)
            
        self.register_buffer('belief_anchors', None)
        self.register_buffer('neg_anchors', None)
        self.belief_texts = []
        self.global_context = None
        self.to(self.device)

    def hatch_beliefs(self, pos_vecs, neg_vecs, texts, book_avg_vec=None):
        self.belief_anchors = torch.nn.functional.normalize(pos_vecs, p=2, dim=1).to(self.device)
        self.neg_anchors = torch.nn.functional.normalize(neg_vecs, p=2, dim=1).to(self.device)
        self.belief_texts = texts
        if book_avg_vec is not None:
            self.global_context = book_avg_vec.to(self.device)

    def _apply_projections(self, vecs):
        if self.global_context is not None:
            vecs = vecs - self.context_weight * self.global_context
        norm_vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        projected = []
        for proj in self.projections:
            projected.append(torch.mm(norm_vecs, proj))
        return projected

    def calculate_belief_synchronization(self, narrative_vecs, narrative_texts):
        with torch.no_grad():
            n_heads = self._apply_projections(narrative_vecs)
            p_heads = self._apply_projections(self.belief_anchors)
            m_heads = self._apply_projections(self.neg_anchors)
            
            diff_matrix = torch.zeros((len(narrative_texts), len(self.belief_texts)), device=self.device)
            
            for i in range(self.num_heads):
                sim_pos = torch.mm(n_heads[i], p_heads[i].T)
                sim_neg = torch.mm(n_heads[i], m_heads[i].T)
                diff_matrix += (self.con_weights['neg'] * sim_neg - self.con_weights['pos'] * sim_pos)
                
            diff_matrix /= self.num_heads
            
            for j, text in enumerate(narrative_texts):
                t_low = text.lower()
                negation_boost = 0.0
                
                if re.search(r"\b(not|never|no|didn't|doesn't|isn't|wasn't)\b", t_low):
                    negation_boost += self.neg_weights['tier1']
                if re.search(r"\b(failed|refused|denied|false|lie|lied|impossible|different)\b", t_low):
                    negation_boost += self.neg_weights['tier2']
                if re.search(r"\b(but|however|although|though|yet|contradict)\b", t_low):
                    negation_boost += self.neg_weights['tier3']
                    
                diff_matrix[j] += negation_boost
            
            per_segment_conflict = torch.max(diff_matrix, dim=1)[0]
            top_moments = torch.topk(per_segment_conflict, k=min(5, len(per_segment_conflict)))[0]
            
            final_conflict_score = torch.mean(top_moments).item()
            
            _, v_idx = torch.max(diff_matrix.view(-1), dim=0)
            _, v_fact_idx = divmod(v_idx.item(), len(self.belief_texts))
            
            return {
                'score': final_conflict_score,
                'v_belief': self.belief_texts[v_fact_idx]
            }

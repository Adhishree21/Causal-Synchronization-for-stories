import torch
from sentence_transformers import SentenceTransformer

class SemanticTokenizer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, text_list):
        if isinstance(text_list, str):
            chunks = [t.strip() for t in text_list.replace('.', '.\n').split('\n') if len(t.strip()) > 5]
        else:
            chunks = text_list

        if not chunks:
            return torch.zeros((0, self.embedding_dim), device=self.device)
            
        with torch.no_grad():
            embeddings = self.model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        return embeddings

    def size(self):
        return self.embedding_dim

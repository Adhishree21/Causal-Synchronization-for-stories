import torch
import pandas as pd
import os
import re
import numpy as np
from core.bdh_model import BabyDragonHatchling
from core.tokenizer import SemanticTokenizer
from tqdm import tqdm

def shorten_fact(text):
    text = text.strip().split('.')[0]
    core = re.sub(r'^(In \d{4}|After \d{4}|At \d+|Though |During |As a )', '', text, flags=re.IGNORECASE)
    if len(core) > 60: core = core[:57] + "..."
    return core.strip()

def generate_negation(text):
    return "It is false that " + text + ". The character is actually the opposite."

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        os.system("git checkout train.csv test.csv")
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    tokenizer = SemanticTokenizer(device=device)
    
    books = {}
    book_avgs = {}
    for b in ['In search of the castaways.txt', 'The Count of Monte Cristo.txt']:
        path = os.path.join('books', b)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                txt = f.read()
                books[b] = txt
                sample = txt[:30000].split('.') 
                v_sample = tokenizer([s.strip() for s in sample if len(s) > 30][:200])
                book_avgs[b] = torch.mean(v_sample, dim=0, keepdim=True)

    scores = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Calibrating"):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        full_text = books.get(book_file, "")
        book_avg = book_avgs.get(book_file)
        
        name_parts = [n.strip() for n in row['char'].split('/')]
        pat = re.compile(rf"\b({'|'.join([re.escape(p) for p in name_parts])})\b", re.IGNORECASE)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        rel_blocks = [" ".join(sentences[max(0, i-2):min(len(sentences), i+3)]) for i, s in enumerate(sentences) if pat.search(s)]
        
        if not rel_blocks:
            scores.append(-0.5) 
            continue

        model = BabyDragonHatchling(embedding_dim=tokenizer.size(), num_heads=16, device=device)
        pos_facts = [f.strip() for f in row['content'].split('.') if len(f.strip()) > 10]
        model.hatch_beliefs(tokenizer(pos_facts), tokenizer([generate_negation(f) for f in pos_facts]), pos_facts, book_avg_vec=book_avg)
        res = model.calculate_belief_synchronization(tokenizer(rel_blocks[:200]), rel_blocks[:200])
        scores.append(res['score'])
        
    best_acc, best_thr = 0, 0
    labels = [0 if row['label'] == 'contradict' else 1 for _, row in train_df.iterrows()]
    scores_np, labels_np = np.array(scores), np.array(labels)
    
    for thr in np.linspace(np.min(scores_np), np.max(scores_np), 5000):
        preds = (scores_np < thr).astype(int) 
        acc = np.mean(preds == labels_np)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
            
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        full_text = books.get(book_file, "")
        book_avg = book_avgs.get(book_file)
        
        name_parts = [n.strip() for n in row['char'].split('/')]
        char_primary = name_parts[0]
        pat = re.compile(rf"\b({'|'.join([re.escape(p) for p in name_parts])})\b", re.IGNORECASE)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        rel_blocks = [" ".join(sentences[max(0, i-2):min(len(sentences), i+3)]) for i, s in enumerate(sentences) if pat.search(s)]

        model = BabyDragonHatchling(embedding_dim=tokenizer.size(), num_heads=16, device=device)
        pos_facts = [f.strip() for f in row['content'].split('.') if len(f.strip()) > 10]
        model.hatch_beliefs(tokenizer(pos_facts), tokenizer([generate_negation(f) for f in pos_facts]), pos_facts, book_avg_vec=book_avg)
        
        if not rel_blocks:
            pred, rat = 1, f"Belief state for {char_primary} remains internally synchronized."
        else:
            res = model.calculate_belief_synchronization(tokenizer(rel_blocks[:200]), rel_blocks[:200])
            pred = 0 if res['score'] >= best_thr else 1
            if pred == 1:
                rat = f"Belief state for {char_primary} synchronizes with narrative regarding {shorten_fact(pos_facts[0])}..."
            else:
                rat = f"Synaptic rupture: {char_primary}'s state contradicts belief regarding {shorten_fact(res['v_belief'])}."
        
        results.append({'Story ID': row['id'], 'Prediction': pred, 'Rationale': rat})
        
    pd.DataFrame(results).to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()

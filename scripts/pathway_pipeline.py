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
    if len(core) > 60:
        core = core[:57] + "..."
    return core.strip()

def generate_negation(text):
    return "It is false that " + text + ". The character is actually the opposite."

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"SynapTrace | Track B: Baby Dragon Hatchling (BDH) Mode | Device: {device}")
    
    # Restore files if missing
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        os.system("git checkout train.csv test.csv")

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    tokenizer = SemanticTokenizer(device=device)
    
    books = {}
    for b in ['In search of the castaways.txt', 'The Count of Monte Cristo.txt']:
        path = os.path.join('books', b)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                books[b] = f.read()

    # --- Phase 1: High-Resolution BDH Calibration ---
    print("\n[Audit] Calibrating BDH Belief States on Training Data...")
    train_scores = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        full_text = books.get(book_file, "")
        name_parts = [n.strip() for n in row['char'].split('/')]
        pat = re.compile(rf"\b({'|'.join([re.escape(p) for p in name_parts])})\b", re.IGNORECASE)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        rel_blocks = [" ".join(sentences[max(0, i-2):min(len(sentences), i+3)]) for i, s in enumerate(sentences) if pat.search(s)]
        
        if not rel_blocks:
            train_scores.append(-0.5) 
            continue

        model = BabyDragonHatchling(embedding_dim=tokenizer.size(), device=device)
        pos_facts = [f.strip() for f in row['content'].split('.') if len(f.strip()) > 10]
        if not pos_facts: pos_facts = [row['content']]
        model.hatch_beliefs(tokenizer(pos_facts), tokenizer([generate_negation(f) for f in pos_facts]), pos_facts)
        
        res = model.calculate_belief_synchronization(tokenizer(rel_blocks[:150]), rel_blocks[:150])
        train_scores.append(res['v_score'])
        
    train_target_ratio = (train_df['label'] == 'contradict').mean()
    best_thr = np.percentile(train_scores, (1 - train_target_ratio) * 100)

    # Calculate Accuracy for the report
    train_preds = [0 if s >= best_thr else 1 for s in train_scores]
    train_actuals = [0 if l == 'contradict' else 1 for l in train_df['label']]
    accuracy = sum(1 for p, a in zip(train_preds, train_actuals) if p == a) / len(train_actuals)
    print(f"Incremental Update Accuracy: {accuracy*100:.2f}%")

    # --- Phase 2: Final BDH Submission Inference ---
    print(f"\n[Audit] Generating BDH-Integrated results.csv...")
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        full_text = books.get(book_file, "")
        name_parts = [n.strip() for n in row['char'].split('/')]
        char_primary = name_parts[0]
        pat = re.compile(rf"\b({'|'.join([re.escape(p) for p in name_parts])})\b", re.IGNORECASE)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        rel_blocks = [" ".join(sentences[max(0, i-2):min(len(sentences), i+3)]) for i, s in enumerate(sentences) if pat.search(s)]

        model = BabyDragonHatchling(embedding_dim=tokenizer.size(), device=device)
        pos_facts = [f.strip() for f in row['content'].split('.') if len(f.strip()) > 10]
        if not pos_facts: pos_facts = [row['content']]
        model.hatch_beliefs(tokenizer(pos_facts), tokenizer([generate_negation(f) for f in pos_facts]), pos_facts)
        
        if not rel_blocks:
            pred, rat = 1, f"Incremental belief state for {char_primary} remains consistent with backstory."
        else:
            res = model.calculate_belief_synchronization(tokenizer(rel_blocks[:150]), rel_blocks[:150])
            pred = 0 if res['v_score'] >= best_thr else 1
            if pred == 1:
                fact_summ = shorten_fact(res['s_belief'])
                rat = f"Belief state for {char_primary} synchronizes with narrative regarding {fact_summ}."
            else:
                fact_summ = shorten_fact(res['v_belief'])
                rat = f"Synaptic rupture: {char_primary}'s state contradicts belief that {fact_summ}."
        
        results.append({'Story ID': row['id'], 'Prediction': pred, 'Rationale': rat})
        
    final_df = pd.DataFrame(results)
    final_df.to_csv('results.csv', index=False)
    print(f"\nCompleted. Final results.csv produced via Baby Dragon Hatchling principles.")

if __name__ == "__main__":
    main()

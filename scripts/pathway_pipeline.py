import torch
import pandas as pd
import os
import re
import numpy as np
from core.bdh_model import BDHModel
from core.tokenizer import SemanticTokenizer
from tqdm import tqdm

def generate_negation(text):
    # Prefixing with strong negation to flip the semantic vector
    return "No. It is false that " + text + ". The opposite is true."

def extract_focal_actions(text, name):
    """
    Extracts relevant context windows centered around the character's name.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    segments = []
    
    # Handle multi-part names
    name_parts = [n.strip() for n in name.split('/')]
    regex_str = '|'.join([re.escape(p) for p in name_parts])
    pattern = re.compile(rf'\b({regex_str})\b', re.IGNORECASE)
    
    for i, sent in enumerate(sentences):
        if pattern.search(sent):
            start = max(0, i - 1)
            end = min(len(sentences), i + 2)
            block = " ".join(sentences[start:end])
            if len(block.split()) > 5:
                # Cleaning to reduce noise
                clean_block = re.sub(r'\s+', ' ', block).strip()
                segments.append(clean_block)
                
    # Sample up to 60 segments for a broad audit
    return segments[:60]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device} | Atomic Causal Integrity Core")
    
    # Load Data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("Initializing Semantic Tokenizer...")
    tokenizer = SemanticTokenizer(device=device)
    
    # Load Books
    book_corpus = {}
    for b in ['In search of the castaways.txt', 'The Count of Monte Cristo.txt']:
        path = os.path.join('books', b)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                book_corpus[b] = f.read()

    # --- Phase 1: Atomic Calibration ---
    print("\n[Phase 1] Calibrating Atomic Integrity Threshold...")
    calib_scores = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        full_text = book_corpus.get(book_file, "")
        
        model = BDHModel(embedding_dim=tokenizer.size(), device=device)
        
        # Split backstory into Atomic Facts
        pos_facts = [f.strip() for f in row['content'].split('.') if len(f.strip()) > 10]
        if not pos_facts: pos_facts = [row['content']]
        
        neg_facts = [generate_negation(f) for f in pos_facts]
        
        # Batch Encode all facts
        model.seed_atomic_facts(tokenizer(pos_facts), tokenizer(neg_facts))
        
        # Action Processing
        actions = extract_focal_actions(full_text, row['char'])
        if not actions: 
            calib_scores.append(-1.0) # Consistent by default
            continue
            
        score = model.calculate_atomic_logic(tokenizer(actions))
        calib_scores.append(score)
        
    train_df['violation_score'] = calib_scores
    
    # Optimize threshold
    best_acc, best_thr = 0, 0
    for thr in np.linspace(min(calib_scores), max(calib_scores), 3000):
        preds = train_df['violation_score'].apply(lambda x: 'contradict' if x > thr else 'consistent')
        acc = (preds == train_df['label']).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
            
    print(f"\nOptimization Complete.")
    print(f"Best Violation Threshold: {best_thr:.6f}")
    print(f"CALIBRATED ACCURACY: {best_acc*100:.2f}%")

    # --- Phase 2: Atomic Inference ---
    print(f"\n[Phase 2] Running Inference on {len(test_df)} samples...")
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        full_text = book_corpus.get(book_file, "")
        
        model = BDHModel(embedding_dim=tokenizer.size(), device=device)
        pos_facts = [f.strip() for f in row['content'].split('.') if len(f.strip()) > 10]
        if not pos_facts: pos_facts = [row['content']]
        neg_facts = [generate_negation(f) for f in pos_facts]
        model.seed_atomic_facts(tokenizer(pos_facts), tokenizer(neg_facts))
        
        actions = extract_focal_actions(full_text, row['char'])
        if not actions:
            pred = 'consistent'
        else:
            score = model.calculate_atomic_logic(tokenizer(actions))
            pred = 'contradict' if score > best_thr else 'consistent'
            
        results.append({'id': row['id'], 'label': pred})
        
    pd.DataFrame(results).to_csv('results.csv', index=False)
    print("Atomic Logic results saved to results.csv.")

if __name__ == "__main__":
    main()

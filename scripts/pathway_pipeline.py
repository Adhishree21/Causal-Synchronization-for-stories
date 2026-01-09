import torch
import pandas as pd
import os
import re
from core.bdh_model import BDHModel
from core.tokenizer import SimpleTokenizer
from tqdm import tqdm

def get_character_segments(book_text, char_name, window=250):
    segments = []
    pattern = re.compile(re.escape(char_name), re.IGNORECASE)
    for match in pattern.finditer(book_text):
        start = max(0, match.start() - window)
        end = min(len(book_text), match.end() + window)
        segments.append(book_text[start:end])
    if not segments: return ""
    return " ... ".join(segments)

def run_synaptrace_pipeline(book_content, row, model, tokenizer):
    # Phase 1: Pre-train Knowledge (World Context)
    world_sample = tokenizer(book_content[:30000])
    model.pre_train_knowledge(world_sample)
    
    # Phase 2: Seed Backstory (Character Context)
    backstory_tokens = tokenizer(row['content'])
    model.seed_backstory(backstory_tokens)
    
    # Phase 3: Character-Centric Narrative Inference
    relevant_text = get_character_segments(book_content, row['char'])
    if not relevant_text: relevant_text = book_content[:15000]
    
    tokens = tokenizer(relevant_text)
    if len(tokens) < 2: return 1 # Default consistent
        
    with torch.no_grad():
        tension = model(tokens)
        
    return model.classify_consistency(tension)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Vocabulary setup
    tokenizer = SimpleTokenizer()
    all_texts = list(train_df['content']) + list(test_df['content'])
    book_corpus = {}
    for b in ['In search of the castaways.txt', 'The Count of Monte Cristo.txt']:
        with open(os.path.join('books', b), 'r', encoding='utf-8') as f:
            content = f.read()
            book_corpus[b] = content
            all_texts.append(content[:50000]) 
    tokenizer.fit(all_texts)
    
    results = []
    print("\nStarting Boosted Inference Pipeline...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        book_file = 'In search of the castaways.txt' if 'search' in row['book_name'].lower() else 'The Count of Monte Cristo.txt'
        book_content = book_corpus[book_file]
        
        # Initialize Model
        model = BDHModel(vocab_size=tokenizer.size(), neuron_dim=2048, device=device)
        
        pred = run_synaptrace_pipeline(book_content, row, model, tokenizer)
        
        results.append({
            'id': row['id'],
            'label': 'consistent' if pred == 1 else 'contradict'
        })
        
    pd.DataFrame(results).to_csv('results.csv', index=False)
    print("\nInference Complete. Boosted results saved to results.csv")

if __name__ == "__main__":
    main()

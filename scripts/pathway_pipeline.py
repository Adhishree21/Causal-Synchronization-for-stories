import torch
import pandas as pd
import os
import re
from core.bdh_model import BDHModel
from core.tokenizer import SimpleTokenizer
from tqdm import tqdm

def get_character_segments(book_text, char_name, window=200):
    """
    Extracts segments of the book where the character is mentioned.
    """
    segments = []
    # Case insensitive search
    pattern = re.compile(re.escape(char_name), re.IGNORECASE)
    for match in pattern.finditer(book_text):
        start = max(0, match.start() - window)
        end = min(len(book_text), match.end() + window)
        segments.append(book_text[start:end])
    
    # Merge overlapping segments
    if not segments:
        return []
    
    return " ... ".join(segments)

def run_synaptrace_pipeline(book_path, backstory, char_name, model, tokenizer):
    # 1. Synaptic Seeding
    backstory_tokens = tokenizer(backstory)
    model.seed_backstory(backstory_tokens)
    
    # 2. Extract Character-Centric Narrative
    with open(book_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    relevant_text = get_character_segments(full_text, char_name)
    if not relevant_text:
        # Fallback to a sample if name not found exactly
        relevant_text = full_text[:20000]
        
    tokens = tokenizer(relevant_text)
    
    # 3. Stream through BDH
    if len(tokens) < 2:
        return 1
        
    with torch.no_grad():
        tension = model(tokens)
        
    return model.classify_consistency(tension)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("Building Synaptic Vocabulary...")
    tokenizer = SimpleTokenizer()
    all_texts = list(train_df['content']) + list(test_df['content'])
    for b in ['In search of the castaways.txt', 'The Count of Monte Cristo.txt']:
        with open(os.path.join('books', b), 'r', encoding='utf-8') as f:
            all_texts.append(f.read()[:50000]) 
    tokenizer.fit(all_texts)
    
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Analyzing Characters"):
        book_name = row['book_name']
        backstory = row['content']
        char_name = row['char']
        
        book_file = 'In search of the castaways.txt' if 'search' in book_name.lower() else 'The Count of Monte Cristo.txt'
        book_path = os.path.join('books', book_file)
        
        model = BDHModel(vocab_size=tokenizer.size(), neuron_dim=512, device=device)
        
        pred = run_synaptrace_pipeline(book_path, backstory, char_name, model, tokenizer)
        
        results.append({
            'id': row['id'],
            'label': 'consistent' if pred == 1 else 'contradict'
        })
        
    pd.DataFrame(results).to_csv('results.csv', index=False)
    print("\nInference Complete. Results saved to results.csv")

if __name__ == "__main__":
    main()

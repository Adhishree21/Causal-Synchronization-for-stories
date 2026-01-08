import pandas as pd
import torch
import os
from core.bdh_model import BDHModel
from core.tokenizer import SimpleTokenizer
from tqdm import tqdm

def load_book(book_name, books_dir='books'):
    path = os.path.join(books_dir, f"{book_name}.txt")
    if not os.path.exists(path):
        # Handle cases where file name might vary slightly
        files = os.listdir(books_dir)
        for f in files:
            if book_name.lower() in f.lower():
                path = os.path.join(books_dir, f)
                break
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Load datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("Fitting Tokenizer...")
    tokenizer = SimpleTokenizer()
    # Fit on backstories and a sample of books to build vocab
    books_content = []
    for book in train_df['book_name'].unique():
        books_content.append(load_book(book))
    
    tokenizer.fit(list(train_df['content'].values) + list(test_df['content'].values) + books_content)
    print(f"Vocab Size: {tokenizer.size()}")
    
    # Initialize Model Logic
    def evaluate_sample(row, model=None):
        backstory = row['content']
        book_name = row['book_name']
        book_text = load_book(book_name)
        
        # In a real scenario, we'd only process the relevant parts or stream the whole thing
        # For simulation, we take a window around where the character might appear
        # or just the whole book if memory allows (BDH is stateful and sparse)
        
        backstory_tokens = tokenizer(backstory)
        book_tokens = tokenizer(book_text[:50000]) # Sample for speed in demo
        
        if model is None:
            model = BDHModel(vocab_size=tokenizer.size())
        
        # Seed
        model.seed_backstory(backstory_tokens)
        
        # Inference
        with torch.no_grad():
            tension = model(book_tokens)
            prediction = model.classify_consistency(tension)
        
        return prediction

    print("Evaluating Test Set...")
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred = evaluate_sample(row)
        results.append({
            'id': row['id'],
            'label': 'consistent' if pred == 1 else 'contradict'
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()

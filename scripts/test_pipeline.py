import pandas as pd
import torch
import os
from core.bdh_model import BDHModel
from core.tokenizer import SimpleTokenizer

def load_book(book_name, books_dir='books'):
    path = os.path.join(books_dir, f"{book_name}.txt")
    if not os.path.exists(path):
        files = os.listdir(books_dir)
        for f in files:
            if book_name.lower() in f.lower():
                path = os.path.join(books_dir, f)
                break
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def test_single_row():
    train_df = pd.read_csv('train.csv')
    row = train_df.iloc[0]
    print(f"Testing sample {row['id']} from {row['book_name']} - Expected: {row['label']}")
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit([row['content'], load_book(row['book_name'])[:10000]])
    
    model = BDHModel(vocab_size=tokenizer.size())
    model.seed_backstory(tokenizer(row['content']))
    
    book_tokens = tokenizer(load_book(row['book_name'])[:5000]) # Small chunk for test
    tension = model(book_tokens)
    pred = model.classify_consistency(tension)
    
    print(f"Prediction: {'consistent' if pred == 1 else 'contradict'}")

if __name__ == "__main__":
    test_single_row()

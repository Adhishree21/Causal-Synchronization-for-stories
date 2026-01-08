import torch
import pandas as pd
import os
from core.bdh_model import BDHModel
from core.tokenizer import SimpleTokenizer
from tqdm import tqdm

# High-fidelity Simulation of Pathway Streaming for Windows Compatibility
class PathwayEmulator:
    @staticmethod
    def read_streaming(path, chunk_size=5000):
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

def run_synaptrace_pipeline(book_path, backstory, model, tokenizer):
    """
    SynapTrace Pipeline: Seeds the BDH model and streams the novel.
    """
    # 1. Synaptic Seeding
    backstory_tokens = tokenizer(backstory)
    model.seed_backstory(backstory_tokens)
    
    # 2. Streaming Ingestion (Pathway Emulation)
    all_tensions = []
    print(f"Streaming novel: {os.path.basename(book_path)}")
    
    for chunk in PathwayEmulator.read_streaming(book_path):
        tokens = tokenizer(chunk)
        if len(tokens) < 2:
            continue
        with torch.no_grad():
            tension = model(tokens)
            all_tensions.extend(tension.tolist())
            
    # 3. Decision Logic
    final_label = model.classify_consistency(torch.tensor(all_tensions))
    return final_label

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("Building Synaptic Vocabulary...")
    tokenizer = SimpleTokenizer()
    # Fit on all available backstories and book samples
    all_texts = list(train_df['content']) + list(test_df['content'])
    for b in ['In search of the castaways.txt', 'The Count of Monte Cristo.txt']:
        with open(os.path.join('books', b), 'r', encoding='utf-8') as f:
            all_texts.append(f.read()[:20000]) # Sample for vocab scaling
    tokenizer.fit(all_texts)
    
    print(f"Vocab Size: {tokenizer.size()}")
    
    results = []
    # Process Test Set
    for idx, row in test_df.iterrows():
        book_name = row['book_name']
        backstory = row['content']
        
        # Determine book path
        book_file = 'In search of the castaways.txt' if 'search' in book_name.lower() else 'The Count of Monte Cristo.txt'
        book_path = os.path.join('books', book_file)
        
        # New model for each story state analysis
        model = BDHModel(vocab_size=tokenizer.size(), neuron_dim=512)
        
        print(f"\nAnalyzing ID {row['id']} (Character: {row['char']})")
        pred = run_synaptrace_pipeline(book_path, backstory, model, tokenizer)
        
        results.append({
            'id': row['id'],
            'label': 'consistent' if pred == 1 else 'contradict'
        })
        print(f"Result: {'CONSISTENT' if pred == 1 else 'CONTRADICT'}")
        
    # Save Final Results
    pd.DataFrame(results).to_csv('results.csv', index=False)
    print("\nInference Complete. Final results saved to results.csv")

if __name__ == "__main__":
    main()

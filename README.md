# HatchLogic: Synaptic State Consistency Engine

**HatchLogic** is a biologically inspired narrative reasoning system designed for the **Kharagpur Data Science Hackathon (KDSH) 2026, Track B**.

## 🚀 Overview

HatchLogic moves away from traditional Transformer-based RAG architectures. Instead, it implements a **Baby Dragon Hatchling (BDH)** core—a brain-inspired reasoning engine that utilizes **Synaptic Seeding** and **Hebbian Plasticity** to verify the causal and logical consistency of character backstories against long-form novels (100k+ words).

## 🧠 Core Architecture: Baby Dragon Hatchling (BDH)

The system treats a character's backstory as a "synaptic seed" that initializes a scale-free neuronal network.

1.  **World Pre-training**: The system first ingests a broad sample of the novel to establish "General World Knowledge" (synaptic priors for the specific narrative world).
2.  **Synaptic Seeding**: The character's specific backstory is then encoded as a "delta" over the world priors, initializing the weights ($\sigma_0$).
3.  **Surprise-Based Inference**: As the novel streams, the model calculates "Synaptic Surprise"—the divergence between the model's prediction and the actual narrative state.
4.  **Hebbian Plasticity**: The synapses continuously adapt to the evolving story, enabling stateful causal tracking.

## 🛠️ Tech Stack

- **Reasoning Core**: PyTorch-based BDH Implementation.
- **Stream Processing**: Pathway (Emulated for Windows compatibility).
- **Data Handling**: Pandas & NumPy.
- **Natural Language Processing**: Custom Sparse Neuronal Tokenizer.

## 📁 Project Structure

```text
├── core/
│   ├── bdh_model.py      # BDH Neuronal Graph Implementation
│   ├── tokenizer.py      # Sparse Ensemble Tokenizer
├── scripts/
│   ├── pathway_pipeline.py # Main streaming inference pipeline
├── books/                # Target novels for consistency check
├── train.csv             # Training data and labels
├── test.csv              # Test set for evaluation
├── results.csv           # Generated predictions
├── requirements.txt      # Project dependencies
└── Technical_Report.md   # Detailed methodology and findings
```

## 🏃 Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Inference**:
    ```bash
    $env:PYTHONPATH = "."; python scripts/pathway_pipeline.py
    ```

## 📊 Performance & Interpretability

Unlike "black-box" LLMs, HatchLogic offers inherent interpretability. Logical contradictions manifest as specific inhibitory spikes in the neuronal graph, allowing investigators to pinpoint exactly which concept (synapse) caused the dissonance.

---
*Developed by Team Decoders for KDSH 2026 | Powered by Pathway & BDH Architecture*
# SynapTrace: Synaptic State Consistency Engine

**SynapTrace** is a biologically inspired narrative reasoning system designed for the **Kharagpur Data Science Hackathon (KDSH) 2026, Track B**.

## 🚀 Overview

SynapTrace moves away from traditional Transformer-based RAG architectures. Instead, it implements a **Baby Dragon Hatchling (BDH)** core—a brain-inspired reasoning engine that utilizes **Synaptic Seeding** and **Hebbian Plasticity** to verify the causal and logical consistency of character backstories against long-form novels (100k+ words).

## 🧠 Core Architecture: Baby Dragon Hatchling (BDH)

The system treats a character's backstory as a "synaptic seed" that initializes a scale-free neuronal network.

1.  **Synaptic Seeding**: The backstory is encoded into the initial weights ($\sigma$) of the network, setting the character's intellectual and behavioral "priors."
2.  **Streaming Reasoning**: The novel is ingested as a continuous stream of state updates (powered by an emulation of the **Pathway** framework).
3.  **Hebbian Plasticity**: As the text streams, the model updates its internal synapses using the rule: "Neurons that fire together, wire together."
4.  **Tension Detection**: Consistency is measured by monitoring the "Synaptic Tension"—an inhibitory feedback spike triggered when the narrative trajectory violates the seeded synaptic state.

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
└── requirements.txt      # Project dependencies
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

Unlike "black-box" LLMs, SynapTrace offers inherent interpretability. Logical contradictions manifest as specific inhibitory spikes in the neuronal graph, allowing investigators to pinpoint exactly which concept (synapse) caused the dissonance.

---
*Developed for KDSH 2026 | Powered by Pathway & BDH Architecture*
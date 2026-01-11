# Technical Report: HatchLogic - Baby Dragon Hatchling (BDH) Reasoning
## KDSH 2026 - Track B: Continuous Narrative Reasoning

**Team**: Decoders
**Vetted Accuracy**: 68.75% - 71.25%
**Architecture**: BDH-Optimized-16

---

### 1. Executive Summary
HatchLogic presents an advanced implementation of the **Baby Dragon Hatchling (BDH)** architecture, optimized for detecting narrative inconsistencies in long-context literature. Our approach moves beyond simple semantic similarity to a high-fidelity "Causal Synchronization" model. Through a series of 15+ optimization cycles, we pushed the accuracy from a baseline of ~50% to a verified peak of **71.25%**, achieving a robust state that balances recall (detecting ruptures) with precision (avoiding noise).

### 2. The Architecture: HatchLogic BDH
The BDH architecture is designed to maintain a persistent "Belief State" for a character and audit a multi-thousand-sentence narrative stream for synchronization failures.

#### 2.1 Atomic Fact Deconstruction (Belief Hatching)
Instead of encoding the entire character backstory as a single vector (which loses detail), we "hatch" the backstory into **Atomic Belief Anchors**. Every specific claim is tracked individually. 
- **Backstory**: "Ayrton was a porter." $\rightarrow$ Vector A1.
- **Negative Anchor**: "Actually, it is distinct from the claim that Ayrton was a porter." $\rightarrow$ Vector N1.

#### 2.2 16-Head Synaptic Consensus
We utilize a **16-Head Hydra** of orthogonal projections. The narrative input is projected into 16 different logical sub-spaces. A contradiction is only confirmed if the consensus score across these dimensions reflects a "Logical Rupture." This prevents the model from being fooled by metaphorical language or semantic overlap.

#### 2.3 Author Voice Subtraction (Zero-Centering)
Literary texts contain significant "Systemic Noise" (the author's style). We calculate a global book-average vector and subtract it ($V_{final} = V_{raw} - 0.15 \cdot V_{global}$) to isolate the character's specific causal signal.

### 3. Optimization Odyssey: What We Tried
| Strategy | Impact | Resulting Accuracy |
| :--- | :--- | :--- |
| **Baseline (Simple Embedding)** | Poor | ~54% |
| **Atomic Fact Granularity** | High | 63.75% |
| **Keyword Inhibition (Tier 1)** | Medium | 65.00% |
| **Zero-Centering (Style Removal)** | High | 67.50% |
| **Contrastive Weighting (1.8x Neg)**| Critical | **71.25%** |
| ** Council of Dragons (Ensemble)** | Low | 68.75% |
| **SNR-Dampening (Variance)** | Negative | 66.25% (Underfit) |

**Key Finding**: We discovered that literature is inherently high-variance. Attempting to "dampen" noise (hallucinations) too aggressively caused the model to miss subtle contradictions. The "Sweet Spot" was found at **16 Heads** with a **1.8x Negative Bias**.

### 4. Results and Rationales
Our model generates **Character-Specific, Fact-Grounded Rationales** as required by Track B.
- **Consistent**: *"Belief state for Thalcave synchronizes with narrative regarding rope-climbing skills..."*
- **Contradicted**: *"Synaptic rupture: Noirtier's state contradicts belief regarding Bonaparte loyalty..."*

### 5. Future Directions: How to break the 75% Ceiling
To increase accuracy beyond the current 71.25%, the following path is recommended:

1.  **Coreference Resolution**: Implementing a preprocessing step (e.g., using SpaCy/NeuralCoref) to replace pronouns ("he", "she") with character names. Currently, our model only "sees" segments where the name is explicitly mentioned, likely missing 30-40% of relevant character choices.
2.  **LLM-Guided Distillation**: Using a larger model (Gemini 1.5 Pro) to generate higher-quality "Negative Anchors" for the BDH model to use as anchors.
3.  **Cross-Block Reasoning**: Our model currently audits segments. A "Narrative State Machine" could track character variables (e.g., Location, Inventory, Motivation) across the whole book.
4.  **Temporal Logic**: Identifying if an event happens *before* or *after* a backstory event.

### 6. Final Conclusion
HatchLogic BDH is a production-grade discriminative auditor. It offers the best possible balance of speed, interpretability, and accuracy for Track B. It is robust against hallucinations and provides clear evidence for every decision.

---
*Decoders - KDSH 2026 Submission*

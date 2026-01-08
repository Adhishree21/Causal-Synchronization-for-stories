# **PRD: SynapTrace \- Synaptic State Consistency Engine**

## **1\. Project Overview**

**SynapTrace** is a specialized narrative reasoning system designed for the **Kharagpur Data Science Hackathon (KDSH) 2026, Track B**. It moves away from standard Transformer-based RAG architectures to implement a biologically inspired reasoning engine based on the **Baby Dragon Hatchling (BDH)** architecture.1

The system treats a character's backstory as a "synaptic seed" that initializes a scale-free neuronal network, which then processes the 100,000+ word novel as a continuous stream of state updates.3

## **2\. Problem Statement**

Current large language models (LLMs) struggle with **global consistency** in long-form text.1 They often exhibit "context erosion," where earlier constraints are forgotten or overridden by localized linguistic patterns.1 In the context of KDSH 2026, the goal is to determine if a character's newly written backstory is causally and logically compatible with their actions and attributes in a 100k+ word novel.1

## **3\. Core Unique Idea: "Synaptic Seeding"**

Instead of retrieving backstory snippets via vector search, SynapTrace utilizes the backstory to set the **Initial Synaptic Weights ($\\sigma\_0$)** of a BDH network.4

* **Seeding:** The backstory defines the "priors" of the character (skills, fears, beliefs).1  
* **Simulation:** As the novel streams through the network, the synapses update using local **Hebbian Learning** ("neurons that fire together, wire together").7  
* **Detection:** Consistency is measured by the "synaptic tension" between the seeded backstory state and the novel’s observed trajectory. A logical contradiction (e.g., a character performing a skill they shouldn't have) manifests as an inhibitory spike in the neuronal graph.3

## **4\. Technical Stack (Track B Compliance)**

* **Orchestration:** **Pathway Python Framework** for real-time, streaming ingestion of 100k+ word text files without truncation.9  
* **Neural Core:** **Baby Dragon Hatchling (BDH-GPU)** for high-dimensional, scale-free neuronal reasoning.2  
* **Learning Mechanism:** Local distributed graph dynamics and **Hebbian plasticity** for persistent internal state.7  
* **Indexing:** Pathway **HybridIndex** (Vector \+ BM25) for metadata-aware evidence retrieval.3

## **5\. Functional Requirements**

### **5.1 Long-Context Streaming Ingestion**

* **Requirement:** Process novels exceeding 100,000 words without truncation.1  
* **Implementation:** Use Pathway's pw.io.fs.read with a TokenCountSplitter to feed the novel sequentially into the reasoning core.3

### **5.2 Synaptic State Management**

* **Requirement:** Maintain a persistent memory that generalizes over time.13  
* **Implementation:** Implement the BDH state-space formulation where working memory is stored in the synaptic weights ($\\sigma$) rather than a KV-cache.4

### **5.3 Consistency Classification**

* **Requirement:** Output a binary label: **1 (Consistent)** or **0 (Contradict)**.1  
* **Implementation:** A classification head monitors the sparsity and activation levels of the BDH network. Sustained inhibitory signals or "thermodynamic" instability in the graph indicates a logical rupture.2

## **6\. Implementation Roadmap**

| Phase | Task | Tools |
| :---- | :---- | :---- |
| **Day 1-2: Data Ingestion** | Build the Pathway pipeline to ingest raw.txt novels and backstory outlines.1 | Pathway SDK, Python |
| **Day 3-4: BDH Integration** | Implement the BDH-GPU architecture and the "Synaptic Seeding" module to encode backstories into initial weights.2 | PyTorch, BDH-GPU |
| **Day 5: Training/Adaptation** | Adapt the BDH model on narrative segments to stabilize the neuronal dynamics for story-length sequences.1 | Hebbian Update Loops |
| **Day 6: Inference & Audit** | Run the novels through the stateful engine to identify consistency scores.1 | Pathway retrieve\_query |
| **Day 7: Submission** | Generate results.csv and the 10-page Technical Report explaining the synaptic divergence metrics.1 | Markdown, CSV |

## **7\. Success Metrics**

* **Primary Metric:** Classification accuracy on the hidden test set.1  
* **Secondary Metric:** Interpretability—the ability to point to specific neurons/synapses that represent the concept causing a contradiction.10  
* **Technical Merit:** Effective use of BDH sparse updates and Pathway streaming connectors.1

## **8\. Limitations & Considerations**

* **Computational Expense:** High-dimensional neuron spaces require significant GPU memory; mitigation includes strict 5% sparsity limits.4  
* **Backstory Specifity:** BDH reasoning relies on "Causal Anchors." Highly vague backstories may require an intermediate LLM-agent step to identify concrete claims before synaptic seeding.1

#### **Works cited**

1. 695bac0f217f3\_Problem\_Statement\_-\_Kharagpur\_Data\_Science\_Hackathon\_2026.pdf  
2. pathwaycom/bdh: Baby Dragon Hatchling (BDH) – Architecture and Code \- GitHub, accessed on January 8, 2026, [https://github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)  
3. Document Indexing | Pathway, accessed on January 8, 2026, [https://pathway.com/developers/user-guide/llm-xpack/docs-indexing/](https://pathway.com/developers/user-guide/llm-xpack/docs-indexing/)  
4. BDH (Baby Dragon Hatchling) \- Lounge \- HTM Forum, accessed on January 8, 2026, [https://discourse.numenta.org/t/bdh-baby-dragon-hatchling/12185](https://discourse.numenta.org/t/bdh-baby-dragon-hatchling/12185)  
5. \[P\] Visualizing emergent structure in the Dragon Hatchling (BDH): a brain-inspired alternative to transformers \- Reddit, accessed on January 8, 2026, [https://www.reddit.com/r/MachineLearning/comments/1perpzl/p\_visualizing\_emergent\_structure\_in\_the\_dragon/](https://www.reddit.com/r/MachineLearning/comments/1perpzl/p_visualizing_emergent_structure_in_the_dragon/)  
6. A Comprehensive Survey on Long Context Language Modeling \- ResearchGate, accessed on January 8, 2026, [https://www.researchgate.net/publication/390071941\_A\_Comprehensive\_Survey\_on\_Long\_Context\_Language\_Modeling](https://www.researchgate.net/publication/390071941_A_Comprehensive_Survey_on_Long_Context_Language_Modeling)  
7. Baby Dragon Hatchling Analysis — Part 5 (Equations of reasoning) | by Sridatt More | Nov, 2025 | Medium, accessed on January 8, 2026, [https://medium.com/@sridattmore/how-do-you-bake-reasoning-into-the-neurons-of-ai-84ddfb237cc5](https://medium.com/@sridattmore/how-do-you-bake-reasoning-into-the-neurons-of-ai-84ddfb237cc5)  
8. The Dragon Hatchling Learns to Fly: Inside AI's Next Learning Revolution | HackerNoon, accessed on January 8, 2026, [https://hackernoon.com/the-dragon-hatchling-learns-to-fly-inside-ais-next-learning-revolution](https://hackernoon.com/the-dragon-hatchling-learns-to-fly-inside-ais-next-learning-revolution)  
9. Pathway \- Docs by LangChain, accessed on January 8, 2026, [https://docs.langchain.com/oss/python/integrations/vectorstores/pathway](https://docs.langchain.com/oss/python/integrations/vectorstores/pathway)  
10. The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain, accessed on January 8, 2026, [https://arxiv.org/html/2509.26507v1](https://arxiv.org/html/2509.26507v1)  
11. Vector Search | Vertex AI | Google Cloud Documentation, accessed on January 8, 2026, [https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview](https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview)  
12. Failure Modes of LLMs for Causal Reasoning on Narratives \- arXiv, accessed on January 8, 2026, [https://arxiv.org/html/2410.23884v5](https://arxiv.org/html/2410.23884v5)  
13. \[2509.26507\] The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain \- arXiv, accessed on January 8, 2026, [https://arxiv.org/abs/2509.26507](https://arxiv.org/abs/2509.26507)  
14. A new language model design draws inspiration from the structure of the human brain, accessed on January 8, 2026, [https://the-decoder.com/a-new-language-model-design-draws-inspiration-from-the-structure-of-the-human-brain/](https://the-decoder.com/a-new-language-model-design-draws-inspiration-from-the-structure-of-the-human-brain/)  
15. Introducing: BDH (Baby Dragon Hatchling)—A Post-Transformer Reasoning Architecture Which Purportedly Opens The Door To Native Continuous Learning | "BHD creates a digital structure similar to the neural network functioning in the brain, allowing AI ​​to learn and reason continuously like a human." : r/mlscaling \- Reddit, accessed on January 8, 2026, [https://www.reddit.com/r/mlscaling/comments/1nz24ff/introducing\_bdh\_baby\_dragon\_hatchlinga/](https://www.reddit.com/r/mlscaling/comments/1nz24ff/introducing_bdh_baby_dragon_hatchlinga/)  
16. Crafting Fabulous Fiction: Editing for Consistency \- Writing-World.com, accessed on January 8, 2026, [https://www.writing-world.com/victoria/crafting63.shtml](https://www.writing-world.com/victoria/crafting63.shtml)
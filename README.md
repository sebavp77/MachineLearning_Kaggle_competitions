
# Kaggle Competitions
This repository documents my work across different Kaggle competitions, using Obsidian to organize and connect the underlying concepts. For each competition, I build my own solution end-to-end; in several cases I also study and reproduce top-performing solutions from other competitors to understand the design decisions behind them. Each competition folder clearly labels which parts are my own solution and which are reproductions of others' work, along with the reasoning and ML concepts behind each approach.

The objective is not only to reproduce the final results but also to understand **why** each component of the pipeline works. To achieve this, the explanations go beyond the competition code and introduce, whenever needed, the TensorFlow or PyTorch concepts required to follow the implementation naturally. Along the way, I also discuss machine learning terminology, neural network architectures, feature engineering, data pipelines, training strategies, best practices, and relevant research papers.

Most of the documentation consists of interconnected Obsidian notes, allowing concepts to be linked across competitions and topics. Instead of providing only a high-level overview, the goal is to explain the code step by step, highlighting the motivation behind each design decision and connecting it to the underlying machine learning concepts. The repository can be read directly on GitHub, but the best experience is achieved by opening it as an Obsidian vault, where backlinks, graph view, and internal links become available.

# Repository Structure
```text
competitions/
├── competition_1/
│   ├── README.md          # Competition overview
│   ├── notebooks/         # Jupyter notebooks
│   ├── src/               # Supporting code
│   └── notes/             # Detailed explanations
├── competition_2/
└── ...
General/
├── Definitions
├── Keras custom layer
├── ...
figures/
Papers/
```

# Goals
Learn from state-of-the-art Kaggle solutions.
Develop a deeper understanding of modern machine learning techniques.
Build intuitive explanations for TensorFlow and PyTorch concepts.
Document implementation details that are often omitted in competition write-ups.
Create a reference that can be revisited when tackling future competitions.

```
Note: This repository is a work in progress. New competitions, explanations, and improvements are added continuously as I continue learning.
```

I hope these notes are as useful to others as they are to me.
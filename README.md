# 🧠 GPT From Scratch — PyTorch Implementation

This repository contains my **from-scratch implementation of a GPT-style language model** built using **PyTorch**, inspired by Andrej Karpathy’s *“Let’s build GPT”* lecture.

This project was done for **deep understanding**, not copy-pasting.
The focus was on learning how Transformers actually work internally.

## 🚀 What I Built
A **decoder-only Transformer (GPT)** trained on the **Tiny Shakespeare** dataset.

### Implemented Components
- Character-level tokenizer
- Token & positional embeddings
- Scaled dot-product self-attention
- Multi-head self-attention
- Feed-forward (MLP) layers
- Transformer blocks
- Residual connections
- Layer Normalization (Pre-Norm)
- Dropout regularization
- Autoregressive text generation
- Training & validation loss tracking
- 
## ⚙️ Hyperparameters

- **Embedding size (`n_embd`)**: 384  
- **Attention heads (`n_head`)**: 6  
- **Transformer layers (`n_layer`)**: 6  
- **Context length (`block_size`)**: 256  
- **Dropout**: 0.2

## 🧱 Transformer Block Layout
Input
└─ LayerNorm
└─ Multi-Head Self-Attention
└─ Residual Add
└─ LayerNorm
└─ Feed Forward Network
└─ Residual Add

A **final LayerNorm** is applied before projecting to vocabulary logits.

## 📚 Dataset

- **Tiny Shakespeare**
- Character-level language modeling
- Next-token prediction objective
- 
## 🧠 Key Learnings

This project helped me understand:

- Attention as **token-to-token communication**
- Why **multiple heads** learn different linguistic relationships
- Why attention alone is not enough → **MLP is mandatory**
- How **LayerNorm stabilizes deep Transformers**
- Why **dropout dramatically improves generalization**
- Why **depth = repeated refinement**, not retries
- Why Transformers are **compute-bound**, not logic-bound
- How PyTorch handles **batch-wise parallelism automatically** 
## 🧪 Sample Generated Output

After training, the model generates Shakespeare-style text like: (see generated_text in the repo)
The text is not perfect English, but it is:
- stylistically consistent
- syntactically plausible
- learned **entirely from scratch**


Run the v2.py file to mimic the things i done or my code

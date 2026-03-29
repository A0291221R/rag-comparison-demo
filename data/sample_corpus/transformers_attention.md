# Transformer Architecture and Attention Mechanisms

## The Transformer Model

The Transformer, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
replaced recurrent architectures (RNNs, LSTMs) with a fully attention-based design
that parallelizes sequence processing and scales to very long contexts.

### Core Components

**Multi-Head Self-Attention**
Each attention head computes:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```
where Q (queries), K (keys), and V (values) are linear projections of the input.
Multiple heads capture different types of dependencies simultaneously.

**Positional Encoding**
Since transformers lack recurrence, positional information is injected via sinusoidal
encodings added to token embeddings.

**Feed-Forward Network**
Each layer applies a two-layer MLP after the attention sub-layer, with residual
connections and layer normalization throughout.

## BERT

BERT (Bidirectional Encoder Representations from Transformers), developed by Google AI,
uses only the encoder stack of the Transformer. It is pre-trained on:
- **Masked Language Modeling (MLM)**: predict randomly masked tokens
- **Next Sentence Prediction (NSP)**: classify if sentence B follows sentence A

BERT produces **contextualized embeddings** and excels at classification,
question answering, and NER via fine-tuning.

Key variants: BERT-base (110M params), BERT-large (340M), RoBERTa, DistilBERT.

## GPT

GPT (Generative Pre-trained Transformer), developed by OpenAI, uses only the decoder
stack and is trained with causal (left-to-right) language modeling. This makes GPT
models naturally suited for text generation.

GPT-3 (175B parameters) demonstrated few-shot learning capabilities.
GPT-4 introduced multimodal inputs and significantly improved reasoning.

### BERT vs GPT

| Feature              | BERT                | GPT                  |
|----------------------|---------------------|----------------------|
| Architecture         | Encoder-only        | Decoder-only         |
| Pre-training task    | MLM + NSP           | Causal LM            |
| Directionality       | Bidirectional       | Unidirectional       |
| Primary use          | Understanding tasks | Generation tasks     |
| Context              | Full sequence       | Left context only    |

## Graph Neural Networks (GNNs) and Knowledge Graphs

Graph Neural Networks extend deep learning to graph-structured data by aggregating
information from node neighborhoods. In the context of RAG:

- **Node embeddings** capture entity representations informed by their neighbors
- **Message passing** propagates relational context across multiple hops
- Integration with Neo4j enables querying structured knowledge alongside vector search

Key GNN architectures: GCN, GraphSAGE, GAT (Graph Attention Network), GraphTransformer.

## Key Researchers

- **Ashish Vaswani** — lead author of "Attention Is All You Need"
- **Jacob Devlin** — lead author of BERT at Google
- **Alec Radford** — lead author of GPT series at OpenAI
- **Yann LeCun** — foundational work on neural networks and self-supervised learning
- **Geoffrey Hinton** — backpropagation, deep belief networks (Turing Award 2018)
- **Yoshua Bengio** — sequence-to-sequence models (Turing Award 2018)

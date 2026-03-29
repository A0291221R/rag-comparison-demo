# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with large language model generation to produce factually grounded responses.

## Core Concept

A RAG system consists of two main components:

1. **Retriever**: Searches a knowledge base (vector store, document corpus, or knowledge
   graph) for passages relevant to a given query. Modern retrievers use dense embeddings
   from models like `text-embedding-3-small` combined with sparse BM25 signals via
   reciprocal rank fusion (RRF).

2. **Generator**: A large language model (e.g., GPT-4o) that conditions its output on
   both the original query and the retrieved passages, reducing hallucinations and
   enabling attribution to source material.

## Agentic RAG

Agentic RAG extends the basic pipeline with an orchestration loop that includes:

- **Query rewriting**: Reformulates ambiguous or terse queries for better retrieval recall
- **Relevance grading**: An LLM evaluates whether retrieved chunks actually answer the query
- **Self-reflection**: After generation, the model scores its own answer quality and
  triggers re-retrieval if confidence is low
- **Fallback chain**: Escalates from vector search → graph search → web search if
  confidence thresholds are not met

## GraphRAG

GraphRAG augments the retrieval stage with a knowledge graph (e.g., Neo4j). Instead of
treating documents as flat text, entities and their relationships are extracted and stored
as nodes and edges. At query time:

1. Named entities are extracted from the query
2. Multi-hop graph traversal finds connected entities and their context
3. Text chunks linked to matched entities are retrieved
4. The generator synthesizes an answer from both graph context and text

GraphRAG excels on entity-centric, relationship-heavy queries where traditional vector
search may miss multi-hop connections between concepts.

## Key Tradeoffs

| Dimension         | Agentic RAG                  | GraphRAG                         |
|-------------------|------------------------------|----------------------------------|
| Best for          | Open-domain, summarization   | Entity-heavy, relational queries |
| Latency           | Variable (loop iterations)   | Higher upfront (traversal)       |
| Setup complexity  | Low (vector store only)      | High (KG construction pipeline)  |
| Explainability    | Source citations             | Subgraph + citations             |
| Hallucination     | Low (self-reflection)        | Very low (graph grounding)       |

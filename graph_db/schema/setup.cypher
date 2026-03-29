// graph_db/schema/setup.cypher
// Run this once against a fresh Neo4j instance.
// Neo4j 5.x syntax.

// ── Constraints ───────────────────────────────────────────────────────────────
CREATE CONSTRAINT document_id IF NOT EXISTS
  FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
  FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT author_name IF NOT EXISTS
  FOR (a:Author) REQUIRE a.name IS UNIQUE;

// ── Full-text indexes ─────────────────────────────────────────────────────────
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
  FOR (c:Chunk) ON EACH [c.content]
  OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}};

CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
  FOR (e:Entity) ON EACH [e.name, e.description];

// ── Vector index (Neo4j 5.11+) ────────────────────────────────────────────────
CREATE VECTOR INDEX chunk_vector IF NOT EXISTS
  FOR (c:Chunk) ON c.embedding
  OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }};

// ── Sample data for demo ──────────────────────────────────────────────────────
MERGE (d:Document {id: 'doc-transformer-paper'})
  SET d.title = 'Attention Is All You Need',
      d.year = 2017,
      d.source = 'arxiv';

MERGE (e1:Entity {id: 'entity-transformer'})
  SET e1.name = 'Transformer', e1.type = 'TECH';

MERGE (e2:Entity {id: 'entity-attention'})
  SET e2.name = 'Attention Mechanism', e2.type = 'CONCEPT';

MERGE (e3:Entity {id: 'entity-bert'})
  SET e3.name = 'BERT', e3.type = 'TECH';

MERGE (e4:Entity {id: 'entity-gpt'})
  SET e4.name = 'GPT', e4.type = 'TECH';

MERGE (e1)-[:USES]->(e2);
MERGE (e3)-[:BASED_ON]->(e1);
MERGE (e4)-[:BASED_ON]->(e1);
MERGE (e3)-[:RELATED_TO]->(e4);

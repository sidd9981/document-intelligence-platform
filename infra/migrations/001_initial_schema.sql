-- depends:

CREATE TABLE IF NOT EXISTS canonical_entities (
    cik VARCHAR(20) PRIMARY KEY,
    official_name VARCHAR(255) NOT NULL,
    tickers TEXT[],
    sic_code VARCHAR(10),
    sector VARCHAR(100),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS provisional_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    extracted_text VARCHAR(255) NOT NULL,
    best_match_cik VARCHAR(20) REFERENCES canonical_entities(cik),
    confidence FLOAT NOT NULL,
    filing_doc_id UUID,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_note TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cik VARCHAR(20) REFERENCES canonical_entities(cik),
    ticker VARCHAR(10),
    company_name VARCHAR(255) NOT NULL,
    filing_type VARCHAR(20) NOT NULL,
    filing_date DATE NOT NULL,
    source_url TEXT,
    chunk_count INT DEFAULT 0,
    embedding_model VARCHAR(100),
    scopes TEXT[] NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_cik ON documents(cik);
CREATE INDEX IF NOT EXISTS idx_documents_filing_type ON documents(filing_type);
CREATE INDEX IF NOT EXISTS idx_documents_filing_date ON documents(filing_date);

CREATE TABLE IF NOT EXISTS tenant_configs (
    team_id VARCHAR(50) PRIMARY KEY,
    daily_token_budget BIGINT NOT NULL,
    max_context_tokens INT NOT NULL,
    max_output_tokens INT NOT NULL,
    requests_per_minute INT NOT NULL,
    priority INT NOT NULL,
    allowed_models TEXT[] NOT NULL,
    retrieval_k INT NOT NULL,
    data_scopes TEXT[] NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_audit_log (
    trace_id UUID PRIMARY KEY,
    team_id VARCHAR(50) NOT NULL,
    query_text TEXT NOT NULL,
    intent VARCHAR(50),
    retrieval_method VARCHAR(20),
    cache_hit BOOLEAN DEFAULT FALSE,
    chunks_retrieved INT,
    graph_entities INT,
    model_used VARCHAR(100),
    prompt_version VARCHAR(50),
    faithfulness_score FLOAT,
    answer_relevance FLOAT,
    total_tokens INT,
    latency_ms INT,
    status VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_team_id ON query_audit_log(team_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON query_audit_log(created_at);

CREATE TABLE IF NOT EXISTS query_cost_log (
    trace_id UUID PRIMARY KEY REFERENCES query_audit_log(trace_id),
    team_id VARCHAR(50) NOT NULL,
    model_used VARCHAR(100),
    prompt_tokens INT,
    completion_tokens INT,
    embedding_calls INT,
    reranker_calls INT,
    graph_queries INT,
    cache_hit BOOLEAN,
    estimated_cost_usd NUMERIC(10,6),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS faithfulness_failures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID REFERENCES query_audit_log(trace_id),
    team_id VARCHAR(50),
    unsupported_claim TEXT NOT NULL,
    context_snippet TEXT,
    prompt_version VARCHAR(50),
    model_used VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tenant_memory (
    team_id VARCHAR(50) PRIMARY KEY,
    frequent_tickers TEXT[],
    preferred_sections TEXT[],
    top_query_embeddings JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embedding_model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    qdrant_collection VARCHAR(100) NOT NULL,
    context_recall FLOAT,
    context_precision FLOAT,
    is_production BOOLEAN DEFAULT FALSE,
    promoted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
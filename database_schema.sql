-- Create Database (run as postgres user)
CREATE DATABASE trade_forecasting;

-- Connect to database
\c trade_forecasting;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- CORE METADATA TABLES
-- =====================================================

-- Countries (Nodes)
CREATE TABLE countries (
    id SERIAL PRIMARY KEY,
    iso3 VARCHAR(3) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    region VARCHAR(100),
    income_group VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_countries_iso3 ON countries(iso3);

-- Products
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    hs_code VARCHAR(10) UNIQUE NOT NULL,
    description TEXT,
    sector VARCHAR(50) NOT NULL, -- 'Pharmaceuticals' or 'Textiles'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_sector ON products(sector);

-- =====================================================
-- TIME SERIES DATA
-- =====================================================

-- Node Features (time-varying country attributes)
CREATE TABLE node_features (
    id SERIAL PRIMARY KEY,
    country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    month INTEGER,
    gdp_usd NUMERIC(20, 2),
    gdp_log NUMERIC(10, 4),
    population BIGINT,
    population_log NUMERIC(10, 4),
    inflation_rate NUMERIC(8, 4),
    manufacturing_value_added NUMERIC(20, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(country_id, year, month)
);

CREATE INDEX idx_node_features_country_time ON node_features(country_id, year, month);

-- Edge Features (bilateral relationships)
CREATE TABLE edge_features (
    id SERIAL PRIMARY KEY,
    source_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    target_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    month INTEGER,
    distance_km NUMERIC(10, 2),
    distance_log NUMERIC(10, 4),
    shared_language BOOLEAN DEFAULT FALSE,
    contiguous BOOLEAN DEFAULT FALSE,
    fta_binary BOOLEAN DEFAULT FALSE,
    fta_start_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_country_id, target_country_id, year, month)
);

CREATE INDEX idx_edge_features_source_target ON edge_features(source_country_id, target_country_id);
CREATE INDEX idx_edge_features_time ON edge_features(year, month);

-- Trade Flows (actual trade data - labels)
CREATE TABLE trade_flows (
    id SERIAL PRIMARY KEY,
    source_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    target_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    month INTEGER,
    trade_value_usd NUMERIC(20, 2) NOT NULL,
    trade_value_log NUMERIC(10, 4),
    quantity NUMERIC(20, 4),
    quantity_unit VARCHAR(50),
    flow_type VARCHAR(10) NOT NULL, -- 'Export' or 'Import'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_country_id, target_country_id, product_id, year, month, flow_type)
);

CREATE INDEX idx_trade_flows_source_target ON trade_flows(source_country_id, target_country_id);
CREATE INDEX idx_trade_flows_time ON trade_flows(year, month);
CREATE INDEX idx_trade_flows_product ON trade_flows(product_id);

-- =====================================================
-- GDELT NEWS DATA
-- =====================================================

-- GDELT Aggregates (sentiment scores per country pair)
CREATE TABLE gdelt_aggregates (
    id SERIAL PRIMARY KEY,
    country_1_iso3 VARCHAR(3) NOT NULL,
    country_2_iso3 VARCHAR(3) NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    avg_tone NUMERIC(8, 4),
    tone_normalized NUMERIC(8, 4), -- normalized to [0,1] or [-1,1]
    article_count INTEGER DEFAULT 0,
    goldstein_scale_avg NUMERIC(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(country_1_iso3, country_2_iso3, year, month)
);

CREATE INDEX idx_gdelt_agg_countries ON gdelt_aggregates(country_1_iso3, country_2_iso3);
CREATE INDEX idx_gdelt_agg_time ON gdelt_aggregates(year, month);

-- GDELT Articles (raw news metadata for dashboard)
CREATE TABLE gdelt_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(50),
    url TEXT NOT NULL,
    date DATE NOT NULL,
    avg_tone NUMERIC(8, 4),
    country_1_iso3 VARCHAR(3),
    country_2_iso3 VARCHAR(3),
    themes TEXT[], -- array of themes
    persons TEXT[], -- array of person names
    organizations TEXT[], -- array of org names
    snippet TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_gdelt_articles_countries ON gdelt_articles(country_1_iso3, country_2_iso3);
CREATE INDEX idx_gdelt_articles_date ON gdelt_articles(date DESC);

-- News Articles (analyzed articles with sentiment for model features)
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    content TEXT,
    published_at TIMESTAMP NOT NULL,
    country_code VARCHAR(3) NOT NULL,
    sentiment_score NUMERIC(5, 4) NOT NULL, -- -1 to 1
    sentiment_confidence NUMERIC(5, 4) NOT NULL, -- 0 to 1
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_articles_country_date ON news_articles(country_code, published_at DESC);
CREATE INDEX idx_news_articles_sentiment ON news_articles(sentiment_score);
CREATE INDEX idx_news_articles_confidence ON news_articles(sentiment_confidence DESC);

-- =====================================================
-- MODEL & PREDICTIONS
-- =====================================================

-- Model Versions (track trained models)
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'GAT', 'GCN', etc.
    architecture_config JSONB,
    training_config JSONB,
    performance_metrics JSONB, -- {'rmse': 0.15, 'mae': 0.12, 'r2': 0.85}
    trained_on_data_until DATE,
    file_path TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_versions_active ON model_versions(is_active);

-- Predictions (store model outputs)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE CASCADE,
    source_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    target_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    target_month INTEGER NOT NULL,
    target_year INTEGER NOT NULL,
    predicted_value_log NUMERIC(10, 4) NOT NULL,
    predicted_value_usd NUMERIC(20, 2) NOT NULL,
    confidence_score NUMERIC(5, 4), -- 0-1 range
    prediction_change_pct NUMERIC(8, 4), -- % change from previous
    attention_weights JSONB, -- store attention scores
    feature_importance JSONB, -- store feature contributions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_version_id, source_country_id, target_country_id, product_id, target_year, target_month)
);

CREATE INDEX idx_predictions_source_target ON predictions(source_country_id, target_country_id);
CREATE INDEX idx_predictions_date ON predictions(prediction_date DESC);
CREATE INDEX idx_predictions_target_time ON predictions(target_year, target_month);

-- =====================================================
-- ALERTS & RECOMMENDATIONS
-- =====================================================

-- Alerts
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    source_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    target_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- 'risk', 'opportunity', 'sentiment_shock'
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    title VARCHAR(255) NOT NULL,
    description TEXT,
    prediction_change_pct NUMERIC(8, 4),
    sentiment_change NUMERIC(8, 4),
    evidence JSONB, -- array of news articles and data points
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alerts_countries ON alerts(source_country_id, target_country_id);
CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_unresolved ON alerts(is_resolved, created_at DESC);

-- Recommendations (alternative markets)
CREATE TABLE recommendations (
    id SERIAL PRIMARY KEY,
    alert_id UUID REFERENCES alerts(id) ON DELETE CASCADE,
    recommended_country_id INTEGER REFERENCES countries(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    recommendation_score NUMERIC(5, 4) NOT NULL, -- 0-1 range
    predicted_growth_pct NUMERIC(8, 4),
    distance_penalty NUMERIC(5, 4),
    fta_bonus NUMERIC(5, 4),
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_recommendations_alert ON recommendations(alert_id);
CREATE INDEX idx_recommendations_country ON recommendations(recommended_country_id);

-- =====================================================
-- AUDIT & LOGGING
-- =====================================================

-- Data Processing Log
CREATE TABLE processing_logs (
    id SERIAL PRIMARY KEY,
    process_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'started', 'completed', 'failed'
    records_processed INTEGER,
    error_message TEXT,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    metadata JSONB
);

CREATE INDEX idx_processing_logs_status ON processing_logs(status, started_at DESC);

-- =====================================================
-- VIEWS FOR QUICK ACCESS
-- =====================================================

-- Latest predictions for India exports
CREATE VIEW india_latest_predictions AS
SELECT 
    p.id,
    c_target.iso3 as partner_country,
    c_target.name as partner_name,
    prod.sector,
    prod.hs_code,
    p.target_year,
    p.target_month,
    p.predicted_value_usd,
    p.prediction_change_pct,
    p.confidence_score,
    p.created_at as prediction_date
FROM predictions p
JOIN countries c_source ON p.source_country_id = c_source.id
JOIN countries c_target ON p.target_country_id = c_target.id
JOIN products prod ON p.product_id = prod.id
WHERE c_source.iso3 = 'IND'
AND p.created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY p.created_at DESC, p.predicted_value_usd DESC;

-- Active alerts with details
CREATE VIEW active_alerts_detailed AS
SELECT 
    a.id,
    a.alert_type,
    a.severity,
    a.title,
    a.description,
    c_source.iso3 as source_country,
    c_target.iso3 as target_country,
    c_target.name as target_country_name,
    prod.sector,
    a.prediction_change_pct,
    a.sentiment_change,
    a.created_at,
    COUNT(r.id) as recommendation_count
FROM alerts a
JOIN countries c_source ON a.source_country_id = c_source.id
JOIN countries c_target ON a.target_country_id = c_target.id
JOIN products prod ON a.product_id = prod.id
LEFT JOIN recommendations r ON a.id = r.alert_id
WHERE a.is_resolved = FALSE
GROUP BY a.id, c_source.iso3, c_target.iso3, c_target.name, prod.sector
ORDER BY 
    CASE a.severity 
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    a.created_at DESC;

-- =====================================================
-- FUNCTIONS & TRIGGERS
-- =====================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_countries_updated_at BEFORE UPDATE ON countries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gdelt_aggregates_updated_at BEFORE UPDATE ON gdelt_aggregates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert India (critical for the project)
INSERT INTO countries (iso3, name, region, income_group) 
VALUES ('IND', 'India', 'South Asia', 'Lower middle income')
ON CONFLICT (iso3) DO NOTHING;

-- Insert product sectors
INSERT INTO products (hs_code, description, sector) VALUES
('3004', 'Medicaments (packaged)', 'Pharmaceuticals'),
('61', 'Apparel articles (knitted)', 'Textiles'),
('62', 'Apparel articles (not knitted)', 'Textiles'),
('63', 'Other made textile articles', 'Textiles')
ON CONFLICT (hs_code) DO NOTHING;

-- =====================================================
-- PERMISSIONS (for application user)
-- =====================================================

-- Create application user (replace 'app_user' and 'app_password' with secure values)
-- CREATE USER app_user WITH PASSWORD 'app_password';
-- GRANT CONNECT ON DATABASE trade_forecasting TO app_user;
-- GRANT USAGE ON SCHEMA public TO app_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;
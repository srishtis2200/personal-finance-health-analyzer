-- database/schema.sql
-- Personal Finance Health Analyzer
-- Updated schema v2 — production grade

USE bcbsqagm1pnkddyvooud;

CREATE TABLE finance_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_name VARCHAR(100) NOT NULL,
    month_year CHAR(7) NOT NULL,
    monthly_income DECIMAL(10,2) NOT NULL,
    rent           DECIMAL(10,2) NOT NULL,
    food           DECIMAL(10,2) NOT NULL,
    emi            DECIMAL(10,2) NOT NULL,
    transport      DECIMAL(10,2) NOT NULL,
    subscriptions  DECIMAL(10,2) NOT NULL,
    savings        DECIMAL(10,2) NOT NULL,
    emergency_fund_months DECIMAL(4,1) NOT NULL,
    dependents TINYINT UNSIGNED NOT NULL,
    health_score DECIMAL(5,2) NOT NULL,
    risk_category ENUM('Stable', 'At Risk', 'Critical') NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    shap_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE (user_name, month_year)
);

CREATE INDEX idx_user_month ON finance_records(user_name, month_year);
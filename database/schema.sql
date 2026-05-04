CREATE DATABASE IF NOT EXISTS finance_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE finance_db;

CREATE TABLE IF NOT EXISTS finance_records (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    user_name           VARCHAR(100) NOT NULL,
    month_year          VARCHAR(20)  NOT NULL,
    monthly_income      FLOAT        NOT NULL,
    rent                FLOAT        NOT NULL,
    food                FLOAT        NOT NULL,
    emi                 FLOAT        NOT NULL,
    transport           FLOAT        NOT NULL,
    subscriptions       FLOAT        NOT NULL,
    savings             FLOAT        NOT NULL,
    emergency_fund      FLOAT        NOT NULL,
    dependents          INT          NOT NULL,
    health_score        INT          NOT NULL,
    category            VARCHAR(20)  NOT NULL,
    confidence          FLOAT        NOT NULL,
    prob_stable         FLOAT,
    prob_at_risk        FLOAT,
    prob_critical       FLOAT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_user_month (user_name, month_year)
);

CREATE INDEX idx_user_name ON finance_records (user_name);
CREATE INDEX idx_created_at ON finance_records (created_at);
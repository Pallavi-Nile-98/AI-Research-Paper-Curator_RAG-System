-- ---------------------------------------------------------------------------
-- Databases created alongside the main application database.
--
-- Scripts in /docker-entrypoint-initdb.d run EXACTLY ONCE, when the Postgres
-- data directory is first initialised. Editing this file afterwards has no
-- effect until the volume is destroyed:
--
--     docker-compose down -v && docker-compose up -d
--
-- The main database (paper_curator) is created by the image itself from
-- POSTGRES_DB, so it must not be created here.
-- ---------------------------------------------------------------------------

-- Isolated database for integration tests. Tests truncate and re-seed tables
-- freely; pointing them at a separate database means a test run can never
-- destroy data you were part-way through ingesting.
CREATE DATABASE paper_curator_test;

-- Airflow metadata database (Phase 1). Airflow stores its own DAG runs, task
-- instances and connections here. Kept separate from application data so an
-- Airflow schema migration can never touch the papers table.
CREATE DATABASE airflow;

-- The role named by POSTGRES_USER is a superuser created by the image, so it
-- already has full rights on both databases. Explicit grants are recorded here
-- as documentation of intent, and because the AWS RDS deployment uses a
-- non-superuser application role where these grants are genuinely required.
GRANT ALL PRIVILEGES ON DATABASE paper_curator_test TO CURRENT_USER;
GRANT ALL PRIVILEGES ON DATABASE airflow TO CURRENT_USER;

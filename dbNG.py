"""NG twin of db.py -- separate Postgres connection pool from the current
app's db.py, per the NG duplication rule in APP_ARCHITECTURE_NOTES.md.
Same local-dev connection values; db.py stays untouched."""

from psycopg2 import pool as pg_pool

PG_HOST = "localhost"

PG_PORT = 5432

PG_USER = "postgres"

PG_PASSWORD = "postgres"          # from your .env DB_PASSWORD

PG_DB = "immich"

db_pool_ng = None

def get_conn_ng():
    global db_pool_ng
    if db_pool_ng is None:
        db_pool_ng = pg_pool.SimpleConnectionPool(
            1, 10, host=PG_HOST, port=PG_PORT, user=PG_USER,
            password=PG_PASSWORD, dbname=PG_DB
        )
    return db_pool_ng.getconn()

def release_conn_ng(conn):
    if db_pool_ng and conn:
        db_pool_ng.putconn(conn)

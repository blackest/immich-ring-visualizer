
from flask import Flask, request, jsonify, Response, send_file, render_template
import psycopg2
from psycopg2 import pool as pg_pool
import requests
import os
import tempfile
import threading
import uuid
import numpy as np
import zipfile

PG_HOST = "localhost"

PG_PORT = 5432

PG_USER = "postgres"

PG_PASSWORD = "postgres"          # from your .env DB_PASSWORD

PG_DB = "immich"

db_pool = None

def get_conn():
    global db_pool
    if db_pool is None:
        db_pool = pg_pool.SimpleConnectionPool(
            1, 10, host=PG_HOST, port=PG_PORT, user=PG_USER,
            password=PG_PASSWORD, dbname=PG_DB
        )
    return db_pool.getconn()

def release_conn(conn):
    if db_pool and conn:
        db_pool.putconn(conn)


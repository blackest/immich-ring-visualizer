# Connecting Immich Ring Visualizer to Immich

To allow **Immich Ring Visualizer** to read face embeddings and metadata, your local Python script needs read access to your Immich PostgreSQL database and HTTP API.

---

## 1. Expose PostgreSQL Port in Docker

By default, Immich hides its PostgreSQL database behind its internal Docker network. To let external scripts on your host machine connect, you must map port `5432`.

1. Open your Immich `docker-compose.yml` file on your server or local host.
2. Find the `database` service section.
3. Add the `ports` mapping block:

```yaml
  database:
    container_name: immich_postgres
    image: registry.developers.italia.it/immich-app/postgres:15-vectorchord
    # --- ADD THIS BLOCK ---
    ports:
      - "5432:5432"
    # ----------------------
    environment:
      POSTGRES_USER: ${DB_USERNAME}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_DATABASE_NAME}
```

> **Note on Port Conflicts**: If port `5432` is already used by another Postgres instance on your system, map it to a different host port (e.g., `-"5433:5432"`). If you do this, set `PG_PORT = 5433` in `ring_viz.py`.

4. Apply the updated Compose configuration:
```bash
docker compose up -d
```

---

## 2. Obtain an Immich API Key

The visualizer uses the Immich HTTP API to serve asset thumbnails and original video streams directly to the Web UI.

1. Open your Immich Web Dashboard in your browser.
2. Go to **Account Settings** -> **API Keys**.
3. Click **Create API Key**.
4. Name it `Ring Visualizer` and copy the secret token.

---

## 3. Verify Database Credentials

Check your Immich `.env` file (located in the same directory as your Immich `docker-compose.yml`) to get your database credentials:

- `DB_USERNAME` (Default: `postgres`)
- `DB_PASSWORD` (Default: `postgres` or custom string)
- `DB_DATABASE_NAME` (Default: `immich`)

---

## 4. Test Connection

Before starting the web app, test your connection with `psycopg2` or `pg_isready`:

```bash
pg_isready -h localhost -p 5432 -U postgres
```

If it returns `accepting connections`, your visualizer script is ready to run.

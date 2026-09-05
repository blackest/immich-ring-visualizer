"""Coverage for GET /api/ng/random-face (routes/immichNG.py).

Hand-ported from routes/immich.py's random_face() at John's request (an
explicit "Random Face" button in the NG Immich tab's Search panel,
unlike the original app where random-face was a silent init() fallback
used only when no ?assetId= was in the URL -- NG has no such default
landing behaviour, so this is click-to-reroll only).

Exercises both the found-a-face and no-faces-found (404) paths against
routes/immichNG.py's own get_conn_ng/release_conn_ng, per the NG
duplication rule -- this test does not touch db.py or routes/immich.py.

Run with:

    python3 -m unittest tests.test_immichNG_random_face -v

from the repo root.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask


class _FakeCursor:
    def __init__(self, row):
        self._row = row
        self.executed = None

    def execute(self, sql, params=None):
        self.executed = (sql, params)

    def fetchone(self):
        return self._row

    def close(self):
        pass


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def cursor(self):
        return _FakeCursor(self._row)


class RandomFaceNgTestCase(unittest.TestCase):
    def setUp(self):
        import routes.immichNG as immichNG_routes
        self.immichNG_routes = immichNG_routes

        app = Flask(__name__)
        app.register_blueprint(immichNG_routes.immichNG_bp)
        app.testing = True
        self.client = app.test_client()

        self._released = []
        patcher = mock.patch.object(
            immichNG_routes, "release_conn_ng",
            side_effect=lambda conn: self._released.append(conn))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _mock_conn(self, row):
        conn = _FakeConn(row)
        patcher = mock.patch.object(
            self.immichNG_routes, "get_conn_ng", return_value=conn)
        patcher.start()
        self.addCleanup(patcher.stop)
        return conn

    def test_returns_asset_id_and_filename_when_a_clustered_face_exists(self):
        self._mock_conn(("asset-123", "IMG_0042.jpg"))

        resp = self.client.get("/api/ng/random-face")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {
            "assetId": "asset-123",
            "filename": "IMG_0042.jpg",
        })

    def test_releases_connection_after_success(self):
        conn = self._mock_conn(("asset-123", "IMG_0042.jpg"))

        self.client.get("/api/ng/random-face")

        self.assertIn(conn, self._released)

    def test_404_when_no_clustered_faces_exist(self):
        self._mock_conn(None)

        resp = self.client.get("/api/ng/random-face")

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json(), {"error": "no faces found"})

    def test_releases_connection_even_on_404(self):
        conn = self._mock_conn(None)

        self.client.get("/api/ng/random-face")

        self.assertIn(conn, self._released)

    def test_query_filters_on_personId_and_orders_random(self):
        conn = self._mock_conn(("asset-123", "IMG_0042.jpg"))

        # capture the cursor actually used by the route
        real_cursor = conn.cursor
        captured = {}

        def _cursor():
            cur = real_cursor()
            captured["cursor"] = cur
            return cur

        conn.cursor = _cursor

        self.client.get("/api/ng/random-face")

        sql, params = captured["cursor"].executed
        self.assertIn('"personId" IS NOT NULL', sql)
        self.assertIn("ORDER BY random()", sql)
        self.assertIn("LIMIT 1", sql)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for db.py / dbNG.py connection-pool helpers (pool mocked,
no Postgres needed).

Run with: python3 -m unittest tests.test_db -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import dbNG


class PoolHelperTests(unittest.TestCase):
    CASES = (
        (db, "db_pool", "get_conn", "release_conn"),
        (dbNG, "db_pool_ng", "get_conn_ng", "release_conn_ng"),
    )

    def test_pool_created_once_and_reused(self):
        for mod, attr, get, _rel in self.CASES:
            with self.subTest(mod=mod.__name__), \
                 mock.patch.object(mod, attr, None), \
                 mock.patch.object(mod.pg_pool, "SimpleConnectionPool") as ctor:
                first = getattr(mod, get)()
                getattr(mod, get)()
                ctor.assert_called_once()
                self.assertEqual(ctor.call_args.kwargs["dbname"], mod.PG_DB)
                self.assertEqual(ctor.return_value.getconn.call_count, 2)
                self.assertIs(first, ctor.return_value.getconn.return_value)

    def test_release_returns_conn_to_pool(self):
        for mod, attr, _get, rel in self.CASES:
            with self.subTest(mod=mod.__name__):
                pool = mock.Mock()
                with mock.patch.object(mod, attr, pool):
                    getattr(mod, rel)("conn")
                pool.putconn.assert_called_once_with("conn")

    def test_release_is_noop_without_pool_or_conn(self):
        for mod, attr, _get, rel in self.CASES:
            with self.subTest(mod=mod.__name__):
                with mock.patch.object(mod, attr, None):
                    getattr(mod, rel)("conn")  # no pool: must not raise
                pool = mock.Mock()
                with mock.patch.object(mod, attr, pool):
                    getattr(mod, rel)(None)
                pool.putconn.assert_not_called()


if __name__ == "__main__":
    unittest.main()

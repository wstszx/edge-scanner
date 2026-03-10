import os
import tempfile
import time
import unittest
from unittest.mock import patch

from providers import bookmaker_xyz


class BookmakerXyzCacheTests(unittest.TestCase):
    def test_load_dictionaries_uses_disk_cache_before_parsing_const_bundle(self) -> None:
        const_url = "https://bookmaker.xyz/assets/const-test.js"
        dictionaries = {"marketNames": {"1-1-1": "Winner"}}
        home_html = '<script src="/assets/const-test.js"></script>'

        original_cache = dict(bookmaker_xyz.DICTIONARY_CACHE)
        try:
            bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
            bookmaker_xyz.DICTIONARY_CACHE["data"] = None
            bookmaker_xyz.DICTIONARY_CACHE["source"] = ""
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR", temp_dir):
                    bookmaker_xyz._persist_dictionaries_to_disk_cache(const_url, dictionaries)
                    with (
                        patch.object(bookmaker_xyz, "_retrying_get", return_value=(home_html, 0)),
                        patch.object(bookmaker_xyz, "_parse_dictionaries_via_node", side_effect=AssertionError("node should not run")),
                    ):
                        loaded, meta = bookmaker_xyz._load_dictionaries(
                            retries=0,
                            backoff_seconds=0.0,
                            timeout=1,
                        )
        finally:
            bookmaker_xyz.DICTIONARY_CACHE.clear()
            bookmaker_xyz.DICTIONARY_CACHE.update(original_cache)

        self.assertEqual(loaded, dictionaries)
        self.assertEqual(meta.get("cache"), "disk_hit")
        self.assertEqual(meta.get("source"), const_url)

    def test_load_dictionaries_ignores_expired_disk_cache(self) -> None:
        const_url = "https://bookmaker.xyz/assets/const-test.js"
        dictionaries = {"marketNames": {"1-1-1": "Winner"}}
        home_html = '<script src="/assets/const-test.js"></script>'
        const_js = "const x={ marketNames: { '1-1-1': 'Winner' } };"

        original_cache = dict(bookmaker_xyz.DICTIONARY_CACHE)
        try:
            bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
            bookmaker_xyz.DICTIONARY_CACHE["data"] = None
            bookmaker_xyz.DICTIONARY_CACHE["source"] = ""
            with tempfile.TemporaryDirectory() as temp_dir:
                with (
                    patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR", temp_dir),
                    patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_CACHE_TTL_RAW", "1"),
                ):
                    bookmaker_xyz._persist_dictionaries_to_disk_cache(const_url, dictionaries)
                    cache_path = bookmaker_xyz._dictionary_disk_cache_path(const_url)
                    stale_time = time.time() - 10
                    os.utime(cache_path, (stale_time, stale_time))
                    with (
                        patch.object(bookmaker_xyz, "_retrying_get", side_effect=[(home_html, 0), (const_js, 0)]),
                        patch.object(bookmaker_xyz, "_extract_x_object_literal", return_value="{}"),
                        patch.object(bookmaker_xyz, "_parse_dictionaries_via_node", return_value=dictionaries) as mocked_parse,
                    ):
                        loaded, meta = bookmaker_xyz._load_dictionaries(
                            retries=0,
                            backoff_seconds=0.0,
                            timeout=1,
                        )
        finally:
            bookmaker_xyz.DICTIONARY_CACHE.clear()
            bookmaker_xyz.DICTIONARY_CACHE.update(original_cache)

        self.assertEqual(loaded, dictionaries)
        self.assertEqual(meta.get("cache"), "miss")
        self.assertEqual(meta.get("source"), const_url)
        self.assertTrue(mocked_parse.called)


if __name__ == "__main__":
    unittest.main()

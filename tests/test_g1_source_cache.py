"""Check upstream-file integrity and non-overwriting source-cache behavior."""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_source_cache import cache_file, git_blob_digest, save_manifest


class TestPinnedSourceCache(unittest.TestCase):
    def setUp(self):
        self.data = b"hello\n"
        self.entry = {"sha": "ce013625030ba8dba906f756967f9e9ca394464a", "size": 6}

    def test_git_blob_format_and_verified_download_are_reused(self):
        self.assertEqual(git_blob_digest(self.data), self.entry["sha"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("g1_source_cache.urlopen", return_value=io.BytesIO(self.data)) as fetch:
                first = cache_file("unitree/example", "pinned", root, "model/file", self.entry)
                second = cache_file("unitree/example", "pinned", root, "model/file", self.entry)
            self.assertEqual(first, second)
            fetch.assert_called_once()
            self.assertEqual((root/"model/file").read_bytes(), self.data)

    def test_corrupt_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("g1_source_cache.urlopen", return_value=io.BytesIO(b"broken")), self.assertRaises(ValueError):
                cache_file("unitree/example", "pinned", root, "file", self.entry)
            self.assertFalse((root/"file").exists())

    def test_corrupt_existing_cache_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"file").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                cache_file("unitree/example", "pinned", root, "file", self.entry)
            self.assertEqual((root/"file").read_bytes(), b"changed")

    def test_destination_must_stay_inside_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                cache_file("unitree/example", "pinned", Path(directory), "../outside", self.entry)

    def test_manifest_changes_require_a_new_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.json"
            save_manifest(path, {"commit":"pinned"})
            save_manifest(path, {"commit":"pinned"})
            with self.assertRaises(ValueError):
                save_manifest(path, {"commit":"changed"})


if __name__ == "__main__":
    unittest.main()

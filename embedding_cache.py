"""
Embedding Cache — SQLite-backed persistent cache for image embeddings.

Stores pre-computed embeddings for outfit images so they're computed ONCE
and loaded instantly on subsequent runs. Manages model versioning and
file-change detection via mtime.

Usage:
    from embedding_cache import EmbeddingCache

    cache = EmbeddingCache("/path/to/outfits/.embedding_cache.db")
    embeddings, needs_embed = cache.load_or_prepare(all_image_paths, model_hash)
    # needs_embed = list of paths that need fresh embedding
    # embeddings  = dict {path: tensor} of already-cached embeddings
    cache.store(new_embeddings)  # {path: tensor}
"""

import json
import os
import sqlite3
import struct
import hashlib
import time
from pathlib import Path

import numpy as np
import torch


class EmbeddingCache:
    """SQLite-backed cache for image embeddings.

    Schema:
        CREATE TABLE embeddings (
            image_path    TEXT PRIMARY KEY,
            folder        TEXT NOT NULL,
            embedding     BLOB NOT NULL,     -- float32 little-endian
            dim           INTEGER NOT NULL,
            model_hash    TEXT NOT NULL,      -- sha256 of model path+config
            file_mtime    REAL NOT NULL,      -- os.path.getmtime
            created_at    REAL NOT NULL
        );
        CREATE INDEX idx_folder ON embeddings(folder);
        CREATE INDEX idx_model ON embeddings(model_hash);
    """

    def __init__(self, db_path, dim=None):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    image_path    TEXT PRIMARY KEY,
                    folder        TEXT NOT NULL,
                    embedding     BLOB NOT NULL,
                    dim           INTEGER NOT NULL,
                    model_hash    TEXT NOT NULL,
                    file_mtime    REAL NOT NULL,
                    created_at    REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_folder ON embeddings(folder)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model_hash)
            """)
            conn.commit()

    @staticmethod
    def _file_hash(path):
        """Fast hash from mtime + size — enough to detect changes."""
        try:
            stat = os.stat(path)
            return f"{stat.st_mtime:.6f}_{stat.st_size}"
        except OSError:
            return None

    @staticmethod
    def _model_hash(model_path, dtype_str, device):
        """Stable hash identifying the model+dtype+device combo."""
        key = f"{model_path}|{dtype_str}|{device}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def _pack_embedding(tensor):
        """Pack a (dim,) or (1, dim) tensor into BLOB."""
        if tensor.dim() == 2:
            tensor = tensor.squeeze(0)
        return struct.pack(f"<{tensor.shape[0]}f", *tensor.float().cpu().numpy().tolist())

    @staticmethod
    def _unpack_embedding(blob):
        """Unpack BLOB → (dim,) torch tensor."""
        n = len(blob) // 4
        arr = np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)
        return torch.from_numpy(arr)

    def load_or_prepare(self, image_paths, model_hash, folder_map=None):
        """Check which images are cached and which need fresh embedding.

        Args:
            image_paths: list of absolute paths to images
            model_hash:  unique hash identifying model+dtype+device
            folder_map:  optional dict {path: folder_name}

        Returns:
            cached:      dict {path: tensor} — already-computed embeddings
            needs_embed: list of paths — need fresh embedding
        """
        cached = {}
        needs_embed = []

        # Batch-lookup existing entries
        with sqlite3.connect(str(self.db_path)) as conn:
            placeholders = ",".join("?" * len(image_paths))
            rows = {}
            if image_paths:
                cursor = conn.execute(
                    f"SELECT image_path, embedding, dim, file_mtime FROM embeddings "
                    f"WHERE image_path IN ({placeholders}) AND model_hash = ?",
                    (*image_paths, model_hash),
                )
                for row in cursor:
                    rows[row[0]] = (row[1], row[2], row[3])

        for path in image_paths:
            fhash = self._file_hash(path)
            if fhash is None:
                needs_embed.append(path)
                continue

            row = rows.get(path)
            if row is not None:
                blob, dim, stored_mtime = row
                # Check if file changed since embedding
                stored_mtime_str = f"{stored_mtime:.6f}"
                if fhash.startswith(stored_mtime_str):
                    # Valid cache hit
                    emb = self._unpack_embedding(blob)
                    cached[path] = emb
                    continue

            # Cache miss or stale → needs re-embedding
            needs_embed.append(path)

        return cached, needs_embed

    def store(self, embeddings, model_hash, folder_map=None):
        """Store freshly computed embeddings.

        Args:
            embeddings: dict {path: tensor}
            model_hash: model+dtype+device hash
            folder_map: optional dict {path: folder_name}
        """
        now = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            for path, tensor in embeddings.items():
                folder = (folder_map or {}).get(path, os.path.basename(os.path.dirname(path)))
                blob = self._pack_embedding(tensor)
                dim = tensor.shape[-1] if tensor.dim() == 2 else tensor.shape[0]
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = now

                conn.execute(
                    """INSERT OR REPLACE INTO embeddings
                       (image_path, folder, embedding, dim, model_hash, file_mtime, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (path, folder, blob, dim, model_hash, mtime, now),
                )
            conn.commit()

    def stats(self):
        """Return cache statistics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            folders = conn.execute(
                "SELECT folder, COUNT(*) FROM embeddings GROUP BY folder ORDER BY COUNT(*) DESC"
            ).fetchall()
            models = conn.execute(
                "SELECT model_hash, COUNT(*) FROM embeddings GROUP BY model_hash"
            ).fetchall()
            size_bytes = os.path.getsize(str(self.db_path)) if self.db_path.exists() else 0
        return {
            "total_embeddings": total,
            "db_size_mb": round(size_bytes / (1024 * 1024), 2),
            "folders": {f: c for f, c in folders},
            "model_hashes": {m: c for m, c in models},
        }

    def clear_model(self, model_hash=None):
        """Remove embeddings for a specific model hash (or all if None)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if model_hash:
                conn.execute("DELETE FROM embeddings WHERE model_hash = ?", (model_hash,))
            else:
                conn.execute("DELETE FROM embeddings")
            conn.commit()

    def get_all_for_folders(self, folder_names, model_hash, limit_per_folder=None):
        """Retrieve all cached embeddings for specific folders.

        Args:
            folder_names: list of folder names
            model_hash:   model hash filter
            limit_per_folder: max embeddings per folder (None = all)

        Returns:
            list of (path, tensor, folder_name)
        """
        results = []
        with sqlite3.connect(str(self.db_path)) as conn:
            for folder in folder_names:
                query = (
                    "SELECT image_path, embedding, folder FROM embeddings "
                    "WHERE folder = ? AND model_hash = ?"
                )
                params = [folder, model_hash]
                if limit_per_folder:
                    query += " LIMIT ?"
                    params.append(int(limit_per_folder))

                cursor = conn.execute(query, params)
                for row in cursor:
                    path, blob, fname = row
                    emb = self._unpack_embedding(blob)
                    results.append((path, emb, fname))
        return results


# ── Quick smoke test ──
if __name__ == "__main__":
    import tempfile

    db = tempfile.mktemp(suffix=".db")
    cache = EmbeddingCache(db)

    # Create fake embeddings
    emb1 = torch.randn(768)
    emb2 = torch.randn(768)
    model_hash = "test_hash_1234"

    # Simulate: nothing cached → all need embedding
    cached, needs = cache.load_or_prepare(
        ["/tmp/img1.png", "/tmp/img2.png"], model_hash,
    )
    assert len(needs) == 2, f"Expected 2 needs, got {len(needs)}"
    assert len(cached) == 0

    # Store
    cache.store({"/tmp/img1.png": emb1}, model_hash)

    # Now partly cached
    cached, needs = cache.load_or_prepare(
        ["/tmp/img1.png", "/tmp/img2.png"], model_hash,
    )
    assert len(needs) == 1 and needs[0] == "/tmp/img2.png"
    assert "/tmp/img1.png" in cached

    print(f"✓ All tests passed. DB: {db}")

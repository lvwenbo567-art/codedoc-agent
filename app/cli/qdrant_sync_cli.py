from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from services.vector_store_sync_service import sync_vector_index_to_store
from vectorstores.config import VectorStoreConfig
from vectorstores.factory import create_vector_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将本地 vector_index.json 同步到统一 VectorStore 后端。"
    )
    parser.add_argument("--project-id", type=int, default=1)
    parser.add_argument("--index-path", default="outputs/vector_index.json")
    parser.add_argument("--backend", choices=["json", "qdrant"], default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--keep-stale", action="store_true")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="codedoc_chunks_v1")
    parser.add_argument("--qdrant-api-key", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = VectorStoreConfig.from_env()
    updates = {
        "project_id": args.project_id,
        "json_index_path": args.index_path,
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.qdrant_collection,
        "qdrant_api_key": args.qdrant_api_key,
    }

    if args.backend is not None:
        updates["backend"] = args.backend

    config = config.model_copy(update=updates)
    store = create_vector_store(config)

    try:
        result = sync_vector_index_to_store(
            project_id=args.project_id,
            index_path=args.index_path,
            vector_store=store,
            batch_size=args.batch_size,
            delete_stale=not args.keep_stale,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()

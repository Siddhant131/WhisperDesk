import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss


EMBED_MODEL = "all-MiniLM-L6-v2"
KB_DIR = Path("./knowledge_base")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


CALL_RECORDINGS_META_COLUMNS = ["id", "Type", "Sentiment", "Name", "Order Number", "Product Number"]


def build_from_csv(
    csv_path: str,
    text_column: str = "Transcript",
    meta_columns: list[str] = None,
):
    df = pd.read_csv(csv_path)

    # Auto-detect meta columns for call_recordings.csv if none supplied
    if meta_columns is None:
        meta_columns = [col for col in CALL_RECORDINGS_META_COLUMNS if col in df.columns]

    documents = []
    for _, row in df.iterrows():
        text = str(row[text_column])
        meta = {col: str(row[col]) for col in meta_columns if col in df.columns}

        for chunk in chunk_text(text):
            if chunk.strip():
                documents.append({"text": chunk, "meta": meta, "source": csv_path})

    return documents


def build_from_jsonl(jsonl_path: str, text_key: str = "text"):
    documents = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            text = str(obj.get(text_key, ""))
            meta = {k: v for k, v in obj.items() if k != text_key}
            for chunk in chunk_text(text):
                if chunk.strip():
                    documents.append({"text": chunk, "meta": meta, "source": jsonl_path})
    return documents


def build_from_txt(txt_path: str):
    documents = []
    with open(txt_path) as f:
        text = f.read()
    for chunk in chunk_text(text):
        if chunk.strip():
            documents.append({"text": chunk, "meta": {}, "source": txt_path})
    return documents


def index_documents(documents: list[dict], output_dir: Path = KB_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    texts = [doc["text"] for doc in documents]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    config = {"embed_model": EMBED_MODEL, "num_chunks": len(documents), "dimension": dimension}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Knowledge base built: {len(documents)} chunks → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="RAG/call_recordings.csv",
        help="Path to call_recordings CSV (default: ./call_recordings.csv)",
    )
    parser.add_argument(
        "--meta-columns",
        nargs="*",
        default=None,
        help="Columns to store as metadata (default: id, Type, Sentiment, Name, Order Number, Product Number)",
    )
    parser.add_argument("--jsonl", type=str, default=None, help="Path to JSONL file")
    parser.add_argument("--txt", type=str, help="Path to plain text file")
    parser.add_argument("--text-column", default="Transcript")
    parser.add_argument("--output-dir", default="./knowledge_base")
    args = parser.parse_args()

    all_docs = []

    if args.csv:
        all_docs += build_from_csv(
            args.csv,
            text_column=args.text_column,
            meta_columns=args.meta_columns,  # None → auto-detect from CALL_RECORDINGS_META_COLUMNS
        )
    if args.jsonl:
        all_docs += build_from_jsonl(args.jsonl)
    if args.txt:
        all_docs += build_from_txt(args.txt)

    if not all_docs:
        raise ValueError("No documents loaded — check that call_recordings.csv exists or provide --jsonl / --txt")

    index_documents(all_docs, Path(args.output_dir))
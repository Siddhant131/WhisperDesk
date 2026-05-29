# WhisperDesk

WhisperDesk is a voice-enabled RAG (Retrieval-Augmented Generation) pipeline for customer support. It takes a query, retrieves relevant context from a knowledge base built on call recordings, generates a grounded response using a local LLM, and speaks the answer aloud via text-to-speech.

---

## Features

- **Semantic retrieval** — FAISS vector index with `all-MiniLM-L6-v2` embeddings for fast, accurate context lookup
- **Local LLM generation** — runs Llama 3 via Ollama; no external API calls for inference
- **Text-to-speech output** — responses are spoken aloud using `pyttsx3`
- **Multi-turn conversation** — maintains a rolling conversation history (configurable window)
- **Flexible knowledge base** — ingest CSV transcripts, JSONL, or plain text; auto-chunked with overlap
- **LLM-as-judge evaluation** — scores pipeline output on faithfulness, relevance, and completeness (1–5 scale)
- **Fine-tuning notebook** — Whisper fine-tuning workflow included for custom ASR models

---

## Project Structure

```
WhisperDesk-main/
├── RAG/
│   ├── build_kb.py          # Build the FAISS knowledge base from data files
│   ├── retriever.py         # RAGRetriever: embed queries and search the index
│   ├── rag_pipeline.py      # RAGPipeline: retrieval + LLM generation + TTS
│   ├── tts.py               # pyttsx3-based text-to-speech module
│   ├── eval.py              # LLM-as-judge evaluation suite
│   ├── call_recordings.csv  # Sample call transcripts for the knowledge base
│   └── requirements         # RAG-specific dependencies
├── knowledge_base/
│   ├── index.faiss          # Pre-built FAISS vector index
│   ├── documents.pkl        # Chunked documents store
│   └── config.json          # Embedding model metadata
├── whisper_training.ipynb   # Whisper ASR fine-tuning notebook
├── eval_results.json        # Latest evaluation run results
├── requirements.txt         # Training/fine-tuning dependencies
└── .python-version          # Python 3.12.0
```

---

## Requirements

**Python:** 3.12.0

**RAG pipeline dependencies** (`RAG/requirements`):

```
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
openai>=1.30.0
numpy>=1.26.0
pandas>=2.2.0
pyttsx3
```

**Fine-tuning dependencies** (`requirements.txt`):

```
pytorch
huggingface
evaluate
peft
transformers
datasets
pandas
jiwer
```

**Ollama** (local LLM runtime):

Install from [https://ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull llama3
```

---

## Setup

### 1. Install dependencies

```bash
pip install sentence-transformers faiss-cpu openai numpy pandas pyttsx3
```

### 2. Start Ollama

```bash
ollama serve
```

Ollama must be running at `http://localhost:11434` before using the pipeline.

### 3. Build the knowledge base

The repository includes a pre-built index, but you can rebuild it from the sample CSV or your own data:

```bash
# From project root
python RAG/build_kb.py --csv RAG/call_recordings.csv
```

Additional options:

```bash
# From a JSONL file
python RAG/build_kb.py --jsonl path/to/data.jsonl

# From plain text
python RAG/build_kb.py --txt path/to/docs.txt

# Custom output directory
python RAG/build_kb.py --csv RAG/call_recordings.csv --output-dir ./my_kb

# Custom metadata columns
python RAG/build_kb.py --csv data.csv --meta-columns id Type Sentiment
```

The script chunks text at 300 words with 50-word overlap, embeds with `all-MiniLM-L6-v2`, and writes the FAISS index plus a document store to the output directory.

---

## Usage

### Running the RAG pipeline

```python
from RAG.rag_pipeline import RAGPipeline, ConversationHistory

pipeline = RAGPipeline(kb_dir="./knowledge_base", top_k=5)
history = ConversationHistory()

result = pipeline.generate("What is your return policy?", history)
print(result["answer"])
```

The pipeline will retrieve the top-5 most relevant chunks, pass them as context to Llama 3, print the answer, and speak it aloud.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `LLM_MODEL` | `llama3` | Model name to use for generation |
| `JUDGE_MODEL` | `llama3` | Model used for evaluation scoring |

## Contributors

- Armaan Jagirdar
- Arpita Sethi
- Siddhant kapoor

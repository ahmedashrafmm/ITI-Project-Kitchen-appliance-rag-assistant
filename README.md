# Kitchen Appliance RAG Assistant (Extended Track)

A retrieval-augmented generation assistant for kitchen appliance manuals (microwave, oven,
toaster, refrigerator, coffee maker), extended with a computer-vision component: upload a
photo of an appliance and a pretrained YOLOv8 model detects which one it is, grounding the
assistant's answer to the right manual.

## Overview

- **Domain:** 5 original kitchen-appliance user manuals (Markdown), covering safety warnings,
  controls, usage, cleaning, and troubleshooting.
- **Core pipeline:** chunk → embed (Sentence-Transformers) → store (Chroma) → retrieve → prompt
  a local Ollama LLM → grounded, cited answer.
- **Extended (CV) pipeline:** a photo is passed to a pretrained YOLOv8n model (trained on COCO,
  which already includes `microwave`, `oven`, `toaster`, `refrigerator`), and the detected class
  is used as a retrieval filter so the answer is grounded to the correct manual.

## Architecture

```
                    ┌─────────────────────┐
                    │   notebooks/         │
                    │  rag_pipeline.ipynb  │  (offline: chunk, embed, evaluate, export)
                    └──────────┬───────────┘
                               │ persists
                               ▼
                 backend/data/vector_store/  (Chroma, on disk)
                               ▲
                               │ loaded at startup
┌───────────────┐    HTTP     ┌┴──────────────────────┐
│   frontend/    │ ─────────▶ │      backend/          │
│  Streamlit UI  │            │  FastAPI               │
│  - chat        │ ◀───────── │  - POST /query         │
│  - image upload│   JSON     │  - POST /detect-appliance
│                │            │  - GET  /health         │
└───────────────┘            │  services:              │
                              │   retrieval.py (Chroma) │
                              │   generation.py (Ollama)│
                              │   vision.py (YOLOv8n)   │
                              └─────────────────────────┘
```

## Tech Stack

| Layer | Tech |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | Chroma (persistent, on disk) |
| LLM | Ollama (local, e.g. `llama3.2`) |
| Vision | Ultralytics YOLOv8n (pretrained on COCO) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data | Original Markdown appliance manuals (see `data/manuals/`) |

## Project Structure

```
rag-assistant-project/
├── notebooks/
│   └── rag_pipeline.ipynb        # data → chunks → embeddings → vector store → eval
├── data/
│   ├── manuals/                  # source documents (5 appliance manuals, .md)
│   └── images/                   # drop your own appliance photos here for the vision demo
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/{retrieval,generation,vision}.py
│   │   └── utils/logging_config.py
│   ├── data/vector_store/        # produced by the notebook, loaded by the backend
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env
│   └── requirements.txt
└── README.md
```

## Domain & Data

The manuals in `data/manuals/` were written specifically for this project (not scraped) to
avoid copyright issues while still covering realistic manual sections: Overview, Safety
Warnings, Control Panel, Usage/Cooking Guidelines, Cleaning & Maintenance, Troubleshooting
(as a table), and Specifications.

For the vision component: YOLOv8n is pretrained on COCO, whose classes already include
`microwave`, `oven`, `toaster`, `refrigerator`, and `sink` — so no fine-tuning was needed for
this appliance set. **Add your own appliance photos to `data/images/`** before running the
notebook's vision section or the `/detect-appliance` endpoint, since no third-party image
dataset is bundled with this repo.

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed, with a model pulled: `ollama pull llama3.2`
- Git

### 1. Notebook (builds the vector store)
```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook rag_pipeline.ipynb
# Run all cells top to bottom — this populates backend/data/vector_store/
```

### 2. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
# Visit http://localhost:8000/docs to try /query and /detect-appliance from Swagger UI
```

### 3. Frontend
```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# Visit http://localhost:8501
```

## Environment Variables

**backend/.env**

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_PATH` | `data/vector_store` | Path to the persisted Chroma store |
| `COLLECTION_NAME` | `appliance_manuals` | Chroma collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |
| `OLLAMA_MODEL` | `llama3.2` | Local Ollama model to use for generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `CORS_ORIGINS` | `["http://localhost:8501"]` | Allowed frontend origins |
| `RETRIEVAL_K` | `3` | Default number of chunks to retrieve |

**frontend/.env**

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Backend base URL |

## API Reference

### `GET /health`
Returns `{"status": "ok"}` if the service is up.

### `POST /query`
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I descale my coffee maker?"}'
```
Response:
```json
{
  "answer": "Descale every 2-3 months using a 1:1 water-and-vinegar solution... [source: coffee_maker_manual.md]",
  "sources": ["coffee_maker_manual.md"]
}
```

Optional fields: `appliance_hint` (a manual filename to restrict retrieval to, e.g. from
`/detect-appliance`), `k` (number of chunks to retrieve, default 3).

### `POST /detect-appliance`
```bash
curl -X POST http://localhost:8000/detect-appliance \
  -F "image=@/path/to/photo.jpg"
```
Response:
```json
{"appliance": "microwave", "confidence": 0.87, "manual_source": "microwave_manual.md"}
```

## Evaluation Results

See `notebooks/rag_pipeline.ipynb` Section 2.6 for the full table. Summary: 10/10 sample
questions retrieved a chunk from the correct manual, and answers were grounded in the
retrieved context (no hallucinated facts observed) once the prompt was constrained to
context-only answers with an explicit "say you don't know" instruction. The main failure mode
was retrieving a *correct-manual-but-wrong-section* chunk, mitigated with header-aware chunking
and `k=3` retrieval — see the notebook for full analysis.

## Screenshots

_Add screenshots of the running Streamlit app (chat view, and the image-upload/detection
sidebar) here before submitting._

## Common Pitfalls Avoided

- `.env`, `.venv/`, and YOLO weight files are excluded via `.gitignore`.
- The frontend reads `API_BASE_URL` from an environment variable — never hard-coded.
- The system prompt explicitly forbids answering from the LLM's own knowledge outside the
  retrieved context, to keep answers grounded.
- Tested on 10 sample questions (see notebook), not just 1–2.
- Notebook is designed to run top-to-bottom via **Kernel → Restart & Run All**.

# Zania | QA Bot — starter

This repository contains a FastAPI service that accepts:
- a JSON file containing a list of questions (array of strings)
- a document file (PDF or JSON)

It uses LangChain + OpenAI (gpt-4o-mini) to create embeddings and a retrieval-augmented QA flow.

## Setup

1. Clone repo.

2. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
uvicorn main:app --reload --port 8080


curl -X POST "http://localhost:8080/qa" \
  -F "questions_file=@./sample_inputs/example_questions.json;type=application/json" \
  -F "document_file=@./sample_inputs/example_input.txt;type=text/plain"
```
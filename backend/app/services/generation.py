import ollama
from app.core.config import settings


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    return f"""You are a helpful kitchen appliance assistant. Answer the question using
ONLY the context below. Cite the source file for each fact you use. If the answer is not
contained in the context, say you don't know rather than guessing.

Context:
{context}

Question: {question}

Answer (with citations like [source: filename]):"""


def generate_answer(question: str, hits: list[dict]) -> str:
    if not hits:
        return "I don't have enough information in the manuals to answer that."

    prompt = build_prompt(question, hits)
    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

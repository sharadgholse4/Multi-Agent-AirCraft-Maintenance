import os
import numpy as np
import time
import ollama

from pdf_tools import MANUAL_REGISTRY, embeddings  # reuse the same bge-base embedder

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

# Configure persistent Ollama client with a 5-minute timeout so requests never drop
_ollama_client = ollama.Client(timeout=300.0)


# ---------------------------------------------------------------------------
# ROUTING: embedding-based (accurate, deterministic, no LLM involved).
# Lazily computed so backend starts instantly without waiting at import time.
# ---------------------------------------------------------------------------
_manual_keys = list(MANUAL_REGISTRY.keys())
_manual_descriptions = [MANUAL_REGISTRY[k]["description"] for k in _manual_keys]
_description_embeddings = None


def _get_description_embeddings():
    global _description_embeddings
    if _description_embeddings is None:
        print("[LOG] Computing manual description embeddings for query routing...", flush=True)
        _description_embeddings = embeddings.embed_documents(_manual_descriptions)
        print("[LOG] Description embeddings cached successfully.", flush=True)
    return _description_embeddings


def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def route_query(query: str) -> str:
    desc_embeddings = _get_description_embeddings()
    query_embedding = embeddings.embed_query(query)
    scores = [
        (_manual_keys[i], _cosine_sim(query_embedding, desc_embeddings[i]))
        for i in range(len(_manual_keys))
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_key, best_score = scores[0]
    print(f"[LOG] Routing scores: {[(k, round(s, 4)) for k, s in scores]}")
    print(f"[LOG] Selected manual: '{best_key}' (score={best_score:.4f})")
    return best_key


# ---------------------------------------------------------------------------
# ANSWERING: via Ollama (qwen3.5:9b), tuned for fast, grounded answers.
# ---------------------------------------------------------------------------
def _generate(prompt: str, format: str = None) -> str:
    import re
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)

    # Options optimized for speed, GPU offloading, and preventing runaway token generation
    options = {
        "temperature": 0.3,
        "top_p": 0.9,
        "repeat_penalty": 1.15,
        "num_predict": 384 if format == "json" else int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "3072")),
    }
    if os.getenv("OLLAMA_NUM_GPU") is not None:
        options["num_gpu"] = int(os.getenv("OLLAMA_NUM_GPU"))

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": options,
        "think": False,
    }
    if format:
        kwargs["format"] = format

    response = None
    for attempt in range(2):
        try:
            try:
                response = _ollama_client.chat(**kwargs)
            except (TypeError, ValueError):
                # Fallback if the ollama python client version doesn't accept think kwarg
                kwargs.pop("think", None)
                response = _ollama_client.chat(**kwargs)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "not found" in err_str:
                try:
                    installed = [m.model for m in _ollama_client.list().models]
                    if installed:
                        fallback_model = installed[0]
                        print(f"[WARN] Model '{model}' not found in Ollama. Falling back to installed model '{fallback_model}'.")
                        kwargs["model"] = fallback_model
                        response = _ollama_client.chat(**kwargs)
                        break
                except Exception:
                    raise e
            if attempt == 0 and ("disconnected" in err_str or "connection" in err_str or "timeout" in err_str):
                print(f"[WARN] Ollama request attempt 1 failed ({e}). Retrying once after 2s...")
                time.sleep(2)
                continue
            raise e

    if hasattr(response, "message") and hasattr(response.message, "content"):
        content = response.message.content or ""
    elif isinstance(response, dict) and "message" in response:
        content = response["message"].get("content", "") or ""
    else:
        content = str(response)

    content = content.strip()
    # Strip any <think> reasoning tags if Qwen reasoning is active
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content



def answer_query(query: str, context: str, conversation_context: str = "No prior conversation history.") -> str:
    """
    Given retrieved chunks (context), the original query, and prior
    conversation context (if any), asks the model to produce a well-written,
    grounded paragraph answer.
    """
    prompt = (
        "You are an experienced aircraft maintenance instructor explaining a "
        "procedure to a technician. Using ONLY the manual context below, write "
        "a clear, well-organized answer in flowing paragraphs (not bullet points "
        "or a numbered list unless the source itself describes numbered steps). "
        "Explain the reasoning and any important safety notes, not just a bare "
        "fact. If the context doesn't contain enough information to answer "
        "fully, say so honestly and explain what is missing.\n\n"
        f"Prior conversation context (may be empty):\n{conversation_context}\n\n"
        "If the current question refers back to something discussed earlier "
        "(e.g. 'that same part', 'the procedure you mentioned'), use that "
        "context — otherwise treat this as a fresh question.\n\n"
        f"Manual context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (in the markdown form )"
    )
    print( " Context of the prompt - " , conversation_context)
    return _generate(prompt)
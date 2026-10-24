import json
import re

from gemma_router import _generate


def _extract_json_object(text: str) -> str:
    """
    Extracts the first balanced JSON object {...} handling nested structures,
    strings with escaped quotes, and ignoring trailing text/hallucinations.
    """
    start = text.find('{')
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return text[start:]


def call_llm_json(prompt: str, default: dict) -> dict:
    """
    Calls the local LLM (Ollama) asking for STRICT JSON output only,
    then parses it safely. Falls back to `default` if parsing fails,
    so a single bad generation never crashes the pipeline.
    """
    strict_prompt = (
        f"{prompt}\n\n"
        "Respond with ONLY a valid JSON object matching the requested schema. "
        "No markdown fences, no backticks, and no text before or after the JSON."
    )

    # 1. Ask Ollama with native format='json' (forces valid grammar at token level)
    try:
        raw = _generate(strict_prompt, format="json")
    except Exception as e:
        print(f"[WARN] _generate with format='json' failed ({e}), falling back without format.")
        raw = _generate(strict_prompt)

    # 2. Strip any <think> tags if model is a thinking/reasoning model
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 3. Strip any markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 4. Extract balanced outermost JSON object { ... }
    json_candidate = _extract_json_object(cleaned)
    if not json_candidate.strip().startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            json_candidate = match.group(0)
        else:
            print(f"[WARN] LLM did not return JSON, using default. Raw: {raw[:200]}")
            return default

    # 5. Try standard json.loads
    try:
        parsed = json.loads(json_candidate)
        return {**default, **parsed}
    except json.JSONDecodeError:
        pass

    # 6. Try json.loads with strict=False (allows unescaped control chars / literal newlines)
    try:
        parsed = json.loads(json_candidate, strict=False)
        return {**default, **parsed}
    except json.JSONDecodeError:
        pass

    # 7. Try fixing trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r",\s*([\}\]])", r"\1", json_candidate)
        parsed = json.loads(fixed, strict=False)
        return {**default, **parsed}
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed ({e}), using default. Raw: {raw[:300]}")
        return default
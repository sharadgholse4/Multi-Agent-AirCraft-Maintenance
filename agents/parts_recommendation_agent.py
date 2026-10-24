from agents.common import call_llm_json
from pdf_tools import search_manuals


def recommend_parts(fault_diagnosis: dict, aircraft_model: str, engine_model: str, manual_key: str) -> dict:
    context = search_manuals.invoke({
        "query": f"replacement part number for {fault_diagnosis.get('affected_component', '')}",
        "manual_key": manual_key,
    })

    prompt = (
        "You are a Parts Recommendation Agent. Based on the fault diagnosis, "
        "the aircraft/engine model, and the manual context below, suggest the "
        "replacement part(s) needed. If the manual context does NOT contain an "
        "explicit part number, say so honestly instead of inventing one.\n\n"
        f"Fault diagnosis:\n{fault_diagnosis}\n\n"
        f"Aircraft model: {aircraft_model}\n"
        f"Engine model: {engine_model}\n\n"
        f"Manual context:\n{context}\n\n"
        "Provide a clear, concise technical description:\n"
        "- For \"part_description\", write a 5-6 line description detailing the part's function, "
        "justification for replacement based on the fault, and key installation/torque guidelines.\n\n"
        "Return a JSON object with exactly these keys: "
        "\"part_number\" (string, use \"Not found in available manuals\" if "
        "not present in context), "
        "\"part_description\" (string), "
        "\"quantity_required\" (integer), "
        "\"alternative_part_numbers\" (list of strings, empty list if none)."
    )

    default = {
        "part_number": "Not found in available manuals",
        "part_description": fault_diagnosis.get("affected_component", "Unknown component"),
        "quantity_required": 1,
        "alternative_part_numbers": [],
    }

    result = call_llm_json(prompt, default)
    alts = result.get("alternative_part_numbers")
    if isinstance(alts, str):
        cleaned = alts.strip().strip("[]").strip()
        if cleaned.lower() in ["none", "n/a", "no alternatives", ""]:
            result["alternative_part_numbers"] = []
        else:
            result["alternative_part_numbers"] = [x.strip().strip("'\"") for x in cleaned.split(",") if x.strip()]
    elif not isinstance(alts, list):
        result["alternative_part_numbers"] = []

    print(f"[PartsRecommendationAgent] {result}")
    return result
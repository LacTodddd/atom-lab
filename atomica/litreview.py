"""P5 literature reviewer: validate a structured boolean prediction, provide a paper-blind
baseline, a non-LLM trend heuristic, and (later) the LLM reviewer; score accuracy/precision/recall.
The reviewer proposes; the harness checks against ground truth."""

PREDICT_TOOL = {
    "name": "predict_gap",
    "description": "Predict whether the unexplored region contains a structure with lower energy "
                   "than the study's reported best.",
    "input_schema": {
        "type": "object",
        "properties": {
            "better_in_gap": {"type": "boolean",
                              "description": "true if a better (lower-energy) structure likely exists "
                                             "in the unexplored region"},
        },
        "required": ["better_in_gap"],
        "additionalProperties": False,
    },
    "strict": True,
}

def validate_prediction(raw):
    if not isinstance(raw, dict):
        raise ValueError(f"prediction not a dict: {raw!r}")
    v = raw.get("better_in_gap")
    if not isinstance(v, bool):
        raise ValueError(f"better_in_gap not a bool: {v!r}")
    return {"better_in_gap": v}

def baseline_reviewer(paper):
    return {"better_in_gap": False}

def heuristic_reviewer(paper):
    t = paper["boundary_trend"]
    h = len(t) // 2
    far, near = t[:h], t[h:]
    return {"better_in_gap": bool(min(near) < min(far))}

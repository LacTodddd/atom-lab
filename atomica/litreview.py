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

def review_one(paper, reviewer):
    # cheat-proof: hand the reviewer only observable summary fields — never the label.
    view = {k: v for k, v in paper.items() if k != "better_in_gap"}
    fallback = False
    try:
        pred = validate_prediction(reviewer(view))
    except ValueError:
        pred = {"better_in_gap": False}
        fallback = True
    predicted = pred["better_in_gap"]
    return {"predicted": predicted, "correct": predicted == paper["better_in_gap"],
            "fallback": fallback}

def review_batch(papers, reviewer):
    return [review_one(p, reviewer) for p in papers]

def score(papers, reviews):
    y = [p["better_in_gap"] for p in papers]
    yhat = [r["predicted"] for r in reviews]
    n = len(papers)
    correct = int(sum(a == b for a, b in zip(y, yhat)))
    tp = int(sum(h and t for h, t in zip(yhat, y)))
    pred_pos = int(sum(yhat))
    actual_pos = int(sum(y))
    return {"accuracy": correct / n if n else 0.0,
            "precision": tp / pred_pos if pred_pos else 0.0,
            "recall": tp / actual_pos if actual_pos else 0.0,
            "n": n, "base_rate_better": actual_pos / n if n else 0.0}

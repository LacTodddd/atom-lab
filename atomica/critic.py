"""P4 critic: validate a structured verdict, apply a within-sample stratified
control (sign-flip => reject), and score arms. LLM proposes; harness decides."""
from atomica.critic_world import FEATURE_NAMES, stratified_effect, _sign

VALID_VERDICTS = ("supported", "confounded")

CRITIC_TOOL = {
    "name": "critique_claim",
    "description": "Judge whether a claim that a feature drives energy is supported or confounded "
                   "by another feature, and if confounded, name the confounder.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VALID_VERDICTS),
                        "description": "supported if the claim holds; confounded if another feature explains it"},
            "confounder": {"type": "string", "enum": FEATURE_NAMES,
                           "description": "the feature confounding the claim (required when verdict is confounded; not the target)"},
        },
        "required": ["verdict"],
        "additionalProperties": False,
    },
    "strict": True,
}

def validate_critique(raw, target):
    if not isinstance(raw, dict):
        raise ValueError(f"critique not a dict: {raw!r}")
    verdict = raw.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"bad verdict: {verdict!r}")
    if verdict == "supported":
        return {"verdict": "supported", "confounder": None}
    conf = raw.get("confounder")
    valid = [f for f in FEATURE_NAMES if f != target]
    if conf not in valid:
        raise ValueError(f"bad confounder {conf!r}; must be one of {valid}")
    return {"verdict": "confounded", "confounder": conf}

def apply_control(claim, confounder):
    """Stratify the claim's own sample on `confounder`; sign-flip => reject.
    Returns True if the claim is ACCEPTED (survives), False if REJECTED."""
    if confounder is None or confounder == claim["target"]:
        return True
    s = claim["sample"]
    controlled = _sign(stratified_effect(s[claim["target"]], s[confounder], s["energy"]))
    if controlled != 0 and controlled != claim["claim_sign"]:
        return False
    return True

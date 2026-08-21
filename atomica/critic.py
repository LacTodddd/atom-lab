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

def accept_all_critic(claim):
    return {"verdict": "supported"}

def random_critic(rng, p_confounded=1.0):
    def critic(claim):
        if rng.random() < p_confounded:
            others = [f for f in FEATURE_NAMES if f != claim["target"]]
            return {"verdict": "confounded", "confounder": str(rng.choice(others))}
        return {"verdict": "supported"}
    return critic

def review_one(claim, critic):
    fallback = False
    try:
        crit = validate_critique(critic(claim), claim["target"])
    except ValueError:
        crit = {"verdict": "supported", "confounder": None}
        fallback = True
    accepted = True if crit["verdict"] == "supported" else apply_control(claim, crit["confounder"])
    return {"accepted": bool(accepted), "verdict": crit["verdict"],
            "confounder": crit["confounder"], "fallback": fallback,
            "named_true_confounder": crit["confounder"] == claim["confounder_true"]}

def review_batch(claims, critic):
    return [review_one(c, critic) for c in claims]

def score(claims, reviews):
    acc = [r["accepted"] for r in reviews]
    accepted_true = int(sum(a and c["label_is_true"] for a, c in zip(acc, claims)))
    accepted_false = int(sum(a and not c["label_is_true"] for a, c in zip(acc, claims)))
    n_accepted = accepted_true + accepted_false
    n_true = int(sum(c["label_is_true"] for c in claims))
    n = len(claims)
    return {"fdr": accepted_false / n_accepted if n_accepted else 0.0,
            "retention": accepted_true / n_true if n_true else 0.0,
            "n_accepted": n_accepted, "accepted_false": accepted_false,
            "accepted_true": accepted_true, "n_claims": n, "n_true": n_true,
            "base_rate_false": (n - n_true) / n if n else 0.0}

MODEL = "claude-sonnet-5"

def build_prompt(claim):
    tgt = claim["target"]
    direction = "lower" if claim["claim_sign"] < 0 else "higher"
    others = [f for f in FEATURE_NAMES if f != tgt]
    s = claim["sample"]
    n = len(s["energy"])
    lines = [
        "A scientist studied Cu-Au orderings on a fixed 12-site lattice. Each config has three",
        f"integer structural features {FEATURE_NAMES} and a MACE energy (lower = more stable).",
        f"CLAIM: higher {tgt} is associated with {direction} energy.",
        f"The claim was drawn from this sample of {n} configs. It may be CONFOUNDED: another",
        f"feature ({' or '.join(others)}) may covary with {tgt} and actually drive the energy.",
        "Sample (one row per config):",
        "  " + "  ".join(FEATURE_NAMES) + "  energy",
    ]
    for i in range(n):
        row = "  ".join(f"{s[name][i]:g}" for name in FEATURE_NAMES)
        lines.append(f"  {row}  {s['energy'][i]:.3f}")
    lines.append(f"Call critique_claim: verdict 'supported' if {tgt} genuinely drives energy, or "
                 f"'confounded' naming the confounder among {others}.")
    return "\n".join(lines)

def llm_critic(client, model=MODEL):
    def critic(claim):
        resp = client.messages.create(
            model=model, max_tokens=512,
            tools=[CRITIC_TOOL], tool_choice={"type": "tool", "name": "critique_claim"},
            messages=[{"role": "user", "content": build_prompt(claim)}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "critique_claim":
                return block.input
        raise ValueError("no critique_claim tool_use in response")
    return critic

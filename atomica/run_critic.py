"""ATOMICA P4 CLI: does an LLM critic cut false discoveries vs random vs none? (Cu-Au)."""
import argparse, json
from pathlib import Path
import numpy as np
from atomica.critic_world import build_world, generate_claims
from atomica.critic import (
    accept_all_critic, random_critic, llm_critic, review_batch, score, MODEL,
)

def make_llm_critic(model):
    """Return an llm_critic bound to a real client, or None if no credential/SDK."""
    try:
        import anthropic
        return llm_critic(anthropic.Anthropic(), model=model)
    except Exception as e:                      # missing key/sdk -> skip the LLM arm
        print(f"[run_critic] LLM arm disabled: {e}")
        return None

def main(argv=None):
    p = argparse.ArgumentParser(description="ATOMICA P4 LLM-critic false-discovery benchmark (Cu-Au)")
    p.add_argument("--n-claims", type=int, default=60)
    p.add_argument("--n", type=int, default=40, help="scientist sample size per claim")
    p.add_argument("--strength", type=float, default=2.0, help="confounding strength")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    world = build_world(cache_path=str(Path(a.out) / "alloy_world.json"))
    claims = generate_claims(world, a.n_claims, n=a.n, strength=a.strength, seed0=a.seed)

    arms = {}
    arms["none"] = score(claims, review_batch(claims, accept_all_critic))
    arms["random"] = score(claims, review_batch(claims, random_critic(np.random.default_rng(1000 + a.seed))))

    llm = make_llm_critic(a.model)
    if llm is not None:
        import anthropic  # already imported successfully inside make_llm_critic
        try:
            # anthropic.Anthropic() doesn't validate credentials until the first
            # request (it defers to header-building), so construction can succeed
            # with no key configured. AnthropicError covers real API-level auth/
            # connection failures; the no-credential case specifically raises a
            # bare TypeError from header validation (not an AnthropicError) with
            # a stable "Could not resolve authentication method" message, so it's
            # matched by content rather than type. Any other error (a real bug in
            # the LLM path) propagates instead of being silently mislabeled.
            arms["llm"] = score(claims, review_batch(claims, llm))
        except anthropic.AnthropicError as e:
            print(f"[run_critic] LLM arm disabled: {e}")
        except TypeError as e:
            if "authentication method" not in str(e):
                raise
            print(f"[run_critic] LLM arm disabled: {e}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "critic_report.json").write_text(json.dumps(
        {"arms": arms,
         "config": {"n_claims": a.n_claims, "n": a.n, "strength": a.strength,
                    "seed": a.seed, "model": a.model}}, indent=2))
    print(json.dumps({k: {"fdr": round(v["fdr"], 3), "retention": round(v["retention"], 3)}
                      for k, v in arms.items()}, indent=2))

if __name__ == "__main__":
    main()

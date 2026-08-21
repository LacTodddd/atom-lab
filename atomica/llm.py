"""Shared Anthropic plumbing for the credential-gated LLM arms (P3 tuner, P4 critic, P5 reviewer).

Each phase's CLI has an optional LLM arm that must degrade cleanly when the user has no
credential: the arm is skipped and the non-LLM arms still run. The credential is always the
user's own (ANTHROPIC_API_KEY or an `ant auth login` profile); no key lives in this repo.
"""

def make_llm_arm(build, model, tag):
    """Return `build(client, model=model)` bound to a real client, or None if no SDK/credential."""
    try:
        import anthropic
        return build(anthropic.Anthropic(), model=model)
    except Exception as e:                     # missing sdk/key -> skip the LLM arm
        print(f"[{tag}] LLM arm disabled: {e}")
        return None

def run_llm_arm(tag, body):
    """Run `body()` and return its result, or None if it fails on a credential/auth error.

    The SDK resolves credentials lazily at the first request, so a missing key surfaces here
    rather than at client construction. Only auth failures are swallowed: any other exception
    propagates, so a real bug in the LLM path is never mislabelled "disabled".
    """
    import anthropic
    try:
        return body()
    except anthropic.AnthropicError as e:
        print(f"[{tag}] LLM arm disabled: {e}")
    except TypeError as e:
        if "authentication method" not in str(e):
            raise
        print(f"[{tag}] LLM arm disabled: {e}")
    return None

def call_strict_tool(client, model, tool, prompt, max_tokens=512):
    """One forced strict-tool call. Returns the tool_use block's raw `.input` dict (unvalidated —
    callers validate, so a malformed model reply stays a data problem, not an exception)."""
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
            return block.input
    raise ValueError(f"no {tool['name']} tool_use in response")

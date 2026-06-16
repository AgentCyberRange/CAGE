"""Tests for harness conversation reconstruction (cage/proxy/conversations.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cage.proxy.conversations import reconstruct_forest

ROOT_SYS = "You are Claude Code, the main agent loop. " * 8
SUBAGENT_SYS = "You are a subagent launched via the Task tool. " * 8
SUB2_SYS = "You are a nested exploration subagent. " * 8
BG_SYS = "Generate a concise title for this conversation."


def _entry(idx, *, system, messages, blocks=None, in_tok=2000, out_tok=200, reasoning=0):
    """Build a synthetic proxy.jsonl entry (Anthropic shape)."""
    return {
        "request_id": f"req-{idx:04d}",
        "trial_id": "t",
        "ts_ms": 1_000 + idx,
        "status": "success",
        "anthropic_request": {
            "model": "m",
            "system": [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.{idx}\n" + system}],
            "messages": messages,
        },
        "anthropic_response": {
            "model": "m",
            "content": blocks or [{"type": "text", "text": f"step {idx}"}],
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok, "reasoning_tokens": reasoning},
        },
    }


def _user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(blocks):
    return {"role": "assistant", "content": blocks}


def _tool_result(tid, content):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}]}


def _grow(messages, idx):
    """Append an assistant+user turn so the history grows monotonically."""
    return messages + [_assistant([{"type": "text", "text": f"a{idx}"}]), _user(f"continue {idx}")]


# --------------------------------------------------------------------------- #

def test_root_only_linear():
    msgs = [_user("solve the task <system-reminder>volatile</system-reminder>")]
    entries = []
    for i in range(5):
        entries.append(_entry(i, system=ROOT_SYS, messages=msgs, in_tok=2000 + i * 1000))
        msgs = _grow(msgs, i)
    forest = reconstruct_forest(entries)
    assert len(forest.conversations) == 1
    root = forest.conversations[0]
    assert root.kind == "root"
    assert root.call_indices == [0, 1, 2, 3, 4]
    assert forest.root_rounds == 5
    assert forest.structure["num_subagents"] == 0
    assert forest.structure["num_compactions"] == 0


def test_single_subagent_linked():
    # root call 0 spawns Task(A); subagent runs 3 calls; root resumes at call 4.
    root_msgs = [_user("main task")]
    task_block = {"type": "tool_use", "id": "toolu_A", "name": "Task",
                  "input": {"prompt": "explore the auth module thoroughly", "subagent_type": "general-purpose"}}
    entries = [_entry(0, system=ROOT_SYS, messages=root_msgs, blocks=[task_block], in_tok=3000)]

    sub_msgs = [_user("explore the auth module thoroughly")]
    for i in range(1, 4):
        entries.append(_entry(i, system=SUBAGENT_SYS, messages=sub_msgs, in_tok=1800 + i * 500))
        sub_msgs = _grow(sub_msgs, i)

    # root resumes: history grows with the tool_result for toolu_A
    root_msgs2 = root_msgs + [_assistant([task_block]), _tool_result("toolu_A", "auth findings")]
    entries.append(_entry(4, system=ROOT_SYS, messages=root_msgs2, in_tok=4000))

    forest = reconstruct_forest(entries)
    kinds = {c.id: c.kind for c in forest.conversations}
    assert sorted(kinds.values()) == ["root", "subagent"]
    root = next(c for c in forest.conversations if c.kind == "root")
    sub = next(c for c in forest.conversations if c.kind == "subagent")
    assert root.call_indices == [0, 4]
    assert sub.call_indices == [1, 2, 3]
    assert sub.parent_id == root.id
    assert sub.subagent_type == "general-purpose"
    assert sub.spawned_by["tool_use_id"] == "toolu_A"
    # The spawn prompt (the context handed into the subagent) is recorded so the
    # UI can show what the parent delegated.
    assert "explore the auth module" in sub.spawned_by["prompt"]
    # The return edge: the subagent's result folds back into the parent at the
    # step where toolu_A reappears as a tool_result (root resumes at call 4).
    assert sub.returns_at == 4
    assert sub.depth == 1
    assert forest.structure["num_subagents"] == 1
    # root agent rounds exclude the subagent's calls
    assert forest.root_rounds == 2


def test_parallel_subagents():
    task_a = {"type": "tool_use", "id": "toolu_A", "name": "Task",
              "input": {"prompt": "find all SQL sinks in the repo", "subagent_type": "general-purpose"}}
    task_b = {"type": "tool_use", "id": "toolu_B", "name": "Task",
              "input": {"prompt": "audit the crypto usage everywhere", "subagent_type": "general-purpose"}}
    root_msgs = [_user("security review")]
    entries = [_entry(0, system=ROOT_SYS, messages=root_msgs, blocks=[task_a, task_b], in_tok=3000)]

    a_msgs = [_user("find all SQL sinks in the repo")]
    b_msgs = [_user("audit the crypto usage everywhere")]
    # interleave A and B
    for i in range(1, 4):
        entries.append(_entry(2 * i - 1, system=SUBAGENT_SYS, messages=a_msgs, in_tok=1700 + i * 400))
        entries.append(_entry(2 * i, system=SUBAGENT_SYS, messages=b_msgs, in_tok=1700 + i * 400))
        a_msgs = _grow(a_msgs, i)
        b_msgs = _grow(b_msgs, i)

    forest = reconstruct_forest(entries)
    subs = [c for c in forest.conversations if c.kind == "subagent"]
    assert len(subs) == 2
    ids = {s.spawned_by["tool_use_id"] for s in subs}
    assert ids == {"toolu_A", "toolu_B"}
    assert all(s.parent_id is not None for s in subs)
    assert forest.structure["num_subagents"] == 2


def test_nested_subagent():
    task_a = {"type": "tool_use", "id": "toolu_A", "name": "Task",
              "input": {"prompt": "investigate the parser deeply", "subagent_type": "general-purpose"}}
    nested = {"type": "tool_use", "id": "toolu_N", "name": "Task",
              "input": {"prompt": "read every file in src/parser", "subagent_type": "explore"}}
    entries = [_entry(0, system=ROOT_SYS, messages=[_user("root task")], blocks=[task_a], in_tok=3000)]
    # subagent A: call 1 spawns nested
    a_msgs = [_user("investigate the parser deeply")]
    entries.append(_entry(1, system=SUBAGENT_SYS, messages=a_msgs, blocks=[nested], in_tok=2000))
    # nested subagent runs
    n_msgs = [_user("read every file in src/parser")]
    for i in range(2, 4):
        entries.append(_entry(i, system=SUB2_SYS, messages=n_msgs, in_tok=1500 + i * 300))
        n_msgs = _grow(n_msgs, i)
    # subagent A resumes
    a_msgs2 = a_msgs + [_assistant([nested]), _tool_result("toolu_N", "parser map")]
    entries.append(_entry(4, system=SUBAGENT_SYS, messages=a_msgs2, in_tok=2500))

    forest = reconstruct_forest(entries)
    by_type = {c.subagent_type: c for c in forest.conversations if c.kind == "subagent"}
    assert "general-purpose" in by_type and "explore" in by_type
    a = by_type["general-purpose"]
    n = by_type["explore"]
    assert a.parent_id == next(c.id for c in forest.conversations if c.kind == "root")
    assert n.parent_id == a.id
    assert n.depth == 2
    assert forest.structure["max_depth"] == 2


def test_compaction_call_detected_and_continuation_stays_root():
    # Compaction is a real model call: the harness sends the full history plus a
    # "summarize the conversation" instruction (index 4), the model returns the
    # summary, and the next call (index 5) resumes from that summary — its first
    # user message opens with the continuation marker.
    msgs = [_user("long task")]
    entries = []
    for i in range(4):
        entries.append(_entry(i, system=ROOT_SYS, messages=msgs, in_tok=10000 + i * 5000))
        msgs = _grow(msgs, i)
    # index 4: the compaction CALL — full history + a summarize instruction; the
    # response is the summary. Its first user message is still the original task,
    # so it is NOT itself a continuation.
    compaction_msgs = msgs + [_user("Your task is to create a detailed summary of the conversation so far.")]
    entries.append(_entry(4, system=ROOT_SYS, messages=compaction_msgs, in_tok=40000,
                          blocks=[{"type": "text", "text": "This session is being continued..."}]))
    # index 5+: the continuation resumes from the summary (marker as first message).
    cont = [_user("This session is being continued from a previous conversation that ran "
                  "out of context. Summary: prior work on the task. Continue from here.")]
    entries.append(_entry(5, system=ROOT_SYS, messages=cont, in_tok=6000))
    cont = _grow(cont, 5)
    entries.append(_entry(6, system=ROOT_SYS, messages=cont, in_tok=9000))

    forest = reconstruct_forest(entries)
    assert len(forest.conversations) == 1
    root = forest.conversations[0]
    assert root.kind == "root"
    assert root.call_indices == [0, 1, 2, 3, 4, 5, 6]
    assert forest.structure["num_compactions"] == 1
    assert root.compaction_calls == [4]   # the summary-generating call
    assert root.compaction_at == [5]      # the continuation that resumes from it
    # the compaction call itself is not an agent decision round
    assert forest.root_rounds == 6


def test_big_tool_result_spike_is_not_mistaken_for_compaction():
    # A single huge tool_result inflates one call's input, then the next call's
    # input collapses — but there is NO continuation marker, so it must NOT be
    # counted as a compaction (the old token-collapse heuristic mis-flagged this).
    msgs = [_user("task")]
    entries = []
    for i in range(3):
        entries.append(_entry(i, system=ROOT_SYS, messages=msgs, in_tok=8000 + i * 1000))
        msgs = _grow(msgs, i)
    # index 3: a giant tool_result balloons input to a peak.
    big = msgs + [_assistant([{"type": "text", "text": "reading"}]),
                  _tool_result("tBIG", "X" * 200000)]
    entries.append(_entry(3, system=ROOT_SYS, messages=big, in_tok=90000))
    # index 4: the giant result ages out; input collapses, but it is normal work.
    msgs = _grow(msgs, 3)
    entries.append(_entry(4, system=ROOT_SYS, messages=msgs, in_tok=12000))

    forest = reconstruct_forest(entries)
    assert forest.structure["num_compactions"] == 0


def test_background_calls_demoted():
    msgs = [_user("task")]
    entries = []
    for i in range(3):
        entries.append(_entry(i * 2, system=ROOT_SYS, messages=msgs, in_tok=5000 + i * 1000))
        # interleave a tiny background call (title-gen class)
        entries.append(_entry(i * 2 + 1, system=BG_SYS, messages=[_user("title?")], in_tok=200))
        msgs = _grow(msgs, i)
    forest = reconstruct_forest(entries)
    root = next(c for c in forest.conversations if c.kind == "root")
    bg = [c for c in forest.conversations if c.kind == "background"]
    assert root.call_indices == [0, 2, 4]
    assert forest.structure["num_background_calls"] == 3
    assert len(bg) == 3
    # root rounds count only the real agent decisions, not background blips
    assert forest.root_rounds == 3


def test_real_fixture_compaction_and_background():
    fixture = Path(__file__).parent / "fixtures" / "harness" / "real_compaction_background.jsonl"
    if not fixture.exists():
        pytest.skip("real fixture not staged")
    entries = []
    for line in fixture.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("status") == "success":
            entries.append(obj)
    forest = reconstruct_forest(entries)
    # The real 117-call trial: root delegates to an ``Explore`` subagent (linked
    # via its Task/Agent tool_use id) plus tiny background calls.
    root = next((c for c in forest.conversations if c.kind == "root"), None)
    assert root is not None and root.call_indices[0] == 0
    explore = [c for c in forest.conversations if c.subagent_type == "Explore"]
    assert len(explore) == 1
    assert explore[0].parent_id == root.id
    assert explore[0].depth == 1
    assert explore[0].spawned_by and explore[0].spawned_by["tool_use_id"]
    # Six tiny background/auxiliary calls are demoted out of the agent rounds.
    assert forest.structure["num_background_calls"] == 6
    # The Explore subagent compacts exactly once: a call carrying the "create a
    # detailed summary" instruction, after which every later call carries the
    # summary. Detection keys on that instruction, not a token collapse.
    assert forest.structure["num_compactions"] == 1
    assert explore[0].compaction_calls and explore[0].compaction_at
    # root agent rounds (5) are a tiny fraction of the 117 raw successful calls
    assert forest.root_rounds == 5
    assert forest.root_rounds < len(entries)

"""Reconstruct harness conversation structure from a flat ``proxy.jsonl`` stream.

``proxy.jsonl`` is the immutable audit log: one entry per LLM call, in arrival
order, with full request/response bodies. A modern coding harness is **not** one
linear conversation, though — it compacts context, spawns subagents (the ``Task``
tool), and makes background/auxiliary calls. All of those hit the same in-container
proxy and land interleaved in the one flat stream.

This module derives the structure the flat stream flattened away, *without*
touching capture (the proxy stays httpx-only and dumb). Everything needed is
already recorded, because every request carries its full message history and every
response carries its tool-use ids:

  * **Grouping** — calls are chained into conversations by a stable anchor
    (system body with volatile billing/reminder noise stripped + the first few
    normalized messages) plus monotonic message growth. Distinct subagents have
    distinct anchors; a background call is its own one-call anchor.
  * **Compaction** — within a conversation, the point where the history collapses
    (message count + input tokens drop sharply) is a compaction boundary; the
    continuation stays the *same* conversation.
  * **Linking** — a subagent is a ``Task`` tool call whose result is produced by a
    nested conversation. We pair the parent's ``Task`` tool_use (id + prompt) to the
    child conversation (its first user message ≈ the Task prompt; its final text ≈
    the tool_result returned to the parent). This reuses the exact tool_use↔result
    pairing the trajectory renderer already does.

The output is a :class:`ConversationForest`: a forest of :class:`Conversation`
nodes plus a ``call_to_conv`` index and a ``structure`` summary, including
**root-only** token and round totals so downstream round-counting / cost metrics
stop conflating subagent + background + compaction churn with agent decisions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# Volatile bits that change on every call and must be stripped before a system
# prompt can identify a conversation lineage.
_BILLING_RE = re.compile(r"x-anthropic-billing-header:[^\n]*", re.IGNORECASE)
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

# A call counts as "background/auxiliary" when it is a single-message, cheap,
# one-shot request that never grows into a multi-call conversation.
_BACKGROUND_MAX_MSGS = 1
_BACKGROUND_MAX_INPUT_TOKENS = 1500

# Compaction is a real model call: the harness sends the full history plus a
# "create a detailed summary of the conversation" instruction (the COMPACTION
# CALL), the model returns the summary, and subsequent calls resume carrying that
# summary as a text block that opens with the CONTINUATION marker. We detect both
# directly from request text — far more reliable than an input-token collapse,
# which a single huge tool_result also triggers and a token heuristic mis-flags.
_COMPACTION_SUMMARY_PROMPT = "create a detailed summary of the conversation"
_COMPACTION_MARKER = "this session is being continued from a previous conversation"

# How many leading messages anchor a conversation lineage. The opening turns
# (system + task) do not mutate; trailing reminders sometimes do, so we anchor on
# the head rather than the whole history.
_ANCHOR_HEAD_MSGS = 3

# Tool names a harness uses to spawn a subagent. Claude Code's canonical name is
# ``Task``; some builds expose it as ``Agent`` (observed driving GLM through the
# OpenAI-translated proxy). Both carry ``subagent_type`` + ``prompt``.
_SUBAGENT_TOOL_NAMES = {"task", "agent"}


@dataclass
class Call:
    """One successful model call, with the fields reconstruction needs."""

    index: int
    request_id: str
    ts_ms: int | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    n_msgs: int
    stable_system: str
    msg_head: tuple[str, ...]
    first_user: str
    final_text: str
    is_continuation: bool = False  # request carries a post-compaction summary block
    is_compaction_call: bool = False  # request carries the "summarize the conversation" instruction
    task_spawns: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, str] = field(default_factory=dict)


@dataclass
class Conversation:
    """One reconstructed conversation (root, subagent, agent, or background)."""

    id: str
    kind: str  # "root" | "subagent" | "agent" | "background"
    # "subagent" is reserved for a conversation LINKED to a parent via the
    # harness Task/Agent tool (spawned_by set). An independent top-level
    # conversation that shares the proxy but has no observed spawn edge — a
    # concurrent worker in an engine like Cairn, or a cooperating agent-team
    # member — is "agent": a peer of the root, not a child of it.
    parent_id: str | None = None
    spawned_by: dict[str, Any] | None = None  # {request_id, tool_use_id, parent_index}
    subagent_type: str | None = None
    depth: int = 0
    call_indices: list[int] = field(default_factory=list)
    compaction_calls: list[int] = field(default_factory=list)  # the calls that GENERATE a summary (input=history+summarize prompt, output=summary)
    compaction_at: list[int] = field(default_factory=list)  # the continuation calls that RESUME from a summary
    returns_at: int | None = None  # parent step index where this subagent's result re-enters context
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ConversationForest:
    """Reconstructed structure for one trial's proxy.jsonl."""

    conversations: list[Conversation]
    call_to_conv: dict[int, str]
    structure: dict[str, int]
    root_usage: dict[str, int]
    root_rounds: int

    def conversation_of(self, call_index: int) -> Conversation | None:
        cid = self.call_to_conv.get(call_index)
        if cid is None:
            return None
        for conv in self.conversations:
            if conv.id == cid:
                return conv
        return None


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

def _request_of(entry: dict[str, Any]) -> dict[str, Any]:
    req = entry.get("anthropic_request") or entry.get("openai_request") or {}
    return req if isinstance(req, dict) else {}


def _response_of(entry: dict[str, Any]) -> dict[str, Any]:
    resp = entry.get("anthropic_response") or entry.get("upstream_response") or {}
    return resp if isinstance(resp, dict) else {}


def _strip_volatile(text: str) -> str:
    text = _BILLING_RE.sub("", text)
    text = _REMINDER_RE.sub("", text)
    return text.strip()


def _system_text(req: dict[str, Any]) -> str:
    sysv = req.get("system")
    if isinstance(sysv, list):
        sysv = "\n".join(
            b.get("text", "") for b in sysv if isinstance(b, dict)
        )
    msgs = req.get("messages") or []
    if not sysv and msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        sysv = msgs[0].get("content", "")
    return str(sysv or "")


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic/OpenAI message content to plain text for hashing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("text", "input_text", None) and block.get("text"):
                    parts.append(str(block["text"]))
                elif block.get("type") == "tool_result":
                    parts.append(_content_to_text(block.get("content")))
                elif block.get("type") == "tool_use":
                    parts.append(str(block.get("id") or ""))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def _hash(text: str) -> str:
    return hashlib.md5(_strip_volatile(text).encode("utf-8", "replace")).hexdigest()[:12]


def _message_signatures(req: dict[str, Any]) -> list[str]:
    sigs: list[str] = []
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "?")
        if role == "system":
            continue
        sigs.append(f"{role}:{_hash(_content_to_text(msg.get('content')))}")
    return sigs


def _first_user_text(req: dict[str, Any]) -> str:
    for msg in req.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _strip_volatile(_content_to_text(msg.get("content")))[:4000]
    return ""


def _user_text_blocks(req: dict[str, Any]) -> list[str]:
    """Each text block of each user message, in order (no flattening across blocks)."""
    out: list[str] = []
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("text", "input_text", None) and b.get("text"):
                    out.append(str(b["text"]))
    return out


def _is_compaction_request(req: dict[str, Any]) -> bool:
    """A compaction CALL carries the 'create a detailed summary' instruction."""
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if _COMPACTION_SUMMARY_PROMPT in _content_to_text(msg.get("content")).lower():
            return True
    return False


def _is_continuation_request(req: dict[str, Any]) -> bool:
    """A continuation request carries the summary as a text block that opens with
    the marker. Claude Code keeps the original task as ``messages[0]`` and injects
    the summary as a later user text block, so we scan every user text block's
    head — not just ``messages[0]``."""
    for text in _user_text_blocks(req):
        if _strip_volatile(text).lstrip().lower().startswith(_COMPACTION_MARKER):
            return True
    return False


def _response_blocks(resp: dict[str, Any]) -> list[dict[str, Any]]:
    content = resp.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    # OpenAI chat shape
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        blocks: list[dict[str, Any]] = []
        if isinstance(msg, dict):
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "input": fn.get("arguments") or {},
                })
        return blocks
    return []


def _task_spawns(resp: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract subagent-spawning tool_use blocks (``Task`` / ``Agent``)."""
    spawns: list[dict[str, Any]] = []
    for block in _response_blocks(resp):
        if block.get("type") != "tool_use":
            continue
        name = str(block.get("name") or "")
        if name.lower() not in _SUBAGENT_TOOL_NAMES:
            continue
        raw_input = block.get("input")
        if isinstance(raw_input, str):
            import json
            try:
                raw_input = json.loads(raw_input)
            except (ValueError, TypeError):
                raw_input = {}
        if not isinstance(raw_input, dict):
            raw_input = {}
        spawns.append({
            "id": str(block.get("id") or ""),
            "prompt": _strip_volatile(str(raw_input.get("prompt") or "")),
            "subagent_type": str(
                raw_input.get("subagent_type")
                or raw_input.get("subagentType")
                or ""
            ),
            "description": str(raw_input.get("description") or ""),
        })
    return spawns


def _tool_results_in_request(req: dict[str, Any]) -> dict[str, str]:
    """Tool results carried in this request's messages, keyed by tool-use id."""
    out: dict[str, str] = {}
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        # OpenAI tool message
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            out[str(msg["tool_call_id"])] = _content_to_text(msg.get("content"))
            continue
        # Anthropic tool_result blocks in a user message
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = str(block.get("tool_use_id") or "")
                    if tid:
                        out[tid] = _content_to_text(block.get("content"))
    return out


def _final_text(resp: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in _response_blocks(resp):
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    if parts:
        return "\n".join(parts).strip()
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"].strip()
    return ""


def _usage_of(entry: dict[str, Any]) -> tuple[int, int, int]:
    for key in ("upstream_response", "anthropic_response"):
        resp = entry.get(key)
        if not isinstance(resp, dict):
            continue
        usage = resp.get("usage")
        if not isinstance(usage, dict):
            continue
        in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        reasoning = 0
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
        if isinstance(details, dict):
            reasoning = int(details.get("reasoning_tokens") or 0)
        reasoning = reasoning or int(usage.get("reasoning_tokens") or 0)
        if in_tok or out_tok or reasoning:
            return in_tok, out_tok, reasoning
    return 0, 0, 0


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def _build_call(index: int, entry: dict[str, Any]) -> Call:
    req = _request_of(entry)
    resp = _response_of(entry)
    sigs = _message_signatures(req)
    in_tok, out_tok, reasoning = _usage_of(entry)
    first_user = _first_user_text(req)
    return Call(
        index=index,
        request_id=str(entry.get("request_id") or f"call-{index + 1}"),
        ts_ms=entry.get("ts_ms"),
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=reasoning,
        n_msgs=len(sigs),
        stable_system=_hash(_system_text(req)),
        msg_head=tuple(sigs[:_ANCHOR_HEAD_MSGS]),
        first_user=first_user,
        final_text=_final_text(resp),
        is_continuation=_is_continuation_request(req),
        is_compaction_call=_is_compaction_request(req),
        task_spawns=_task_spawns(resp),
        tool_results=_tool_results_in_request(req),
    )


def _prompts_match(task_prompt: str, first_user: str) -> bool:
    """Does a child conversation's opening message match a Task prompt?"""
    if not task_prompt or not first_user:
        return False
    a = task_prompt.strip()
    b = first_user.strip()
    if not a or not b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    head = short[: min(len(short), 160)]
    return head in long_ or short in long_


class _OpenChain:
    """Mutable chain state while streaming calls in arrival order.

    A chain is one *prefix-growing* run of calls. Compaction (history collapse)
    and subagent linking are resolved later, with global information, so the
    streaming pass only ever does the unambiguous thing: continue a chain when a
    call grows it, otherwise start a new chain.
    """

    __slots__ = ("conv", "last_n_msgs", "peak_in", "stable_system", "anchor_head")

    def __init__(self, conv: Conversation, call: Call):
        self.conv = conv
        self.last_n_msgs = call.n_msgs
        self.peak_in = call.input_tokens
        self.stable_system = call.stable_system
        self.anchor_head = call.msg_head

    def continues(self, call: Call) -> bool:
        if self.stable_system != call.stable_system:
            return False
        # Worker-identity gate: a conversation is anchored on its OPENING user
        # turn (messages[0]), which is invariant across every one of its calls —
        # normal turns, retries, and post-compaction continuations all keep it.
        # A call whose opening turn differs is a different worker/subagent even
        # when it shares the system prompt. This is the Cairn case: many
        # concurrent workers, one system prompt, distinct task-graph snapshots as
        # their opening message — the head0 gate keeps their interleaved calls in
        # separate chains instead of letting monotonic growth merge them.
        self_head0 = self.anchor_head[0] if self.anchor_head else ""
        call_head0 = call.msg_head[0] if call.msg_head else ""
        if self_head0 and call_head0 and self_head0 != call_head0:
            return False
        if call.n_msgs == 0 and self.last_n_msgs == 0:
            return True  # no message-history signal at all (degenerate/bodyless
            # audit entries) — keep them together rather than spawn phantoms
        # A genuine fresh start — only the opening message, with real input
        # tokens — begins a NEW conversation, even under an identical opening
        # anchor (a same-prompt sibling worker). Retries/duplicates (which never
        # carry real input tokens on a bare opening) are absorbed below.
        if call.n_msgs <= 1 and call.input_tokens > 0:
            return False
        if call.n_msgs > self.last_n_msgs:
            anchor_len = min(len(call.msg_head), len(self.anchor_head))
            if anchor_len and call.msg_head[:anchor_len] != self.anchor_head[:anchor_len]:
                return False
            return True
        # call.n_msgs <= last_n_msgs, same worker anchor, not a fresh start. A
        # post-compaction continuation collapses the history but must stay a
        # SEPARATE chain so pass 2 can record the compaction boundary — leave it.
        # Everything else here is a retry / bodyless duplicate / re-sent earlier
        # state of THIS worker (an in=0 audit entry or a non-monotonic dip):
        # absorb it rather than spawn a phantom conversation.
        if call.is_continuation:
            return False
        return True


def reconstruct_forest(entries: list[dict[str, Any]]) -> ConversationForest:
    """Reconstruct the conversation forest from successful proxy.jsonl entries.

    *entries* must be the ``status == "success"`` records in arrival order.
    """

    calls = [_build_call(i, e) for i, e in enumerate(entries)]
    conversations: list[Conversation] = []
    call_to_conv: dict[int, str] = {}
    open_chains: list[_OpenChain] = []
    counter = 0

    # --- Pass 1: prefix-growing chains only. ---
    for call in calls:
        match: _OpenChain | None = None
        for oc in open_chains:
            if oc.continues(call) and (match is None or oc.last_n_msgs > match.last_n_msgs):
                match = oc
        if match is None:
            conv = Conversation(id=f"c{counter}", kind="root" if counter == 0 else "unknown")
            counter += 1
            conversations.append(conv)
            match = _OpenChain(conv, call)
            open_chains.append(match)
        match.conv.call_indices.append(call.index)
        call_to_conv[call.index] = match.conv.id
        # Keep the high-water mark: an absorbed retry / dip must not lower the
        # chain's growth frontier, or the next real turn would look like a jump
        # from the dip and could be mis-split.
        match.last_n_msgs = max(match.last_n_msgs, call.n_msgs)
        match.peak_in = max(match.peak_in, call.input_tokens)

    # --- Pass 2: classify, merge compaction continuations, link subagents. ---
    _classify_and_link(calls, conversations, call_to_conv)
    return _finalize(calls, conversations, call_to_conv)


def _classify_and_link(
    calls: list[Call],
    conversations: list[Conversation],
    call_to_conv: dict[int, str],
) -> None:
    by_index = {c.index: c for c in calls}

    # Gather every Task spawn across the whole stream.
    spawns: list[tuple[dict[str, Any], Call]] = []
    for call in calls:
        for spawn in call.task_spawns:
            spawns.append((spawn, call))
    used_spawn_ids: set[str] = set()

    # --- Subagent linking: a child conversation whose opening message matches a
    #     Task prompt emitted earlier in the stream. ---
    for conv in conversations:
        if conv.kind == "root" or not conv.call_indices:
            continue
        first_call = by_index[conv.call_indices[0]]
        best: tuple[dict[str, Any], Call] | None = None
        for spawn, parent_call in spawns:
            if spawn["id"] in used_spawn_ids or parent_call.index >= first_call.index:
                continue
            if _prompts_match(spawn["prompt"], first_call.first_user):
                if best is None or parent_call.index > best[1].index:
                    best = (spawn, parent_call)
        if best is not None:
            spawn, parent_call = best
            used_spawn_ids.add(spawn["id"])
            conv.kind = "subagent"
            conv.subagent_type = spawn["subagent_type"] or "subagent"
            conv.parent_id = call_to_conv.get(parent_call.index)
            conv.spawned_by = {
                "request_id": parent_call.request_id,
                "tool_use_id": spawn["id"],
                "parent_index": parent_call.index,
                "prompt": spawn["prompt"],
                "description": spawn["description"],
            }

    # --- Compaction merge: a chain that opens with the post-compaction summary
    #     marker is the SAME conversation resuming after the harness summarized it.
    #     The call that produced that summary (input = full history + a "summarize
    #     the conversation" instruction, output = the summary) is the prior call in
    #     the target conversation — that is the compaction *call* we surface. ---
    remaining = sorted(
        (c for c in conversations if c.spawned_by is None and c.call_indices),
        key=lambda c: c.call_indices[0],
    )
    for conv in remaining:
        first_call = by_index[conv.call_indices[0]]
        if not first_call.is_continuation:
            continue  # only a real summary-marker opens a compaction continuation
        target = None
        for cand in conversations:
            if cand is conv or not cand.call_indices:
                continue
            if cand.call_indices[0] >= first_call.index:
                continue
            if by_index[cand.call_indices[0]].stable_system != first_call.stable_system:
                continue
            if target is None or cand.call_indices[0] > target.call_indices[0]:
                target = cand
        if target is not None:
            target.compaction_at.append(first_call.index)
            for idx in conv.call_indices:
                call_to_conv[idx] = target.id
            target.call_indices.extend(conv.call_indices)
            target.call_indices.sort()
            conv.call_indices = []  # mark merged-away

    # Drop merged-away conversations.
    conversations[:] = [c for c in conversations if c.call_indices]

    # The compaction CALLS (those that generate a summary) are flagged directly
    # from their request, so attribute them wherever they landed after merging.
    for conv in conversations:
        conv.compaction_calls = sorted(
            i for i in conv.call_indices if by_index[i].is_compaction_call
        )

    # --- Background and unlinked remainder. ---
    for conv in conversations:
        if conv.kind in ("root", "subagent"):
            continue
        first_call = by_index[conv.call_indices[0]]
        if (
            len(conv.call_indices) == 1
            and first_call.n_msgs <= _BACKGROUND_MAX_MSGS
            and first_call.input_tokens <= _BACKGROUND_MAX_INPUT_TOKENS
        ):
            conv.kind = "background"
        elif len(conv.call_indices) == 1 and first_call.input_tokens <= _BACKGROUND_MAX_INPUT_TOKENS:
            conv.kind = "background"
        else:
            # A substantial conversation with no Task/Agent spawn edge is an
            # independent top-level agent (a Cairn worker, an agent-team peer),
            # NOT a subagent of the root. Calling it "subagent:?" implied a
            # delegation hierarchy that the stream gives no evidence for.
            conv.kind = "agent"

    # --- Depth from parent chain. ---
    conv_by_id = {c.id: c for c in conversations}
    for conv in conversations:
        depth, cursor, seen = 0, conv, set()
        while cursor.parent_id and cursor.parent_id not in seen and cursor.parent_id in conv_by_id:
            seen.add(cursor.parent_id)
            depth += 1
            cursor = conv_by_id[cursor.parent_id]
        conv.depth = depth

    # --- Return edge: the parent step where each linked subagent's result folds
    #     back into context. A subagent runs while the parent blocks; its output
    #     re-enters the parent as a tool_result keyed by the spawning tool_use id.
    #     That tool_result first appears in the parent's *next* request — that step
    #     is where the subagent's context actually enters the parent agent. ---
    sorted_indices = sorted(by_index)
    for conv in conversations:
        if conv.kind != "subagent" or not conv.spawned_by:
            continue
        tool_use_id = conv.spawned_by.get("tool_use_id")
        parent_index = conv.spawned_by.get("parent_index")
        if not tool_use_id or parent_index is None:
            continue
        for idx in sorted_indices:
            if idx <= parent_index:
                continue
            if tool_use_id in by_index[idx].tool_results:
                conv.returns_at = idx
                break


def _finalize(
    calls: list[Call],
    conversations: list[Conversation],
    call_to_conv: dict[int, str],
) -> ConversationForest:
    by_index = {c.index: c for c in calls}
    for conv in conversations:
        u = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "num_requests": 0}
        for idx in conv.call_indices:
            call = by_index[idx]
            u["input_tokens"] += call.input_tokens
            u["output_tokens"] += call.output_tokens
            u["reasoning_tokens"] += call.reasoning_tokens
            u["num_requests"] += 1
        conv.usage = u

    root = next((c for c in conversations if c.kind == "root"), None)
    root_usage = dict(root.usage) if root else {
        "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "num_requests": 0,
    }
    # Root rounds = root calls that are agent decisions. Exclude the compaction
    # CALLS themselves (the model summarizing its own context is bookkeeping, not
    # a new agent turn) AND absorbed retries / bodyless duplicates (in=0 audit
    # entries that carry no new decision). The continuation that resumes from a
    # summary IS a real agent turn, so it stays counted.
    root_rounds = 0
    if root:
        compaction = set(root.compaction_calls)
        root_rounds = sum(
            1
            for i in root.call_indices
            if i not in compaction and by_index[i].input_tokens > 0
        )

    structure = {
        "num_conversations": len(conversations),
        "num_subagents": sum(1 for c in conversations if c.kind == "subagent"),
        "num_agents": sum(1 for c in conversations if c.kind == "agent"),
        "num_background_calls": sum(
            len(c.call_indices) for c in conversations if c.kind == "background"
        ),
        "num_compactions": sum(len(c.compaction_calls) for c in conversations),
        "max_depth": max((c.depth for c in conversations), default=0),
    }
    return ConversationForest(
        conversations=conversations,
        call_to_conv=call_to_conv,
        structure=structure,
        root_usage=root_usage,
        root_rounds=root_rounds,
    )

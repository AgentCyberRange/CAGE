"""Tests for trial trajectory template helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _extract_js_function(source: str, name: str) -> str:
    marker = f"function {name}"
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for idx in range(brace, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"Could not extract {name}")


def test_json_syntax_highlighter_does_not_corrupt_tailwind_classes() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    helper = _extract_js_function(template, "syntaxColorJSON")
    script = f"""
function esc(text) {{
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}
{helper}
const html = syntaxColorJSON(JSON.stringify({{reasoning_content: null, delta: -12}}, null, 2));
if (html.includes('text-slate-<span')) throw new Error(html);
if (html.includes('text-purple-<span')) throw new Error(html);
if (!html.includes('text-slate-500">null</span>')) throw new Error(html);
if (!html.includes('text-purple-400">-12</span>')) throw new Error(html);
"""

    subprocess.run(["node", "-e", script], check=True)


def test_trial_page_initializes_trajectory_incrementally() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")

    assert "fetch(`/api/trajectory/${TRIAL_PATH}?offset=${offset}&limit=${limit}`)" in template
    assert "fetch(`/api/trajectory/${TRIAL_PATH}/all`)" not in template
    assert "setInterval(pollForNewSteps" in template
    assert "data.total_steps_known" in template
    assert "at least" in template
    assert "data.summary" in template
    assert "top_tools" in template
    assert "last_action" in template
    assert "error_count" in template


def test_trajectory_search_does_not_force_full_dom_render() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    search_helper = _extract_js_function(template, "doSearch")
    nav_helper = _extract_js_function(template, "scrollToMatch")

    assert "renderAllRemaining" not in search_helper
    assert "loadStepsForSearch" in search_helper
    assert "highlightRenderedSearchMatches" in search_helper
    assert "renderAllRemaining" not in nav_helper
    assert "ensureLoadedThrough" in nav_helper


def test_trajectory_bottom_bar_uses_bound_controls_instead_of_inline_handlers() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    start = template.index('<div id="bottom-bar"')
    end = template.index("<!-- JSON overlay panel", start)
    bottom_bar = template[start:end]

    assert "onclick=" not in bottom_bar
    assert 'data-trajectory-action="search-prev"' in bottom_bar
    assert 'data-trajectory-action="search-next"' in bottom_bar
    assert 'data-trajectory-action="search-clear"' in bottom_bar
    assert 'data-trajectory-action="top"' in bottom_bar
    assert 'data-trajectory-action="end"' in bottom_bar
    assert 'data-trajectory-jump' in bottom_bar
    assert "bindTrajectoryControls()" in template


def test_json_overlay_uses_bound_controls_instead_of_inline_handlers() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    start = template.index('<div id="json-panel">')
    end = template.index('{% endblock %}', start)
    panel = template[start:end]

    assert "onclick=" not in panel
    assert 'data-context-panel-action="close"' in panel
    assert 'data-context-mode="readable"' in panel
    assert 'data-context-mode="tools"' in panel
    assert 'data-context-mode="raw"' in panel
    assert "bindContextPanelControls()" in template


def test_raw_context_action_bar_uses_bound_controls() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    start = template.index("html += '<div data-context-mode-panel=\"raw\"")
    end = template.index("html += section(\n    'Context Diff'", start)
    raw_panel = template[start:end]

    assert "onclick=" not in raw_panel
    assert 'data-context-action="copy-request"' in raw_panel
    assert 'data-context-action="scroll-section"' in raw_panel
    assert 'data-context-section="context-diff"' in raw_panel
    assert 'data-context-section="full-json"' in raw_panel


def test_build_step_html_renders_placeholder_for_empty_steps() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    helper = _extract_js_function(template, "buildStepHTML")
    script = f"""
function esc(text) {{
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}
function fmtTok(n) {{
  if (!n) return '0';
  n = parseInt(n, 10);
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return '' + n;
}}
{helper}
const html = buildStepHTML({{
  index: 20,
  blocks: [],
  tokens: {{ in: 0, out: 0, reasoning: 0 }},
  cumulative: {{ in: 0, out: 0 }}
}});
if (!html.includes('No response content captured for this step.')) throw new Error(html);
if (!html.includes('Open JSON to inspect the raw proxy record.')) throw new Error(html);
if (html.includes('onclick=')) throw new Error(html);
if (!html.includes('data-context-step-index="20"')) throw new Error(html);
"""

    subprocess.run(["node", "-e", script], check=True)


def test_render_messages_groups_setup_and_internal_compaction_prompts() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    helpers = "\n".join(
        _extract_js_function(template, name)
        for name in (
            "renderToolResult",
            "renderContent",
            "chip",
            "messageText",
            "previewText",
            "isCompactionMessage",
            "messageSummary",
            "renderMessageCard",
            "renderMessages",
        )
    )
    script = f"""
function esc(text) {{
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}
{helpers}
const html = renderMessages([
  {{
    role: 'developer',
    content: [{{ type: 'text', text: '<permissions instructions>\\\\n' + 'x'.repeat(1200) }}],
  }},
  {{
    role: 'user',
    content: [{{ type: 'text', text: '<environment_context>\\\\n  <cwd>/home/agent/workspace</cwd>\\\\n</environment_context>' }}],
  }},
  {{
    role: 'user',
    content: [{{ type: 'text', text: '## Description\\\\nDiscover all vulnerabilities.\\\\n\\\\n## Target\\\\n- http://127.0.0.1:10050/' }}],
  }},
  {{
    role: 'assistant',
    content: [{{ type: 'text', text: 'The app leaks ThinkPHP debug traces on the public routes.' }}],
  }},
  {{
    role: 'user',
    content: [{{ type: 'text', text: 'You are performing a CONTEXT CHECKPOINT COMPACTION.\\\\n\\\\nInclude:\\\\n- Current progress and key decisions made\\\\n- What remains to be done' }}],
  }},
]);
if (!html.includes('Setup context')) throw new Error(html);
if (!html.includes('Task instructions')) throw new Error(html);
if (!html.includes('Context checkpoint compaction prompt')) throw new Error(html);
if (html.includes('Include:\\n- Current progress and key decisions made')) throw new Error(html);
"""

    subprocess.run(["node", "-e", script], check=True)


def test_render_context_diff_shows_before_after_message_cards() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    helpers = "\n".join(
        _extract_js_function(template, name)
        for name in (
            "renderToolResult",
            "renderContent",
            "chip",
            "messageText",
            "previewText",
            "isCompactionMessage",
            "messageSummary",
            "renderMessageCard",
            "renderMessages",
            "renderLineDiff",
            "renderContextDiff",
        )
    )
    script = f"""
function esc(text) {{
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}
{helpers}
const html = renderContextDiff({{
  previous_step_index: 4,
  messages: {{
    added_count: 0,
    removed_count: 0,
    modified_count: 1,
    unchanged_prefix: 3,
    unchanged_suffix: 0,
    added: [],
    removed: [],
    modified: [
      {{
        old: {{ role: 'assistant', content: [{{ type: 'text', text: 'Old reasoning path' }}], text: 'Old reasoning path' }},
        new: {{ role: 'assistant', content: [{{ type: 'text', text: 'New reasoning path' }}], text: 'New reasoning path' }},
      }}
    ],
  }},
  system: {{ changed: false, old_chars: 0, new_chars: 0, old_text: '', new_text: '' }},
  tools: {{ changed: false, old_count: 0, new_count: 0, added: [], removed: [] }},
  params: [],
}});
if (!html.includes('Before')) throw new Error(html);
if (!html.includes('After')) throw new Error(html);
if (!html.includes('Old reasoning path')) throw new Error(html);
if (!html.includes('New reasoning path')) throw new Error(html);
"""

    subprocess.run(["node", "-e", script], check=True)


def test_render_context_detail_defaults_to_readable_mode() -> None:
    template = Path("cage/web/templates/trial.html").read_text(encoding="utf-8")
    assert "function setContextMode" in template
    helpers = "\n".join(
        _extract_js_function(template, name)
        for name in (
            "renderToolResult",
            "renderContent",
            "chip",
            "section",
            "renderTokenUsage",
            "renderRequestTools",
            "messageText",
            "previewText",
            "isCompactionMessage",
            "messageSummary",
            "renderMessageCard",
            "renderMessages",
            "renderResponseContent",
            "renderLineDiff",
            "renderContextDiff",
            "syntaxColorJSON",
            "setContextMode",
            "renderContextDetail",
        )
    )
    script = f"""
const nodes = {{
  'ctx-title': {{ textContent: '' }},
  'ctx-meta': {{ innerHTML: '' }},
  'ctx-detail': {{ innerHTML: '' }},
  'json-scroll': {{ scrollTop: 99 }},
  'json-search': {{ value: 'old' }},
  'json-search-count': {{ textContent: 'old' }},
}};
const document = {{
  getElementById(id) {{
    return nodes[id];
  }},
}};
function esc(text) {{
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}
function fmtTok(n) {{
  if (!n) return '0';
  n = parseInt(n, 10);
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return '' + n;
}}
{helpers}
renderContextDetail({{
  summary: {{ model: 'demo-model', status: 'ok', message_count: 2, tool_count: 1 }},
  request: {{
    tools: [{{ name: 'Bash', description: 'run shell commands', input_schema: {{ properties: {{ cmd: {{ type: 'string' }} }}, required: ['cmd'] }} }}],
    messages: [
      {{ role: 'user', content: [{{ type: 'text', text: 'Find the exploit path.' }}] }},
      {{ role: 'assistant', content: [{{ type: 'text', text: 'I will inspect the target.' }}] }},
    ],
  }},
  response: {{ content: [{{ type: 'text', text: 'The target leaks debug output.' }}], tool_names: ['Bash'] }},
  context_diff: {{ previous_step_index: null, messages: {{}}, system: {{}}, tools: {{}}, params: [] }},
  raw: {{ id: 'raw-record' }},
}}, 7);
const html = nodes['ctx-detail'].innerHTML;
if (!html.includes('Readable evidence')) throw new Error(html);
if (!html.includes('Tool timeline')) throw new Error(html);
if (!html.includes('Raw JSON')) throw new Error(html);
if (!html.includes('data-context-mode-panel="readable"')) throw new Error(html);
if (!html.includes('data-context-mode-panel="tools"')) throw new Error(html);
if (!html.includes('data-context-mode-panel="raw"')) throw new Error(html);
if (!/data-context-mode-panel="raw"[^>]*hidden/.test(html)) throw new Error(html);
if (html.indexOf('Readable evidence') > html.indexOf('Full JSON')) throw new Error(html);
"""

    subprocess.run(["node", "-e", script], check=True)

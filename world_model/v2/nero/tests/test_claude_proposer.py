"""Claude Code CLI proposer backend (pan/llm.ClaudeCodeVLM) + nero/contents/proposer_compare: mock subprocess only, NO live call."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from world_model.v2.nero.contents import proposer_compare as PCmp
from world_model.v2.nero.pan import llm as LLM


def _out(result="{\"ok\": 1}", stop="end_turn", subtype="success", is_error=False):
    lines = [dict(type="system", subtype="init"), dict(type="assistant", message=dict(stop_reason=stop, content=[dict(type="text", text=result)])),
             dict(type="result", subtype=subtype, is_error=is_error, result=result, stop_reason=stop, total_cost_usd=0.01, duration_ms=1234, num_turns=1,
                  usage=dict(input_tokens=2, output_tokens=7))]
    return "\n".join(json.dumps(x) for x in lines) + "\n"


def _patch(monkeypatch, stdout, rc=0, seen=None):
    def run(cmd, input=None, **kw):
        if seen is not None:
            seen.update(cmd=cmd, input=input, kw=kw)
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr="boom")
    monkeypatch.setattr(subprocess, "run", run)


def test_command_line_no_tools_no_settings():
    b = LLM.make("claude:claude-sonnet-5-5:high")
    cmd = b.command("SYS")
    assert cmd[:2] == [b.bin, "-p"]
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5-5" and cmd[cmd.index("--effort") + 1] == "high"
    assert cmd[cmd.index("--tools") + 1] == ""                     # no tools at all
    assert cmd[cmd.index("--setting-sources") + 1] == ""           # no user / project / local settings, hooks
    for f in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands", "--verbose"):
        assert f in cmd
    assert cmd[cmd.index("--input-format") + 1] == "stream-json" and cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert "--bare" not in cmd and "--dangerously-skip-permissions" not in cmd
    assert b.env()["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


def test_spec_defaults_and_bad_effort():
    b = LLM.make("claude")
    assert (b.model, b.reasoning) == ("claude-sonnet-5-5", "high")
    assert LLM.make("claude:claude-sonnet-5-5:low").reasoning == "low"
    with pytest.raises(ValueError):
        LLM.make("claude:claude-sonnet-5-5:ultra")


def test_stdin_message_image_block(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNGdata")
    o = json.loads(LLM.ClaudeCodeVLM.stdin_message("hello", [p]))
    assert o["type"] == "user" and o["message"]["role"] == "user"
    blocks = o["message"]["content"]
    assert blocks[0]["type"] == "text" and blocks[0]["text"].endswith("hello") and blocks[0]["text"].startswith("You are being used as a plain chat-completion backend")
    assert blocks[1]["type"] == "image" and blocks[1]["source"] == dict(type="base64", media_type="image/png", data="iVBOR2RhdGE=")


def test_ask_parses_result_and_runs_in_empty_dir(monkeypatch):
    seen = {}
    _patch(monkeypatch, _out('{"phenomena": []}'), seen=seen)
    b = LLM.make("claude")
    o = b.ask("t", system="S")
    assert o["texts"] == ['{"phenomena": []}'] and o["backend"] == "claude:claude-sonnet-5-5:high"
    assert o["cli"]["stop_reason"] == "end_turn" and o["cli"]["cost_usd"] == 0.01 and o["cli"]["usage"]["output_tokens"] == 7
    assert "ac_claude_" in seen["kw"]["cwd"] and seen["kw"]["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "--tools" in seen["cmd"] and json.loads(seen["input"])["message"]["content"][0]["type"] == "text"


@pytest.mark.parametrize("kw", [dict(stop="max_tokens"), dict(subtype="error_max_turns", is_error=True), dict(result="  ")])
def test_cut_error_or_empty_is_no_answer(monkeypatch, kw):
    _patch(monkeypatch, _out(**kw))
    with pytest.raises(RuntimeError):
        LLM.make("claude").ask("t")


def test_nonzero_exit_without_result_raises(monkeypatch):
    _patch(monkeypatch, "", rc=1)
    with pytest.raises(RuntimeError, match="claude exit 1"):
        LLM.make("claude").ask("t")


def test_info_records_model_effort_caveats():
    i = LLM.info("claude:claude-sonnet-5-5:high")
    assert i["backend"] == "claude" and i["model"] == "claude-sonnet-5-5" and i["reasoning"] == "high"
    assert "REPLACED" in i["caveats"] and "temperature not settable" in i["caveats"] and "plain chat-completion backend" in i["caveats"]


def test_classify_and_select():
    assert PCmp.classify("TASK_ID: box_behind\nTASK: x\n\nReturn the JSON object now.") == ("box_behind", "plan")
    assert PCmp.classify("TASK_ID: box_behind\nYOUR PREVIOUS PLAN (JSON):\n{}")[1] == "rule2"
    assert PCmp.classify("TASK_ID: tray6_behind\nREJECTED PLAN 1 (JSON):\n{}") == ("tray6_behind", "rejected")
    items = [dict(task="a", kind="plan", key=str(i)) for i in range(6)]
    assert [x["key"] for x in PCmp.select(items, {"plan"}, (1, 2), 0)] == ["1", "3", "5"]
    assert len(PCmp.select(items, {"plan"}, (0, 1), 2)) == 2

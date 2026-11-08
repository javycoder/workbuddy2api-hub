"""Historical reasoning mirror regression tests."""
import json
import os
import sys
import tempfile

_startup_dir = tempfile.TemporaryDirectory(prefix="upstream-reasoning-mirror-")
os.environ["ACCOUNTS_DIR"] = _startup_dir.name
os.environ["WB_PROXY_USAGE_DIR"] = _startup_dir.name
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wb_proxy as P

PASS = FAIL = 0
MIRRORED_MODELS = (
    "kimi-k2", "glm-4.6", "minimax-m2", "mimo-v2", "step-3", "space-bunny",
)


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  [PASS] %s" % name)
    else:
        FAIL += 1
        print("  [FAIL] %s  %s" % (name, extra))


def assistant_messages(body):
    return [message for message in body.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "assistant"]


print("[1] Supported models mirror non-empty reasoning_content")
trace = " model thought with leading and trailing spaces "
for model in MIRRORED_MODELS:
    body = P.build_upstream_body({
        "model": model,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": trace},
        ],
    })
    assistant = assistant_messages(body)[0]
    check("%s mirrors the original reasoning_content" % model,
          assistant.get("reasoning") == trace,
          json.dumps(assistant, ensure_ascii=False))

print()
print("[2] GLM mirrors after adjacent assistant repair")
tool_trace = "real GLM reasoning after tool call"
body = P.build_upstream_body({
    "model": "glm-4.6",
    "messages": [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "look", "arguments": "{}"},
        }]},
        {"role": "assistant", "content": "done", "reasoning_content": tool_trace},
        {"role": "tool", "tool_call_id": "call_1", "name": "look", "content": "result"},
        {"role": "user", "content": "next"},
    ],
})
merged_assistants = assistant_messages(body)
check("adjacent assistants are merged", len(merged_assistants) == 1,
      json.dumps(body["messages"], ensure_ascii=False))
check("merged reasoning remains the original trace",
      merged_assistants[0].get("reasoning") == tool_trace,
      json.dumps(merged_assistants[0], ensure_ascii=False))

print()
print("[3] Whitespace reasoning_content is not mirrored")
body = P.build_upstream_body({
    "model": "kimi-k2",
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two", "reasoning_content": "  "},
    ],
})
silent_assistants = assistant_messages(body)
check("whitespace reasoning_content adds no reasoning field",
      all("reasoning" not in message for message in silent_assistants),
      json.dumps(silent_assistants, ensure_ascii=False))

print()
print("[4] Existing non-empty reasoning is not overwritten")
existing = " existing reasoning "
body = P.build_upstream_body({
    "model": "minimax-m2",
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello",
         "reasoning": existing, "reasoning_content": trace},
    ],
})
assistant = assistant_messages(body)[0]
check("existing reasoning remains unchanged", assistant.get("reasoning") == existing,
      json.dumps(assistant, ensure_ascii=False))

print()
print("[5] GPT and hy do not gain reasoning")
for model in ("gpt-5", "hy4-preview"):
    body = P.build_upstream_body({
        "model": model,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": trace},
        ],
    })
    check("%s does not gain reasoning" % model,
          "reasoning" not in assistant_messages(body)[0],
          json.dumps(body, ensure_ascii=False))

print()
print("[6] DeepSeek placeholder and thinking backfill remain unchanged")
body = P.build_upstream_body({
    "model": "deepseek-v4.1-flash",
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ],
})
deepseek_assistant = assistant_messages(body)[0]
check("DeepSeek without thought keeps only the space placeholder",
      deepseek_assistant.get("reasoning") == " ",
      json.dumps(deepseek_assistant, ensure_ascii=False))
check("DeepSeek thinking remains enabled",
      body.get("thinking") == {"type": "enabled"}, json.dumps(body))
check("DeepSeek reasoning effort remains set",
      body.get("reasoning_effort") == "high", json.dumps(body))

print()
print("[7] Mirrored models do not gain DeepSeek reasoning controls")
for model in MIRRORED_MODELS:
    body = P.build_upstream_body({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })
    check("%s body has no thinking field" % model,
          "thinking" not in body, json.dumps(body))
    check("%s body has no injected reasoning effort" % model,
          "reasoning_effort" not in body and "reasoningEffort" not in body,
          json.dumps(body))

print()
print("[8] Responses reasoning summary survives the chat path")
summary = " Responses summary with original spacing "
responses_payload = {
    "model": "kimi-k2",
    "input": [
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": summary},
        ]},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "answer"},
        ]},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "next"},
        ]},
    ],
}
body = P.build_upstream_body(P.responses_to_chat(responses_payload))
responses_assistant = assistant_messages(body)[0]
check("Responses summary becomes the original Kimi reasoning",
      responses_assistant.get("reasoning") == summary,
      json.dumps(responses_assistant, ensure_ascii=False))

print()
print("PASS=%d FAIL=%d" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)

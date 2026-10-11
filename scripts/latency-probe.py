#!/usr/bin/env python3
"""Measure time-to-first-token on trivial prompts.

A ~9s TTFT on a 30-token prompt is far too slow to be prefill. If a one-word
answer shows the same delay, the model is spending that time on something
before emitting content — for Qwen3 that means thinking tokens, which newer
llama.cpp reports separately as `reasoning_content` rather than `content`.

    python3 scripts/latency-probe.py [base_url]
"""

from __future__ import annotations

import http.cookiejar
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"


class _Policy(http.cookiejar.DefaultCookiePolicy):
    def return_ok_secure(self, cookie, request):  # noqa: D102
        return True


jar = http.cookiejar.CookieJar(policy=_Policy())
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def call(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": UA}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with opener.open(req, timeout=120) as res:
        raw = res.read().decode()
        return json.loads(raw) if raw else {}


token = next(
    line.split("=", 1)[1].strip()
    for line in open("/opt/local-llm/.env")
    if line.startswith("APP_ACCESS_TOKEN=")
)
call("/api/auth/unlock", "POST", {"token": token})
cid = call("/api/conversations", "POST", {})["conversation"]["id"]

PROMPTS = [
    "Say hi in exactly two words.",
    "What is 2+2? Reply with just the number.",
    "Name one colour. One word only.",
]

print(f"\nTTFT probe against {BASE}\n" + "-" * 72)

for prompt in PROMPTS:
    started = call(f"/api/conversations/{cid}/messages", "POST", {"content": prompt})
    job_id = started["jobId"]
    url = BASE + "/api/jobs/" + job_id + "/stream"
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    t0 = time.monotonic()
    first: float | None = None
    chunks: list[str] = []

    with opener.open(req, timeout=180) as res:
        event = None
        for raw_line in res:
            line = raw_line.decode().rstrip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event == "delta":
                    if first is None:
                        first = time.monotonic() - t0
                    chunks.append(payload["text"])
                elif event in ("done", "error"):
                    reply = "".join(chunks).replace("\n", " ")[:70]
                    print(
                        f"{prompt[:36]:38s} ttft={first if first else -1:6.2f}s  "
                        f"out={payload.get('completionTokens')}  "
                        f"tok/s={payload.get('tokensPerSec')}"
                    )
                    print(f"{'':38s} reply: {reply!r}")
                    break

print("-" * 72)
print("A multi-second TTFT on a one-word answer means the delay is not prefill.")

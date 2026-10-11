#!/usr/bin/env python3
"""End-to-end verification against a running deployment.

Exercises the whole path a user takes -- sign in, create a conversation, stream
a generation, cancel one, read back what was persisted -- plus the things that
are dangerous to get wrong and invisible from the UI: that one account cannot
reach another's chats or jobs, that admin endpoints refuse non-admins, and that
the worker-artifact endpoint refuses an unauthenticated caller.

Uses only the standard library.

    python3 scripts/e2e-test.py [base_url]

Accounts come from the deployment's .env (E2E_EMAIL / E2E_PASSWORD and
E2E_ALT_EMAIL / E2E_ALT_PASSWORD), and are created on first run if absent.
They are reused across runs rather than made fresh each time: a throwaway
account cannot delete itself (self_delete is refused), so minting new ones per
run would leave a growing pile of users behind. Conversations this script
creates are deleted on the way out.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import threading
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081").rstrip("/")
ENV_PATH = os.environ.get("NEXALM_ENV", "/opt/local-llm/.env")

# Cloudflare's bot protection rejects the default "Python-urllib/3.x" agent with
# a 403 before the request reaches the origin, so testing through the public
# domain needs a normal browser agent. Direct origin requests are unaffected.
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

passed = 0
failed = 0
skipped = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
    return ok


def skip(label: str, why: str) -> None:
    global skipped
    skipped += 1
    print(f"  SKIP  {label} — {why}")


class _TestPolicy(http.cookiejar.DefaultCookiePolicy):
    """Allow Secure cookies over plain HTTP -- test harness only.

    The session cookie is correctly marked Secure: in production the browser
    reaches the app over HTTPS (Cloudflare terminates TLS and forwards plain
    HTTP to the origin). This script may talk to the origin directly, so
    without this relaxation the cookie would never be sent back.
    """

    def return_ok_secure(self, cookie, request):  # noqa: D102
        return True


class Session:
    """One browser-equivalent: its own cookie jar, so accounts stay separate.

    Isolation checks are only meaningful if the second account cannot
    accidentally inherit the first one's session, which a module-level jar
    would have allowed.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.jar = http.cookiejar.CookieJar(policy=_TestPolicy())
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [("User-Agent", UA)]
        self.user: dict = {}

    def call(self, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            BASE + path,
            data=data,
            method=method,
            headers=({"Content-Type": "application/json"} if data else {}) | {"User-Agent": UA},
        )
        try:
            with self.opener.open(req, timeout=60) as res:
                raw = res.read().decode()
                return res.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, {"raw": raw[:300]}
        except urllib.error.URLError as e:
            return 0, {"raw": str(e.reason)}

    def stream(self, job_id: str, stop_after: float | None = None) -> dict:
        """Consume the SSE stream, returning collected text, events and timings."""
        # The opener's cookie processor attaches the jar; setting a Cookie
        # header by hand here would suppress it and send only one cookie.
        req = urllib.request.Request(
            BASE + f"/api/jobs/{job_id}/stream", headers={"User-Agent": UA}
        )
        text: list[str] = []
        events: list[str] = []
        done: dict = {}
        deltas_at: list[float] = []
        started = time.monotonic()

        with self.opener.open(req, timeout=180) as res:
            event = None
            for raw_line in res:
                line = raw_line.decode(errors="replace").rstrip("\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                    events.append(event)
                elif line.startswith("data:"):
                    payload_raw = line[5:].strip()
                    if not payload_raw:
                        continue
                    try:
                        payload = json.loads(payload_raw)
                    except json.JSONDecodeError:
                        continue
                    if event == "delta":
                        text.append(payload.get("text", ""))
                        deltas_at.append(time.monotonic() - started)
                    elif event in ("done", "error"):
                        done = payload
                        done["_event"] = event
                        break
                if stop_after and time.monotonic() - started > stop_after:
                    break

        return {"text": "".join(text), "events": events, "final": done, "delta_times": deltas_at}


def read_env(key: str, default: str | None = None) -> str | None:
    """Read a key from the deployment .env, or fall back to the environment."""
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as fh:
            for line in fh:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip()
    return os.environ.get(key, default)


def ensure_account(sess: Session, email: str, password: str, label: str) -> bool:
    """Sign in, creating the account if it does not exist yet.

    Deliberately tries signup *first*. Probing with a login that might fail
    burns an attempt against the throttle in routes/auth.ts, which locks an
    ip+email pair for up to 15 minutes after 5 failures -- long enough to
    poison the rest of the run and the next one.
    """
    status, body = sess.call(
        "/api/auth/signup", "POST", {"email": email, "password": password, "displayName": label}
    )
    if status == 201:
        sess.user = body.get("user", {})
        print(f"  ....  created {label} account {email}")
        return True
    if status == 403 and body.get("error") == "signup_disabled":
        print(f"  ....  signups are closed; logging in as {email}")
    elif status != 409:
        check(f"{label} account available", False, f"signup got {status} {body.get('error')}")
        return False

    status, body = sess.call("/api/auth/login", "POST", {"email": email, "password": password})
    if status != 200:
        check(f"{label} signed in", False, f"login got {status} {body.get('error')}")
        return False
    sess.user = body.get("user", {})
    return True


def raw_get(path: str, headers: dict | None = None) -> tuple[int, str]:
    """A cookie-less request, for endpoints authenticated by token not session.

    Returns (status, content-type). The content type matters for the traversal
    check: nginx collapses "../" during URI normalisation, so a traversal
    attempt never reaches the API and lands on the SPA fallback with 200 HTML.
    That is correct and safe, but it means a status code alone cannot tell you
    whether a file was disclosed -- the content type can.
    """
    req = urllib.request.Request(BASE + path, headers={"User-Agent": UA} | (headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            res.read(1)  # touch the body without pulling 49 MB
            return res.status, res.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "")
    except urllib.error.URLError:
        return 0, ""


print(f"\n=== NexaLM end-to-end verification against {BASE} ===\n")

anon = Session("anon")
main = Session("primary")
alt = Session("secondary")

created_conversations: list[tuple[Session, str]] = []
original_defaults: dict | None = None

# ── deployment shape ────────────────────────────────────────
print("Deployment")
status, health = anon.call("/api/health")
check("health is public and ok", status == 200 and health.get("status") == "ok", f"got {status}")
check("database reachable", bool(health.get("db", {}).get("ok")), str(health.get("db")))

status, auth_status = anon.call("/api/auth/status")
check("auth status is public", status == 200, f"got {status}")
requires_token = bool(auth_status.get("requiresAccessToken"))
setup_required = bool(auth_status.get("setupRequired"))
print(
    f"  ....  signup={auth_status.get('signupEnabled')} google={auth_status.get('googleEnabled')} "
    f"accessGate={requires_token} setupRequired={setup_required}"
)

# ── access gate ─────────────────────────────────────────────
print("\nAccess control")
status, body = anon.call("/api/conversations")
# A 401 alone no longer distinguishes "locked" from "not signed in" -- the
# access-token gate and the session layer both answer 401 -- so assert the code.
expected = "locked" if requires_token else "unauthenticated"
check(
    "anonymous request refused",
    status == 401 and body.get("error") == expected,
    f"got {status} {body.get('error')}",
)

if requires_token:
    token = read_env("APP_ACCESS_TOKEN")
    if not token:
        skip("access token accepted", "APP_ACCESS_TOKEN not readable")
    else:
        status, _ = anon.call("/api/auth/unlock", "POST", {"token": "definitely-wrong"})
        check("wrong access token rejected", status == 401, f"got {status}")
        for s in (main, alt):
            s.call("/api/auth/unlock", "POST", {"token": token})
        status, body = main.call("/api/auth/unlock", "POST", {"token": token})
        check("correct access token unlocks", status == 200 and body.get("ok"), f"got {status}")
else:
    skip("access-token gate", "REQUIRE_ACCESS_TOKEN is off; the gate is inert")

# ── authentication ──────────────────────────────────────────
print("\nAuthentication")
email = read_env("E2E_EMAIL", "e2e-primary@nexalm.local")
password = read_env("E2E_PASSWORD")
alt_email = read_env("E2E_ALT_EMAIL", "e2e-secondary@nexalm.local")
alt_password = read_env("E2E_ALT_PASSWORD")
# No built-in defaults: a password in the source would sign anyone in to every
# deployment that never set its own.
if not password or not alt_password:
    print("Set E2E_PASSWORD and E2E_ALT_PASSWORD in .env or the environment.")
    sys.exit(2)

signed_in = ensure_account(main, email, password, "primary")
check("primary account signed in", signed_in, email)
if not signed_in:
    print("\n=== cannot continue without a session ===\n")
    sys.exit(1)

status, me = main.call("/api/auth/me")
check("session resolves to a user", status == 200 and me.get("user", {}).get("email") == email,
      f"got {status}")
is_admin = me.get("user", {}).get("role") == "admin"
print(f"  ....  primary role={me.get('user', {}).get('role')}")

alt_ready = ensure_account(alt, alt_email, alt_password, "secondary")
check("secondary account signed in", alt_ready, alt_email)

status, _ = main.call("/api/conversations")
check("signed-in access works", status == 200, f"got {status}")

# ── worker pool ─────────────────────────────────────────────
print("\nWorker pool")
status, pool = main.call("/api/workers")
check("pool reachable", status == 200, f"got {status}")
have_worker = pool.get("available") is True
if have_worker:
    check("a worker is available", True, f"connected={pool.get('connected')}")
else:
    skip("a worker is available", "no GPU worker connected — start one for full coverage")
for w in pool.get("workers", []):
    print(f"        {w['name']}  {w['status']}  engine={w['engine']}  model={w['model']}  gpus={w['gpuCount']}")

# Test against a model the pool actually serves, rather than whatever the global
# default happens to be. Routing no longer substitutes a different model when
# the requested one is absent, so a harness that assumed the default matched the
# connected worker would fail for reasons that have nothing to do with the code
# under test -- a Colab worker runs the reduced build and a Kaggle one does not.
served_model = pool.get("workers", [{}])[0].get("model") if have_worker else None
if served_model:
    print(f"  ....  exercising the pool's own model: {served_model}")

# ── conversation + streaming ────────────────────────────────
print("\nConversation and streaming")
cid = ""
job_id = ""
assistant_id = ""
status, body = main.call("/api/conversations", "POST", {})
cid = (body.get("conversation") or {}).get("id", "")
if check("conversation created", status == 200 and bool(cid), cid or str(body)):
    created_conversations.append((main, cid))
    if served_model:
        main.call(f"/api/conversations/{cid}", "PATCH", {"settings": {"model": served_model}})

if not cid:
    skip("streaming", "no conversation to send to")
elif not have_worker:
    skip("streaming", "no GPU worker connected")
else:
    def send(text: str):
        return main.call(f"/api/conversations/{cid}/messages", "POST", {"content": text})

    status, body = send("Explain what a MoE model is.")

    # 202 means the server accepted the turn and asked a worker to load a model
    # for it. That is now the ordinary state of a fresh session rather than an
    # edge case: workers no longer preload a chat model, because doing so cost
    # every session minutes loading whatever the notebook defaulted to. The
    # streaming assertions below need a real job, so wait for the load rather
    # than either failing or skipping them.
    if status == 202:
        print("        model is loading on demand; waiting for it (up to 8 min)")
        deadline = time.time() + 480
        while time.time() < deadline and status == 202:
            time.sleep(15)
            status, body = send("Explain what a MoE model is.")
        if status == 200:
            print("        loaded; continuing")

    check(
        "message accepted",
        status == 200,
        f"got {status} {body.get('error', '')}".strip(),
    )
    job_id = body.get("jobId", "")
    assistant_id = (body.get("assistantMessage") or {}).get("id", "")

    if not job_id:
        skip("stream", f"no jobId returned ({body.get('error')})")
    else:
        result = main.stream(job_id)
        check("stream produced text", len(result["text"]) > 50, f"{len(result['text'])} chars")
        check("tokens arrived incrementally", len(result["delta_times"]) > 5,
              f"{len(result['delta_times'])} deltas")
        if len(result["delta_times"]) > 2:
            span = result["delta_times"][-1] - result["delta_times"][0]
            check("deltas spread over time (real streaming, not one blob)", span > 0.3,
                  f"spread {span:.2f}s")
        check("first_token event seen", "first_token" in result["events"])
        check("terminal event is done", result["final"].get("_event") == "done",
              str(result["final"].get("_event")))
        final = result["final"]
        check("tokens/sec reported", final.get("tokensPerSec") is not None,
              str(final.get("tokensPerSec")))
        check("TTFT reported", final.get("ttftMs") is not None, f"{final.get('ttftMs')} ms")

# ── persistence ─────────────────────────────────────────────
print("\nPersistence")
if not (cid and assistant_id):
    skip("persistence", "no completed generation to read back")
else:
    time.sleep(1.5)
    status, thread = main.call(f"/api/conversations/{cid}")
    msgs = thread.get("messages", [])
    check("both turns persisted", len(msgs) == 2, f"{len(msgs)} messages")
    title = (thread.get("conversation") or {}).get("title", "")
    check("title auto-generated", title not in ("", "New chat"), title)

    assistant = next((m for m in msgs if m["id"] == assistant_id), None)
    check("assistant content saved", bool(assistant and len(assistant.get("content", "")) > 50))
    check("assistant marked complete", bool(assistant and assistant.get("status") == "complete"),
          assistant.get("status") if assistant else "?")

    if assistant:
        for field, label in (
            ("prompt_tokens", "prompt tokens persisted"),
            ("completion_tokens", "completion tokens persisted"),
            ("total_tokens", "total tokens persisted"),
            ("ttft_ms", "ttft persisted"),
            ("gen_ms", "gen time persisted"),
            ("tokens_per_sec", "tokens/sec persisted"),
            ("worker_name", "worker recorded"),
        ):
            check(label, assistant.get(field) is not None, str(assistant.get(field)))

    status, metrics = main.call(f"/api/conversations/{cid}/metrics")
    runs = int((metrics.get("metrics") or {}).get("runs", 0)) if status == 200 else 0
    check("conversation metrics aggregate", status == 200 and runs >= 1, f"runs={runs}")

# ── SSE replay ──────────────────────────────────────────────
print("\nStream replay")
if not job_id:
    skip("replay of a finished job", "no job to replay")
else:
    replay = main.stream(job_id)
    check("finished job still streams (reload resilience)",
          replay["final"].get("_event") == "done",
          str(replay["final"].get("_event")))
    check("replayed stream carries the text",
          len(replay["text"]) > 50 or bool(replay["final"].get("replayed")),
          f"{len(replay['text'])} chars replayed={replay['final'].get('replayed')}")

# ── model routing ───────────────────────────────────────
print()
print("Model routing")
if not have_worker:
    skip("routing refuses an absent model", "no GPU worker connected")
else:
    # Routing used to fall back to *any* worker when none advertised the model
    # asked for, so a request for one model was silently served by another. The
    # refusal is the fix; this asserts it, and that the refusal is useful.
    status, body = main.call("/api/conversations", "POST", {})
    probe = (body.get("conversation") or {}).get("id", "")
    if probe:
        created_conversations.append((main, probe))
        main.call(f"/api/conversations/{probe}", "PATCH",
                  {"settings": {"model": "no-such-model-exists"}})
        status, body = main.call(f"/api/conversations/{probe}/messages", "POST",
                                 {"content": "This must not be answered by another model."})
        # A model the catalogue has never heard of is refused outright rather
        # than queued: queueing it would leave the turn pending forever with no
        # way to tell a slow start from an impossible one.
        check("unknown model is refused, not substituted or queued",
              status == 503 and body.get("error") == "NO_WORKER_AVAILABLE",
              f"got {status} {body.get('error')}")
        offered = body.get("availableModels") or []
        check("refusal names the models that are available",
              len(offered) > 0 and "no-such-model-exists" not in offered,
              f"offered {offered}")
        thread_status, thread = main.call(f"/api/conversations/{probe}")
        check("no orphan message left behind by the refusal",
              len(thread.get("messages", [])) == 0,
              f"{len(thread.get('messages', []))} messages")
    else:
        skip("routing refuses an absent model", "could not create a probe conversation")

# ── what happens when the model is not loaded ───────────
print()
print("Model availability")
status, body = main.call("/api/conversations", "POST", {})
qid = (body.get("conversation") or {}).get("id", "")
catalogue = main.call("/api/models")[1].get("models", [])
if not qid:
    skip("model availability", "could not create a conversation")
else:
    created_conversations.append((main, qid))

    # A model too large for every connected GPU must be refused outright, with
    # the numbers. Queueing it would be a promise the server cannot keep: no
    # worker that could ever serve it is going to arrive.
    oversized = next((m["id"] for m in catalogue if m.get("availability") == "too_large"), None)
    if not oversized:
        skip("oversized model refused", "every catalogue model fits the connected GPU")
    else:
        main.call(f"/api/conversations/{qid}", "PATCH", {"settings": {"model": oversized}})
        status, body = main.call(f"/api/conversations/{qid}/messages", "POST",
                                 {"content": "This cannot run here."})
        check(f"a model too large for the GPU is refused ({oversized})",
              status == 503 and body.get("error") == "NO_WORKER_AVAILABLE",
              f"got {status} {body.get('error')}")
        check("the refusal explains the VRAM shortfall",
              "GB" in (body.get("message") or ""),
              (body.get("message") or "")[:90])
        status, thread = main.call(f"/api/conversations/{qid}")
        check("a refused turn leaves no orphan message",
              len(thread.get("messages", [])) == 0,
              f"{len(thread.get('messages', []))} messages")

    # With no model named, the pool answers with whatever it has loaded rather
    # than insisting on a global default the conversation never chose.
    if not have_worker:
        skip("unnamed model uses what is loaded", "no GPU worker connected")
    else:
        status, body = main.call("/api/conversations", "POST", {})
        aid = (body.get("conversation") or {}).get("id", "")
        if aid:
            created_conversations.append((main, aid))
            status, body = main.call(f"/api/conversations/{aid}/messages", "POST",
                                     {"content": "One short sentence, please."})
            check("a conversation that names no model is answered",
                  status == 200 and bool(body.get("jobId")),
                  f"got {status} {body.get('error', '')}")
            if body.get("jobId"):
                served = body.get("model")
                check("the reply names the model that answered",
                      served in (pool.get("workers", [{}])[0].get("loaded") or []),
                      str(served))
                main.stream(body["jobId"])

# ── per-conversation settings ───────────────────────────
print()
print("Conversation settings")
status, body = main.call("/api/conversations", "POST", {})
sid = (body.get("conversation") or {}).get("id", "")
if not sid:
    skip("conversation settings", "could not create a conversation")
else:
    created_conversations.append((main, sid))

    # This route used .parse() where the settings route uses safeParse, so an
    # unknown key escaped as a 500 instead of a 400.
    status, body = main.call(f"/api/conversations/{sid}", "PATCH",
                             {"settings": {"nonsenseKey": 1}})
    check("an unknown setting is a 400, not a 500",
          status == 400 and body.get("error") == "invalid_settings",
          f"got {status} {body.get('error')}")

    # Both keys live in the shared ChatSettings type the browser sends, but were
    # missing from this route's schema entirely.
    status, _ = main.call(f"/api/conversations/{sid}", "PATCH",
                          {"settings": {"autoStartWorker": True}})
    check("a deployment-wide setting is accepted per conversation", status == 200,
          f"got {status}")

    # Assigning the whole object meant choosing a model silently discarded
    # every other override on the conversation.
    main.call(f"/api/conversations/{sid}", "PATCH", {"settings": {"temperature": 0.42}})
    main.call(f"/api/conversations/{sid}", "PATCH", {"settings": {"model": "qwen3-30b-a3b-q3"}})
    status, listing = main.call("/api/conversations")
    row = next((c for c in listing.get("conversations", []) if c["id"] == sid), {})
    kept = (row.get("settings") or {})
    check("choosing a model keeps the other overrides",
          kept.get("temperature") == 0.42 and kept.get("model") == "qwen3-30b-a3b-q3",
          str(kept))

# ── web search ──────────────────────────────────────────
print()
print("Web search")
if not have_worker:
    skip("web search", "no GPU worker connected")
else:
    status, body = main.call("/api/conversations", "POST", {"title": "search probe"})
    wid = (body.get("conversation") or {}).get("id", "")
    if not wid:
        skip("web search", "could not create a conversation")
    else:
        created_conversations.append((main, wid))
        main.call(f"/api/conversations/{wid}", "PATCH",
                  {"settings": {"model": served_model, "webSearch": True}})
        status, body = main.call(f"/api/conversations/{wid}/messages", "POST",
                                 {"content": "What is a transformer neural network?"})
        found = body.get("sources") or []
        check("a search runs before the answer", status == 200 and len(found) > 0,
              f"got {status}, {len(found)} sources")
        check("each source carries a real URL",
              all(str(x.get("url", "")).startswith("http") for x in found),
              ", ".join(x.get("source", "?") for x in found[:3]))
        if body.get("jobId"):
            main.stream(body["jobId"])
            time.sleep(1.5)
            status, thread = main.call(f"/api/conversations/{wid}")
            asst = [m for m in thread.get("messages", []) if m["role"] == "assistant"]
            kept = (asst[-1].get("sources") or []) if asst else []
            # Without this the numbered citations in the text point at nothing
            # once the page is reloaded, which is worse than no citations.
            check("citations survive a reload", len(kept) == len(found),
                  f"{len(kept)} kept of {len(found)}")

# ── multi-user isolation ──────────────────────────────────────
print("\nMulti-user isolation")
if not (alt_ready and cid):
    skip("isolation", "need two accounts and a conversation")
else:
    status, listing = alt.call("/api/conversations")
    others = [c for c in listing.get("conversations", []) if c["id"] == cid]
    check("second account cannot see the first's conversation in its list",
          status == 200 and not others, f"{len(listing.get('conversations', []))} own conversations")

    status, _ = alt.call(f"/api/conversations/{cid}")
    check("second account gets 404 reading a foreign conversation", status == 404, f"got {status}")

    status, _ = alt.call(f"/api/conversations/{cid}", "PATCH", {"title": "hijacked"})
    check("second account cannot rename a foreign conversation", status == 404, f"got {status}")

    status, _ = alt.call(f"/api/conversations/{cid}", "DELETE")
    check("second account cannot delete a foreign conversation", status == 404, f"got {status}")

    if job_id:
        status, _ = alt.call(f"/api/jobs/{job_id}/cancel", "POST", {})
        check("second account cannot cancel a foreign job", status == 404, f"got {status}")
        # The SSE route is the one that leaks tokens if ownership is missed.
        code = 0
        try:
            req = urllib.request.Request(
                BASE + f"/api/jobs/{job_id}/stream", headers={"User-Agent": UA}
            )
            with alt.opener.open(req, timeout=30) as res:
                code = res.status
        except urllib.error.HTTPError as e:
            code = e.code
        check("second account cannot stream a foreign job", code == 404, f"got {code}")
    else:
        skip("foreign job checks", "no job id")

# ── admin surface ───────────────────────────────────────────
print("\nAdmin surface")
if alt_ready and not alt.user.get("role") == "admin":
    status, body = alt.call("/api/users")
    check("non-admin refused the user list", status == 403 and body.get("error") == "forbidden",
          f"got {status} {body.get('error')}")

    # Starting a session spends a finite weekly GPU quota, so it must not be
    # reachable by an ordinary account. Only the refusal is exercised here --
    # the harness must never actually start a session.
    for path, label in (("/api/workers/start", "start"), ("/api/workers/stop", "stop")):
        status, body = alt.call(path, "POST", {})
        check(f"non-admin refused worker {label}",
              status == 403 and body.get("error") == "forbidden",
              f"got {status} {body.get('error')}")
else:
    skip("non-admin refusal", "secondary account is admin")

if is_admin:
    status, users = main.call("/api/users")
    check("admin can list users", status == 200 and isinstance(users.get("users"), list),
          f"{len(users.get('users', []))} users")
    status, body = main.call(f"/api/users/{me['user']['id']}", "PATCH", {"role": "user"})
    check("admin cannot demote themselves",
          status == 403 and body.get("error") == "self_role_change",
          f"got {status} {body.get('error')}")
    status, body = main.call(f"/api/users/{me['user']['id']}", "DELETE")
    check("admin cannot delete themselves",
          status == 403 and body.get("error") == "self_delete",
          f"got {status} {body.get('error')}")
else:
    skip("admin endpoints", "primary account is not an admin")

# ── worker artifacts ────────────────────────────────────────
print("\nWorker artifacts")
worker_token = read_env("WORKER_TOKEN")
code, _ = raw_get("/api/artifacts")
check("artifact listing refuses an anonymous caller", code == 401, f"got {code}")
if worker_token:
    auth = {"Authorization": "Bearer " + worker_token}
    code, _ = raw_get("/api/artifacts", auth)
    check("artifact listing accepts the worker token", code == 200, f"got {code}")

    code, ctype = raw_get("/api/artifacts/nope.tar.gz", auth)
    check("unknown artifact name refused", code == 404 and "json" in ctype,
          f"got {code} {ctype}")

    # Assert the security property, not a status code: nginx normalises the
    # traversal away before the API sees it, so this legitimately answers 200
    # with the SPA shell. What must never happen is a file coming back.
    for attempt in ("../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "....//....//etc/passwd"):
        code, ctype = raw_get(
            "/api/artifacts/" + urllib.parse.quote(attempt, safe=""), auth
        )
        served = "gzip" in ctype or "octet-stream" in ctype
        check(f"path traversal discloses nothing ({attempt[:18]})", not served,
              f"got {code} {ctype}")
else:
    skip("artifact token checks", "WORKER_TOKEN not readable")

# ── cancellation ────────────────────────────────────────────
print("\nCancellation")
if not (cid and have_worker):
    skip("cancellation", "needs a worker and a conversation")
else:
    status, body = main.call(
        f"/api/conversations/{cid}/messages", "POST",
        {"content": "Write a very long essay about distributed systems."},
    )
    cancel_job = body.get("jobId", "")
    if not cancel_job:
        skip("cancellation", f"no jobId ({body.get('error')})")
    else:
        # Cancel only AFTER real tokens have arrived. A fixed sleep was wrong:
        # on a cold worker time-to-first-token can exceed 9s, so the cancel
        # landed before generation began and "no partial text" was correct
        # behaviour being reported as a failure.
        collected = {"deltas": 0}
        first_token = threading.Event()

        def _consume() -> None:
            try:
                req = urllib.request.Request(
                    BASE + f"/api/jobs/{cancel_job}/stream", headers={"User-Agent": UA}
                )
                with main.opener.open(req, timeout=180) as res:
                    event = None
                    for raw_line in res:
                        line = raw_line.decode(errors="replace").rstrip(chr(10))
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:") and event == "delta":
                            collected["deltas"] += 1
                            first_token.set()
                        elif line.startswith("data:") and event in ("done", "error"):
                            break
            except Exception:  # noqa: BLE001 — the reader is best-effort
                pass

        reader = threading.Thread(target=_consume, daemon=True)
        reader.start()

        got_tokens = first_token.wait(timeout=90)
        check("generation started before cancel", got_tokens, f"{collected['deltas']} deltas seen")
        time.sleep(0.8)  # let a little more text accumulate

        status, _ = main.call(f"/api/jobs/{cancel_job}/cancel", "POST", {})
        check("cancel accepted", status == 200, f"got {status}")
        reader.join(timeout=20)

        time.sleep(2.0)
        status, thread = main.call(f"/api/conversations/{cid}")
        cancelled = [m for m in thread.get("messages", []) if m.get("status") == "cancelled"]
        check("message marked cancelled", len(cancelled) >= 1, f"{len(cancelled)} cancelled")
        if got_tokens and cancelled:
            check("partial text kept", len(cancelled[0].get("content", "")) > 0,
                  f"{len(cancelled[0].get('content', ''))} chars")

# ── settings ────────────────────────────────────────────────
print("\nSettings")
try:
    status, s = main.call("/api/settings")
    ok = check("settings readable", status == 200 and "defaults" in s,
               f"{len(s.get('models', []))} models")
    if ok:
        original_defaults = s["defaults"]
        status, saved = main.call("/api/settings", "PUT", {**s["defaults"], "temperature": 0.55})
        check("settings writable",
              status == 200 and saved.get("defaults", {}).get("temperature") == 0.55,
              f"got {status}")
finally:
    # app_settings is a single global row shared by every user, so a run that
    # dies here leaves everyone's temperature at 0.55. Restore unconditionally
    # and prove the restore landed.
    if original_defaults is not None:
        status, restored = main.call("/api/settings", "PUT", original_defaults)
        check("settings restored to their original values",
              status == 200
              and restored.get("defaults", {}).get("temperature")
              == original_defaults.get("temperature"),
              f"temperature={restored.get('defaults', {}).get('temperature')}")

# ── history ─────────────────────────────────────────────────
print("\nHistory")
status, listing = main.call("/api/conversations")
check("conversation appears in history",
      any(c["id"] == cid for c in listing.get("conversations", [])),
      f"{len(listing.get('conversations', []))} conversations")

# ── session teardown ────────────────────────────────────────
print("\nSession lifecycle")
status, _ = main.call("/api/auth/logout", "POST", {})
check("logout accepted", status == 200, f"got {status}")
status, _ = main.call("/api/conversations")
check("session no longer works after logout", status == 401, f"got {status}")

# Clean up what this run created. Accounts are deliberately kept (see module
# docstring); conversations are not.
if created_conversations:
    if ensure_account(main, email, password, "primary"):
        removed = 0
        for sess, conv in created_conversations:
            code, _ = main.call(f"/api/conversations/{conv}", "DELETE")
            removed += 1 if code in (200, 204) else 0
        print(f"  ....  cleaned up {removed}/{len(created_conversations)} test conversations")

print(f"\n=== {passed} passed, {failed} failed, {skipped} skipped ===\n")
sys.exit(1 if failed else 0)

"""Subprocess side of F15's baseline-versus-patched HTTP regression."""
import base64
import builtins
import hashlib
import http.client
import io
import json
import os
import re
import sqlite3
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

SELECTED_HEADERS = (
    "content-type", "cache-control", "x-content-type-options", "referrer-policy",
    "x-frame-options", "content-security-policy", "content-length", "accept-ranges",
    "content-range", "set-cookie",
)
TIME_COLUMNS = {"created", "at", "expires", "updated"}


def canonical_response(status, headers, body):
    selected = {}
    for key, value in headers:
        lower = key.lower()
        if lower == "date" or lower not in SELECTED_HEADERS:
            continue
        if lower == "set-cookie":
            value = re.sub(r"plaud_session=[^;]+", "plaud_session=<token>", value)
        selected[lower] = value
    if selected.get("content-type", "").startswith("application/json"):
        try:
            value = json.loads(body)
            if isinstance(value, dict) and "upload_id" in value:
                value["upload_id"] = "<token>"
            if isinstance(value, dict) and isinstance(value.get("url"), str):
                value["url"] = re.sub(r"(?<=ticket=)[^&]+", "<token>", value["url"])
            body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")).encode()
        except (ValueError, TypeError):
            pass
    return {"status": status, "headers": selected,
            "body_b64": base64.b64encode(body).decode("ascii")}


def dump_state(fixture):
    database = fixture / "db.sqlite"
    db = sqlite3.connect(database); db.row_factory = sqlite3.Row
    tables = [row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    result = {}
    for table in tables:
        columns = [row[1] for row in db.execute("PRAGMA table_info(%s)" % table)]
        rows = []
        for row in db.execute("SELECT * FROM %s" % table):
            item = {column: "<ts>" if column in TIME_COLUMNS else row[column]
                    for column in columns}
            rows.append(item)
        rows.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
        result[table] = rows
    db.close()
    files = []
    for path in sorted(fixture.rglob("*")):
        if not path.is_file() or path == database or path.name in ("db.sqlite-wal", "db.sqlite-shm"):
            continue
        relative = path.relative_to(fixture).as_posix()
        relative = re.sub(r"(?<=service/uploads/)[0-9a-f]{32}$", "<token>", relative)
        files.append({"path": relative,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"tables": result, "files": files}


def main():
    fixture = Path(sys.argv[1]).resolve(); os.chdir(fixture)
    from plaud_workflow import store, web
    label_packet = Path(os.environ["F15_LABEL_PACKET"]).resolve()
    real_connect = sqlite3.connect; real_open = builtins.open; real_io_open = io.open

    def denied_write(path):
        try: resolved = Path(path).resolve()
        except TypeError: return False
        text = resolved.as_posix() + "/"
        return ("/data/pipeline/" in text or "/runtime/" in text or
                resolved == label_packet or label_packet in resolved.parents)

    def guarded_connect(path, *args, **kwargs):
        if denied_write(path): raise AssertionError("production sqlite write refused: " + str(path))
        return real_connect(path, *args, **kwargs)

    def guard_open(real):
        def guarded(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+") and denied_write(path):
                raise AssertionError("production file write refused: " + str(path))
            return real(path, mode, *args, **kwargs)
        return guarded

    sqlite3.connect = guarded_connect
    builtins.open = guard_open(real_open); io.open = guard_open(real_io_open)

    now = 2_000_000_000.0
    web.time.time = lambda: now; store.time.time = lambda: now
    counter = {"value": 0}

    def token_hex(size):
        counter["value"] += 1
        return ("%0*x" % (size * 2, counter["value"]))[-size * 2:]

    def token_urlsafe(_size):
        counter["value"] += 1
        return "fixture-ticket-%04d" % counter["value"]

    class FixtureUUID:
        def __init__(self, value): self.hex = "%032x" % value

    def uuid4():
        counter["value"] += 1
        return FixtureUUID(counter["value"])

    web.secrets.token_hex = token_hex; web.secrets.token_urlsafe = token_urlsafe
    store.uuid.uuid4 = uuid4
    web.fetch_reference = lambda *_args, **_kwargs: [
        {"start": 0.0, "end": 8.0, "speaker": "Reference", "text": "fixture words"}]
    web.embedding = lambda *_args, **_kwargs: [1.0, 0.0]
    web.probe = lambda *_args, **_kwargs: (5.0, {"fixture": True})
    web.publish_pending = lambda *_args, **_kwargs: None
    web.review = lambda *_args, **_kwargs: {
        "available": True, "proposals": [{"key": "unknown-1", "checked": None}],
        "held_back": 0, "unknown_words": 2, "offered_words": 2,
    }
    web.subprocess.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("subprocess.run is denied in the F15 fixture"))
    config = {
        "root": str(fixture / "service"), "database": str(fixture / "db.sqlite"),
        "base_url": "http://fixture.test", "require_session": True, "phase": "fixture",
        "notes": str(fixture / "notes"), "cli": str(fixture / "denied-cli"),
        "runtimes": {"ffmpeg": str(fixture / "denied-ffmpeg")},
        "label_packet": os.environ["F15_LABEL_PACKET"],
        "label_root": os.environ["F15_LABEL_ROOT"],
    }
    app = web.Application(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.handler(app)); server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    responses = {}; post_states = {}

    def request(name, path, method="GET", data=None, headers=None, raw=None,
                content_type="application/json", record_state=False):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        values = {"Host": "fixture.test"}; values.update(headers or {})
        if raw is None and data is not None:
            raw = json.dumps(data, ensure_ascii=False).encode()
        if raw is not None: values.setdefault("Content-Type", content_type)
        connection.request(method, path, body=raw, headers=values)
        response = connection.getresponse(); body = response.read()
        responses[name] = canonical_response(response.status, response.getheaders(), body)
        connection.close()
        if record_state: post_states[name] = dump_state(fixture)
        return response.status, dict(response.getheaders()), body

    try:
        request("get_unsigned_sign_in", "/")
        request("head_unsigned_sign_in", "/", method="HEAD")
        request("get_unsigned_api", "/api/status")
        request("head_unsigned_api", "/api/status", method="HEAD")
        request("get_host_mismatch", "/api/status", headers={"Host": "evil.example"})
        request("get_cross_site", "/api/status", headers={"Sec-Fetch-Site": "cross-site"})

        _, _, body = request("post_bootstrap_good", "/bootstrap", method="POST", data={},
                             headers={"X-Local-Key": app.key}, record_state=True)
        ticket = json.loads(body)["url"].split("ticket=", 1)[1]
        status, login_headers, _ = request("get_open_ticket", "/open?ticket=" + ticket)
        if status != 303: raise AssertionError("fixture login failed")
        cookie = login_headers["Set-Cookie"].split(";", 1)[0]
        auth = {"Cookie": cookie}; post = {"Cookie": cookie, "Origin": "http://fixture.test"}

        html_paths = (
            ("root", "/"), ("recording", "/recordings/r1"),
            ("compare", "/compare/r1"), ("unknowns", "/unknowns/r1"),
            ("profiles", "/profiles"), ("lookup", "/lookup"), ("library", "/library"),
        )
        api_paths = (
            ("status", "/api/status"), ("progress", "/api/progress"),
            ("reconciliation", "/api/reconciliation"),
            ("compare_api", "/api/compare/r1"), ("recordings", "/api/recordings"),
            ("profiles_api", "/api/profiles"), ("recording_api", "/api/recordings/r1"),
            ("profile_samples", "/api/profiles/profile-1/samples"),
            ("unknowns_api", "/api/unknowns/r1"),
        )
        for label, path in html_paths + api_paths:
            request("get_" + label, path, headers=auth)
            request("head_" + label, path, method="HEAD", headers=auth)
        request("get_audio_full", "/audio/r1", headers=auth)
        request("head_audio_full", "/audio/r1", method="HEAD", headers=auth)
        request("get_audio_range", "/audio/r1", headers=dict(auth, Range="bytes=10-19"))
        request("head_audio_range", "/audio/r1", method="HEAD", headers=dict(auth, Range="bytes=10-19"))
        request("get_audio_416", "/audio/r1", headers=dict(auth, Range="bytes=999-"))
        request("head_audio_416", "/audio/r1", method="HEAD", headers=dict(auth, Range="bytes=999-"))
        request("get_unknown_route", "/not-a-route", headers=auth)
        request("head_unknown_route", "/not-a-route", method="HEAD", headers=auth)
        request("get_label", "/label", headers=auth)
        request("head_label", "/label", method="HEAD", headers=auth)
        request("post_bootstrap_wrong", "/bootstrap", method="POST", data={},
                headers={"X-Local-Key": "wrong"}, record_state=True)
        request("post_missing_origin", "/api/profiles", method="POST",
                data={"name": "No origin", "key": "missing-origin"},
                headers=auth, record_state=True)
        request("post_non_json", "/api/profiles", method="POST", raw=b"not-json",
                content_type="text/plain", headers=post, record_state=True)
        request("post_unknown_verdict", "/api/unknowns/r1/verdict", method="POST",
                data={"choice": "right", "key": "unknown-1", "proposed": "S01",
                      "evidence": "fixture", "at": 1.0, "text": "fixture words"},
                headers=post, record_state=True)
        request("post_compare_verdict", "/api/compare/r1/verdict", method="POST",
                data={"choice": "reference", "key": "diff-1", "at": 1.0,
                      "reference": "fixture", "ours": "words"},
                headers=post, record_state=True)
        request("post_reconciliation", "/api/reconciliation/run", method="POST", data={},
                headers=post, record_state=True)
        request("post_retry", "/api/retry", method="POST", data={"recording_id": "r1"},
                headers=post, record_state=True)
        request("post_profiles", "/api/profiles", method="POST",
                data={"name": "Second profile", "key": "profile-request-1"},
                headers=post, record_state=True)
        request("post_lookup", "/api/lookup", method="POST",
                data={"recording_id": "r1", "start": 0.0, "end": 5.0},
                headers=post, record_state=True)
        request("post_identify", "/api/identify", method="POST",
                data={"recording_id": "r1", "segment_id": "s1", "profile_id": "profile-1",
                      "scope": "segment", "version": 1, "key": "identify-1",
                      "start": 0.0, "end": 5.0, "confirm_conflicts": False},
                headers=post, record_state=True)
        request("post_upload", "/api/upload", method="POST",
                data={"audio": base64.b64encode(b"fixture-audio").decode()},
                headers=post, record_state=True)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5); app.db.close()
    print(json.dumps({"responses": responses, "post_states": post_states},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

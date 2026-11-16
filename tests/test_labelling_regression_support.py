"""F15: finite, isolated baseline-versus-patched route regression."""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from plaud_workflow.store import Store, encoded

EXPECTED_ATOMS = {
    "GET": {"/open", "/api", "/audio", "/api/status", "/api/progress",
            "/api/compare/", "/api/reconciliation", "/api/recordings", "/api/profiles",
            "/api/recordings/", "/api/profiles/", "/samples", "/api/unknowns/",
            "/audio/", "/", "/recordings/", "/compare/", "/unknowns/",
            "/profiles", "/lookup", "/library"},
    "POST": {"/", "/open?ticket=", "/bootstrap", "/api/unknowns/", "/verdict", "/api/compare/",
             "/api/retry", "/api/reconciliation/run", "/api/profiles", "/api/lookup", "/api/identify",
             "/api/upload"},
}


def strip_dispatch(raw):
    lines = raw.splitlines(keepends=True)
    removed = [line for line in lines if b"labelling.handle" in line]
    if len(removed) != 2:
        raise AssertionError("web.py must contain exactly two labelling dispatch lines")
    if not all(b"route=='/label'" in line and b"from . import labelling" in line
               for line in removed):
        raise AssertionError("unexpected labelling dispatch shape")
    return b"".join(line for line in lines if b"labelling.handle" not in line)


def route_atoms(source):
    tree = ast.parse(source)
    result = {}
    for method in ("do_GET", "do_POST"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method)
        values = {n.value for n in ast.walk(node)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and n.value.startswith("/")}
        result[method[3:]] = values
    return result


def required_cases():
    cases = {"get_unsigned_sign_in", "head_unsigned_sign_in", "get_unsigned_api",
             "head_unsigned_api", "get_host_mismatch", "get_cross_site",
             "post_bootstrap_good", "get_open_ticket", "get_audio_full", "head_audio_full", "get_audio_range",
             "head_audio_range", "get_audio_416", "head_audio_416", "get_unknown_route",
             "head_unknown_route", "get_label", "head_label", "post_bootstrap_wrong",
             "post_missing_origin", "post_non_json", "post_unknown_verdict",
             "post_compare_verdict", "post_reconciliation", "post_retry", "post_profiles", "post_lookup",
             "post_identify", "post_upload"}
    labels = ("root", "recording", "compare", "unknowns", "profiles", "lookup", "library",
              "status", "progress", "reconciliation", "compare_api", "recordings", "profiles_api",
              "recording_api", "profile_samples", "unknowns_api")
    for label in labels:
        cases.add("get_" + label); cases.add("head_" + label)
    return cases


def make_fixture(root):
    root.mkdir(); (root / "service").mkdir(); (root / "notes").mkdir()
    (root / "service/launcher.key").write_text("fixture-local-key")
    audio = root / "fixture.wav"; audio.write_bytes(bytes(range(100)))
    store = Store(root / "db.sqlite")
    store.recording("r1", "Fixture recording", "source-hash", audio, 10,
                    {"plaud_id": "wire-r1", "asr_model": "fixture", "asr_backend": "fixture"})
    store.transcript("r1", [
        {"id": "s1", "start": 0.0, "end": 8.0, "speaker": "S01", "text": "fixture words"},
        {"id": "s2", "start": 8.0, "end": 10.0, "speaker": "UNASSIGNED", "text": "later"},
    ])
    store.db.execute("INSERT INTO profiles(id,name,created) VALUES(?,?,?)",
                     ("profile-1", "Fixture person", 1000.0))
    store.db.execute("""INSERT INTO samples
      (id,recording_id,segment_id,profile_id,start,end,interval_key,embedding,model,
       confidence,active,provenance,created,modality)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      ("sample-1", "r1", "s1", "profile-1", 5.0, 8.0, "interval-1",
       encoded([1.0, 0.0]), "wespeaker-cmn-v1", 1.0, 1, encoded({"fixture": True}),
       1000.0, "voice"))
    store.db.execute("INSERT INTO suggestions VALUES(?,?,?,?,?)",
                     ("r1", "S01", "profile-1", 0.9, 1000.0))
    store.db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                     ("r1", "summary", "failed", 1, 0.0, "fixture failure", 1000.0))
    store.put("baseline_complete", True); store.put("service_error", None)
    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); store.close()


def make_label_fixture(root):
    from plaud_workflow import labelling
    packet = root / "label-packet"; (packet / "audio").mkdir(parents=True)
    audio = packet / "audio/opaque.wav"; audio.write_bytes(b"fixture label audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    (packet / "manifest.json").write_text(json.dumps({
        "contains_provider_output": False,
        "created_utc": "2026-10-27T00:00:00Z",
        "tasks": [{
            "audio": "audio/opaque.wav", "audio_sha256": digest,
            "excerpt_s": 125.0, "excerpt_start_s": 0.0, "family": "opaque",
            "long_session_region": False, "opaque_id": "opaque",
        }],
    }))
    labels = root / "label-root"
    labelling.initialize(packet, labels, "a" * 64)
    return packet, labels


def run_worker(worker, source, fixture, packet, labels):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source)
    env["F15_LABEL_PACKET"] = str(packet); env["F15_LABEL_ROOT"] = str(labels)
    completed = subprocess.run(
        [sys.executable, str(worker), str(fixture)], cwd=fixture, env=env,
        text=True, capture_output=True, timeout=60)
    if completed.returncode:
        raise AssertionError("F15 worker failed:\n" + completed.stdout + completed.stderr)
    return json.loads(completed.stdout)


class ExistingRouteRegression:
    @classmethod
    def run(cls):
        root = Path(__file__).resolve().parents[1]
        package = root / "plaud_workflow"
        current_web = (package / "web.py").read_bytes()
        baseline_web = strip_dispatch(current_web)
        inventory = route_atoms(baseline_web.decode())
        if inventory != EXPECTED_ATOMS:
            raise AssertionError("baseline route inventory changed: %r" % inventory)
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            baseline_source = temporary / "baseline-source"
            patched_source = temporary / "patched-source"
            shutil.copytree(package, baseline_source / "plaud_workflow")
            shutil.copytree(package, patched_source / "plaud_workflow")
            (baseline_source / "plaud_workflow/web.py").write_bytes(baseline_web)
            template = temporary / "template"; make_fixture(template)
            baseline_fixture = temporary / "baseline-fixture"
            patched_fixture = temporary / "patched-fixture"
            shutil.copytree(template, baseline_fixture)
            shutil.copytree(template, patched_fixture)
            packet, labels = make_label_fixture(temporary)
            worker = root / "tests/test_labelling_regression_worker.py"
            baseline = run_worker(worker, baseline_source, baseline_fixture, packet, labels)
            patched = run_worker(worker, patched_source, patched_fixture, packet, labels)
        missing = sorted(required_cases() - set(baseline["responses"]))
        unexpected = []
        for case in sorted(required_cases() - {"get_label", "head_label"}):
            if baseline["responses"].get(case) != patched["responses"].get(case):
                unexpected.append("response:" + case)
        for method in ("get", "head"):
            case = method + "_label"
            if baseline["responses"][case]["status"] != 404:
                unexpected.append("baseline:" + case + "_not_404")
            if patched["responses"][case]["status"] != 200:
                unexpected.append("patched:" + case + "_not_200")
        if set(baseline["post_states"]) != set(patched["post_states"]):
            unexpected.append("post-state-inventory")
        for case in sorted(set(baseline["post_states"]) & set(patched["post_states"])):
            if baseline["post_states"][case] != patched["post_states"][case]:
                unexpected.append("post-state:" + case)
        return {"unexpected_differences": unexpected, "missing_cases": missing,
                "inventory": inventory, "response_cases": len(baseline["responses"]),
                "post_state_cases": len(baseline["post_states"]),
                "baseline_web_sha256": hashlib.sha256(baseline_web).hexdigest(),
                "patched_web_sha256": hashlib.sha256(current_web).hexdigest()}

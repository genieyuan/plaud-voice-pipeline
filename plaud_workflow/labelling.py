"""Blind single-annotator diarization labelling storage and HTTP routes."""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import http.client
import json
import math
import os
import random
import re
import stat
import urllib.parse
import wave
from pathlib import Path

SEED = 20261006
END_TOLERANCE_S = 0.001
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT / "data/labelling/diar-20261006"
SPEAKER = re.compile(r"^spk[1-9][0-9]?$")

class LabelConflict(Exception):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value, mode=0o600):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode() + b"\n"
    temp = path.with_name("." + path.name + ".tmp")
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temp, mode); os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _exclusive(path, raw, mode=0o400):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(path, mode)
    except BaseException:
        Path(path).unlink(missing_ok=True); raise


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _regular(path):
    try:
        value = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode)


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


@contextlib.contextmanager
def _locked(labels):
    lock = Path(labels) / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield

def _manifest_rows(packet):
    packet = Path(packet)
    manifest_path = packet / "manifest.json"
    if not _regular(manifest_path):
        raise ValueError("Packet manifest is unavailable")
    value = json.loads(manifest_path.read_text())
    if isinstance(value, list):
        rows = value
    else:
        rows = value.get("tasks") or value.get("items") or value.get("excerpts") or value.get("clips")
    if not isinstance(rows, list):
        raise ValueError("Packet manifest has no task inventory")
    result = []
    for item in rows:
        task = str(item.get("task") or item.get("id") or item.get("opaque_id") or "")
        family = str(item.get("family") or item.get("family_id") or task)
        seconds = float(item.get("excerpt_s") or item.get("seconds") or
                        item.get("duration_s") or item.get("duration") or 0)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
            raise ValueError("Packet task IDs must be opaque path-safe identifiers")
        if not math.isfinite(seconds): raise ValueError("Packet duration is invalid")
        relative = item.get("audio") or item.get("path") or item.get("file") or f"audio/{task}.wav"
        audio = Path(relative)
        if not audio.is_absolute():
            audio = packet / audio
        expected = item.get("audio_sha256") or item.get("sha256") or item.get("hash")
        if not task or not family or seconds <= 0 or not expected:
            raise ValueError("Packet manifest task is incomplete")
        if task in {row["task"] for row in result}:
            raise ValueError("Packet manifest has duplicate tasks")
        result.append({"task": task, "family": family, "seconds": seconds,
                       "audio": str(audio), "audio_sha256": str(expected)})
    return result


def _repeat_id(prototype, slot):
    digest = hashlib.sha256(f"{SEED}:repeat:{slot}".encode()).hexdigest()
    return digest[:max(8, len(prototype))]


def initialize(packet, labels, retention_frozen_sha256):
    packet = Path(packet).resolve()
    labels = Path(labels)
    if labels.exists():
        raise FileExistsError("Label root already exists")
    if not re.fullmatch(r"[0-9a-f]{64}", str(retention_frozen_sha256)):
        raise ValueError("Retention frozen digest must be sha256")
    rows = _manifest_rows(packet)
    if not rows:
        raise ValueError("Packet has no tasks")
    rng = random.Random(SEED)
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    families = []
    for family in sorted(by_family):
        eligible = sorted((row for row in by_family[family] if row["seconds"] >= 120),
                          key=lambda row: row["task"])
        if eligible:
            families.append(eligible[0])
    rng.shuffle(families)
    planned = {row["task"] for row in families}
    originals = families + sorted((row for row in rows if row["task"] not in planned),
                                  key=lambda row: row["task"])
    plans = []
    for row in families:
        starts = list(range(0, int(math.floor(row["seconds"])) - 59))
        rng.shuffle(starts)
        plans.append({"family": row["family"], "source_task": row["task"],
                      "candidate_starts": starts})
    occupied = {row["task"] for row in rows}
    repeat_ids = []
    for slot in range(1, 4):
        task = _repeat_id(rows[0]["task"], slot)
        while task in occupied:
            task = hashlib.sha256(task.encode()).hexdigest()[:max(8, len(rows[0]["task"]))]
        occupied.add(task); repeat_ids.append(task)
    queue = [{"task": row["task"], "kind": "original"} for row in originals]
    positions = sorted(zip(repeat_ids, rng.sample(range(len(queue) + 3), 3)),
                       key=lambda item: item[1])
    for task, position in positions:
        queue.insert(position, {"task": task, "kind": "repeat"})
    repeat_map = {"schema": "plaud-repeat-map/1", "seed": SEED,
                  "family_plan": plans, "queue": queue,
                  "slots": [{"slot": i + 1, "task": task, "state": "locked"}
                            for i, task in enumerate(repeat_ids)]}
    labels.mkdir(mode=0o700); os.chmod(labels, 0o700)
    for name in ("private", "drafts", "submitted", "repeats"):
        (labels / name).mkdir(mode=0o700)
    _atomic_json(labels / "private/repeat-map.json", repeat_map, 0o400)
    repeat_sha = _sha256(labels / "private/repeat-map.json")
    _atomic_json(labels / "init.json", {
        "schema": "plaud-labelling-init/1", "packet": str(packet),
        "manifest_sha256": _sha256(packet / "manifest.json"),
        "expected_tasks": len(queue),
        "retention_frozen_sha256": retention_frozen_sha256,
        "repeat_map_sha256": repeat_sha, "created_utc": _utc()})
    (labels / "ledger.jsonl").touch(mode=0o600)
    return {"tasks": len(queue), "repeat_map_sha256": repeat_sha}

def load_state(packet, labels):
    packet = Path(packet).resolve()
    labels = Path(labels)
    init = json.loads((labels / "init.json").read_text())
    if Path(init["packet"]).resolve() != packet:
        raise ValueError("Label root belongs to another packet")
    if _sha256(packet / "manifest.json") != init["manifest_sha256"]:
        raise ValueError("Packet manifest changed after init")
    repeat_map = json.loads((labels / "private/repeat-map.json").read_text())
    rows = _manifest_rows(packet)
    manifest = {row["task"]: row for row in rows}
    return {"packet": packet, "labels": labels, "init": init,
            "manifest": manifest, "repeat_map": repeat_map}


def _slot(state, task):
    for slot in state["repeat_map"]["slots"]:
        if slot["task"] == task:
            return slot
    return None


def task_info(state, task):
    if task in state["manifest"]:
        return dict(state["manifest"][task], kind="original")
    slot = _slot(state, task)
    if not slot or slot.get("state") == "void":
        raise KeyError("Unknown task")
    result = {"task": task, "kind": "repeat", "seconds": 60.0}
    if slot.get("audio_sha256"):
        result.update(family=slot["family"], audio=slot["audio"],
                      audio_sha256=slot["audio_sha256"])
    return result


def task_state(state, task):
    info = task_info(state, task)
    submitted = state["labels"] / "submitted" / f"{task}.rttm"
    draft = state["labels"] / "drafts" / f"{task}.json"
    if submitted.exists():
        return "submitted"
    if info["kind"] == "repeat":
        slot = _slot(state, task)
        if not slot.get("source_task"):
            return "locked"
        count = len(list((state["labels"] / "submitted").glob("*.rttm")))
        if count < int(slot["selected_after_submissions"]) + 5:
            return "locked"
    return "draft" if draft.exists() else "todo"


def validate_rttm(text, task, seconds, no_speech=False):
    if not isinstance(text, str):
        raise ValueError("RTTM must be text")
    rows = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        fields = line.split()
        if fields[0] == "SPEAKER" and len(fields) == 10:
            if fields[1] != task or fields[2] != "1" or not SPEAKER.fullmatch(fields[7]):
                raise ValueError(f"Invalid RTTM line {number}")
            kind = "speaker"
        elif fields[0] == "NOSCORE" and len(fields) == 5:
            if fields[1] != task or fields[2] != "1":
                raise ValueError(f"Invalid RTTM line {number}")
            kind = "noscore"
        else:
            raise ValueError(f"Invalid RTTM line {number}")
        try:
            start = float(fields[3]); duration = float(fields[4])
        except ValueError:
            raise ValueError(f"Invalid RTTM time on line {number}")
        if not math.isfinite(start) or not math.isfinite(duration):
            raise ValueError(f"Invalid RTTM time on line {number}")
        if start < 0 or duration <= 0:
            raise ValueError(f"Invalid RTTM time on line {number}")
        if start + duration > float(seconds) + END_TOLERANCE_S:
            raise ValueError(f"RTTM line {number} exceeds task duration")
        speaker = fields[7] if kind == "speaker" else None
        rows.append({"kind": kind, "start": start, "end": start + duration,
                     "speaker": speaker})
    by_speaker = {}
    for row in rows:
        if row["kind"] == "speaker":
            by_speaker.setdefault(row["speaker"], []).append((row["start"], row["end"]))
    for speaker, intervals in by_speaker.items():
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"Overlapping rows for {speaker} cannot be normalized")

    if not rows and not no_speech:
        raise ValueError("A task with no RTTM rows must be marked no_speech")
    return rows


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _duration(intervals):
    return sum(end - start for start, end in _union(intervals))


def _scorable_window(rows, start, end):
    speaker_parts = []
    noscore_parts = []
    for row in rows:
        if row["end"] <= start or row["start"] >= end:
            continue
        part = (max(start, row["start"]), min(end, row["end"]))
        if row["kind"] == "speaker":
            speaker_parts.append(part)
        else:
            noscore_parts.append(part)
    speakers = _union(speaker_parts)
    noscore = _union(noscore_parts)
    overlap = []
    for left, right in speakers:
        for nleft, nright in noscore:
            if nright > left and nleft < right:
                overlap.append((max(left, nleft), min(right, nright)))
    return max(0.0, _duration(speakers) - _duration(overlap))


def _append_ledger(labels, event):
    raw = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with (Path(labels) / "ledger.jsonl").open("a") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _audio_path(state, info):
    path = Path(info.get("audio", ""))
    root = state["packet"] / "audio" if info["kind"] == "original" else state["labels"] / "repeats"
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError:
        raise KeyError("Task audio unavailable")
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise KeyError("Task audio unavailable")
    if not _regular(path) or not _inside(path, root):
        raise KeyError("Task audio unavailable")
    if _sha256(path) != info.get("audio_sha256"):
        raise KeyError("Task audio unavailable")
    return path


def _cut_repeat(source, target, start):
    with wave.open(str(source), "rb") as incoming:
        rate = incoming.getframerate()
        first = int(start * rate)
        count = int(60 * rate)
        if first + count > incoming.getnframes():
            raise ValueError("Repeat window exceeds WAV")
        incoming.setpos(first)
        frames = incoming.readframes(count)
        with wave.open(str(target), "wb") as outgoing:
            outgoing.setparams(incoming.getparams())
            outgoing.writeframes(frames)
    os.chmod(target, 0o400)


def _select_repeats(state):
    labels = state["labels"]
    repeat_map = state["repeat_map"]
    selected = {slot.get("source_task") for slot in repeat_map["slots"]
                if slot.get("source_task")}
    submission_count = len(list((labels / "submitted").glob("*.rttm")))
    changed = False
    for plan in repeat_map["family_plan"]:
        source_task = plan["source_task"]
        rttm_path = labels / "submitted" / f"{source_task}.rttm"
        if not rttm_path.exists():
            break
        if source_task in selected:
            continue
        prov = json.loads((labels / "submitted" / f"{source_task}.prov.json").read_text())
        rows = validate_rttm(rttm_path.read_text(), source_task,
                             state["manifest"][source_task]["seconds"],
                             bool(prov.get("no_speech")))
        chosen = None
        for start in plan["candidate_starts"]:
            if _scorable_window(rows, start, start + 60) >= 30:
                chosen = start; break
        if chosen is None:
            continue
        slot = next((item for item in repeat_map["slots"] if not item.get("source_task")
                     and item.get("state") != "void"), None)
        if not slot:
            break
        source_info = dict(state["manifest"][source_task], kind="original")
        source = _audio_path(state, source_info)
        target = labels / "repeats" / f'{slot["task"]}.wav'
        _cut_repeat(source, target, chosen)
        slot.update({"family": plan["family"], "source_task": source_task,
                     "window_start_s": chosen, "audio": str(target),
                     "audio_sha256": _sha256(target),
                     "first_pass_scorable_s": _scorable_window(rows, chosen, chosen + 60),
                     "selected_after_submissions": submission_count, "state": "chosen"})
        selected.add(source_task); changed = True
    originals = set(state["manifest"])
    done = all((labels / "submitted" / f"{task}.rttm").exists() for task in originals)
    if done and len(selected) < 3:
        for slot in repeat_map["slots"]:
            if not slot.get("source_task"):
                slot["state"] = "void"; changed = True
        repeat_map["queue"] = [row for row in repeat_map["queue"]
                               if not (_slot({"repeat_map": repeat_map}, row["task"])
                                       and _slot({"repeat_map": repeat_map}, row["task"]).get("state") == "void")]
    if changed:
        _atomic_json(labels / "private/repeat-map.json", repeat_map, 0o400)
        _append_ledger(labels, {"type": "repeat_map_updated", "at": _utc(),
                                "sha256": _sha256(labels / "private/repeat-map.json")})


def _require_writable(state, task):
    task_info(state, task)
    if (state["labels"] / "SEALED").exists():
        raise LabelConflict("Labels are sealed")
    status = task_state(state, task)
    if status == "locked":
        raise LabelConflict("Task is locked")
    if status == "submitted":
        raise LabelConflict("Task is already submitted")


def _active_total(state, current_task, replacement):
    total = float(replacement)
    submitted = state["labels"] / "submitted"
    for path in submitted.glob("*.prov.json"):
        if path.name != f"{current_task}.prov.json":
            total += float(json.loads(path.read_text()).get("active_seconds", 0))
    for path in (state["labels"] / "drafts").glob("*.json"):
        task = path.name[:-5]
        if task == current_task or (submitted / f"{task}.prov.json").exists():
            continue
        total += float(json.loads(path.read_text()).get("active_seconds", 0))
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid active time")
    if total >= 600 * 60:
        raise LabelConflict("Active labelling reached 600 minutes; ask the operator")


def save_draft(packet, labels, task, body):
    with _locked(labels):
        state = load_state(packet, labels)
        _require_writable(state, task)
        if not isinstance(body, dict):
            raise ValueError("Draft must be an object")
        value = dict(body)
        _active_total(state, task, value.get("active_seconds", 0))
        value["task"] = task; value["saved_utc"] = _utc()
        _atomic_json(state["labels"] / "drafts" / f"{task}.json", value, 0o600)
        _append_ledger(labels, {"type": "draft_saved", "at": value["saved_utc"],
                                "task": task})
        return {"task": task, "state": "draft", "saved": True}


def save_submission(packet, labels, task, body):
    with _locked(labels):
        state = load_state(packet, labels)
        _require_writable(state, task)
        info = task_info(state, task)
        audio = _audio_path(state, info)
        _active_total(state, task, body.get("active_seconds", 0))
        no_speech = bool(body.get("no_speech", False))
        rttm = body.get("rttm", "")
        validate_rttm(rttm, task, info["seconds"], no_speech)
        attestations = body.get("attestations")
        if not isinstance(attestations, dict) or not attestations or not all(attestations.values()):
            raise ValueError("Blindness attestation is required")
        languages = body.get("languages", [])
        if not isinstance(languages, list) or any(not isinstance(x, str) for x in languages):
            raise ValueError("Languages must be a list")
        now = _utc()
        provenance = {
            "task": task, "annotator": "owner", "role": "sole",
            "started_utc": str(body.get("started_utc") or now), "submitted_utc": now,
            "active_seconds": float(body.get("active_seconds", 0)),
            "playback_seconds": float(body.get("playback_seconds", 0)),
            "edits": int(body.get("edits", 0)), "audio_sha256": _sha256(audio),
            "label_html_sha256": _sha256(Path(__file__).with_name("label.html")),
            "labelling_py_sha256": _sha256(__file__), "languages": languages,
            "strata_human": body.get("strata_human", {}),
            "attestations": attestations, "no_speech": no_speech,
            "session_hash_prefix": str(body.get("session_hash_prefix", ""))[:32],
        }
        if any(not math.isfinite(provenance[key]) or provenance[key] < 0
               for key in ("active_seconds", "playback_seconds")):
            raise ValueError("Invalid time accounting")
        submitted = state["labels"] / "submitted"
        rttm_path = submitted / f"{task}.rttm"
        prov_path = submitted / f"{task}.prov.json"
        raw_rttm = rttm.encode()
        raw_prov = json.dumps(provenance, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False).encode() + b"\n"
        made_rttm = made_prov = False
        try:
            _exclusive(rttm_path, raw_rttm); made_rttm = True
            _exclusive(prov_path, raw_prov); made_prov = True
        except FileExistsError:
            if made_rttm: rttm_path.unlink(missing_ok=True)
            if made_prov: prov_path.unlink(missing_ok=True)
            raise LabelConflict("Task is already submitted")
        except BaseException:
            if made_rttm: rttm_path.unlink(missing_ok=True)
            if made_prov: prov_path.unlink(missing_ok=True)
            raise
        _append_ledger(labels, {"type": "submitted", "at": now, "task": task,
                                "rttm_sha256": _sha256(rttm_path),
                                "prov_sha256": _sha256(prov_path)})
        if info["kind"] == "original":
            _select_repeats(state)
            from . import labelling_words
            labelling_words.on_original_submitted(packet, labels)
        return {"task": task, "state": "submitted", "saved": True}


def seal_inventory(packet, labels):
    state = load_state(packet, labels)
    expected = []
    for row in state["repeat_map"]["queue"]:
        try:
            info = task_info(state, row["task"])
        except KeyError:
            continue
        expected.append((row["task"], info))
    directory = state["labels"] / "submitted"
    wanted = {f"{task}.rttm" for task, _ in expected}
    wanted.update(f"{task}.prov.json" for task, _ in expected)
    actual = {path.name for path in directory.iterdir()}
    if actual != wanted:
        raise FileNotFoundError("Submitted inventory is incomplete or has extra entries")
    tasks = []
    for task, info in expected:
        rttm = directory / f"{task}.rttm"; prov = directory / f"{task}.prov.json"
        for path in (rttm, prov):
            if not _regular(path) or stat.S_IMODE(os.lstat(path).st_mode) != 0o400:
                raise ValueError(f"Submitted file is not immutable: {task}")
        provenance = json.loads(prov.read_text())
        rows = validate_rttm(rttm.read_text(), task, info["seconds"],
                             bool(provenance.get("no_speech")))
        if provenance.get("audio_sha256") != info.get("audio_sha256"):
            raise ValueError(f"Audio digest mismatch: {task}")
        if not provenance.get("attestations") or not all(provenance["attestations"].values()):
            raise ValueError(f"Attestation missing: {task}")
        speaker_s = sum(row["end"] - row["start"] for row in rows
                        if row["kind"] == "speaker")
        noscore_s = _duration((row["start"], row["end"]) for row in rows
                              if row["kind"] == "noscore")
        tasks.append({"task": task, "family": info["family"], "kind": info["kind"],
                      "rttm_sha256": _sha256(rttm), "prov_sha256": _sha256(prov),
                      "audio_sha256": info["audio_sha256"], "speaker_s": speaker_s,
                      "noscore_s": noscore_s})
    return {"expected_tasks": state["init"]["expected_tasks"], "tasks": tasks,
            "bundle_members": len(tasks) * 2 + 1,
            "repeat_map_sha256": _sha256(state["labels"] / "private/repeat-map.json")}


def trash_after_ack(run, labels, trash_root=None):
    run = Path(run); labels = Path(labels)
    seal = run / "labels-seal.json"; ack_path = run / "custody-ack.json"
    labels_ack = run / "labels-ack.json"
    if not all(_regular(path) for path in (seal, ack_path, labels_ack)):
        raise ValueError("LABELS_ACKED evidence is incomplete")
    ack_raw = ack_path.read_bytes(); ack = json.loads(ack_raw)
    transition = json.loads(labels_ack.read_text())
    if transition.get("custody_ack_sha256") != hashlib.sha256(ack_raw).hexdigest():
        raise ValueError("Custody acknowledgement digest mismatch")
    if transition.get("state") != "LABELS_ACKED":
        raise ValueError("Run has not reached LABELS_ACKED")
    required = {"seal_sha256", "files", "ok", "acked_at", "actor",
                "bundle_sha256", "reference_index_sha256"}
    if set(ack) != required or ack["actor"] != "custodian" or ack["ok"] is not True:
        raise ValueError("Custody acknowledgement is invalid")
    if ack["seal_sha256"] != _sha256(seal):
        raise ValueError("Custody acknowledgement names another seal")
    if not (labels / "SEALED").exists():
        raise ValueError("Labels are not sealed")
    trash_root = Path(trash_root) if trash_root else Path.home() / ".Trash"
    trash_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    operation = hashlib.sha256(ack_raw).hexdigest()[:16]
    destination = trash_root / f"plaud-diar-20261006-labels-{operation}"
    if destination.exists():
        raise FileExistsError("Trash destination already exists")
    destination.mkdir(mode=0o700)
    moved = []
    try:
        for name in ("submitted", "drafts"):
            source = labels / name; target = destination / name
            if source.exists():
                os.rename(source, target); moved.append(str(target))
    except BaseException:
        for target in reversed(moved):
            os.rename(target, labels / Path(target).name)
        destination.rmdir(); raise
    _append_ledger(labels, {"type": "labels_trashed_after_ack", "at": _utc(),
                            "custody_ack_sha256": hashlib.sha256(ack_raw).hexdigest(),
                            "trash_path": str(destination)})
    return moved


def _roots(app):
    config = app.c.get("labelling", {})
    labels = Path(app.c.get("label_root") or config.get("labels") or DEFAULT_LABELS)
    packet_value = app.c.get("label_packet") or config.get("packet")
    if packet_value:
        packet = Path(packet_value)
    else:
        init = json.loads((labels / "init.json").read_text())
        packet = Path(init["packet"])
    return packet, labels


def _peaks(path, buckets=256):
    with wave.open(str(path), "rb") as stream:
        width = stream.getsampwidth(); channels = stream.getnchannels()
        total = stream.getnframes(); step = max(1, int(math.ceil(total / buckets)))
        result = []
        while len(result) < buckets:
            raw = stream.readframes(step)
            if not raw:
                break
            values = []
            frame_width = width * channels
            for offset in range(0, len(raw) - frame_width + 1, frame_width):
                sample = int.from_bytes(raw[offset:offset + width], "little", signed=True)
                values.append(abs(sample))
            maximum = (1 << (width * 8 - 1)) - 1
            result.append(round((max(values) if values else 0) / maximum, 5))
        return result


def _task_rows(state):
    result = []
    for row in state["repeat_map"]["queue"]:
        try:
            info = task_info(state, row["task"])
        except KeyError:
            continue
        result.append({"task": row["task"], "seconds": info["seconds"],
                       "state": task_state(state, row["task"])})
    return result


def _html(handler):
    raw = Path(__file__).with_name("label.html").read_bytes()
    handler.headers_out(200, "text/html; charset=utf-8", len(raw))
    if handler.command != "HEAD":
        handler.wfile.write(raw)


def _error(handler, error):
    if isinstance(error, LabelConflict): status = 409
    elif isinstance(error, KeyError): status = 404
    elif isinstance(error, (ValueError, FileNotFoundError)): status = 400
    else: status = 500
    message = str(error) if status != 500 else "Labelling service could not complete request"
    handler.respond({"error": message}, status)


def handle(handler, app, route, body=None):
    """Dispatch only routes already admitted by web.py's existing security checks."""
    try:
        packet, labels = _roots(app)
        if route.startswith("/api/label/words"):
            from . import labelling_words
            labelling_words.handle(handler, app, route, body); return
        state = load_state(packet, labels)
        if handler.command in ("GET", "HEAD"):
            if route == "/label":
                _html(handler); return
            if route.startswith("/label/"):
                task = route[len("/label/"):]
                if not task or "/" in task:
                    raise KeyError("Unknown task")
                if task_state(state, task) == "locked":
                    raise LabelConflict("Task is locked")
                _html(handler); return
            if route == "/api/label/tasks":
                handler.respond(_task_rows(state)); return
            prefix = "/api/label/audio/"
            if route.startswith(prefix):
                task = route[len(prefix):]
                if not task or "/" in task:
                    raise KeyError("Unknown task")
                if task_state(state, task) == "locked":
                    raise LabelConflict("Task is locked")
                handler.audio(_audio_path(state, task_info(state, task))); return
            prefix = "/api/label/tasks/"
            if route.startswith(prefix):
                task = route[len(prefix):]
                if not task or "/" in task:
                    raise KeyError("Unknown task")
                status = task_state(state, task)
                if status == "locked":
                    raise LabelConflict("Task is locked")
                info = task_info(state, task)
                draft_path = labels / "drafts" / f"{task}.json"
                draft = json.loads(draft_path.read_text()) if _regular(draft_path) else None
                handler.respond({"task": task, "seconds": info["seconds"],
                                 "state": status, "draft": draft,
                                 "peaks": _peaks(_audio_path(state, info))})
                return
            raise KeyError("Not found")
        if handler.command == "POST":
            prefix = "/api/label/tasks/"
            if not route.startswith(prefix):
                raise KeyError("Not found")
            tail = route[len(prefix):]
            if tail.endswith("/draft"):
                task = tail[:-len("/draft")]; action = "draft"
            elif tail.endswith("/submit"):
                task = tail[:-len("/submit")]; action = "submit"
            else:
                raise KeyError("Not found")
            if not task or "/" in task:
                raise KeyError("Unknown task")
            result = (save_draft(packet, labels, task, body) if action == "draft"
                      else save_submission(packet, labels, task, body))
            handler.respond(result); return
        raise KeyError("Not found")
    except (LabelConflict, KeyError, ValueError, FileNotFoundError) as error:
        _error(handler, error)
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as error:
        _error(handler, error)


def probe(config_path=None, base_url=None, key_path=None):
    if config_path:
        config = json.loads(Path(config_path).read_text())
        base_url = base_url or config["base_url"]
        key_path = key_path or Path(config["root"]) / "launcher.key"
    if not base_url or not key_path:
        raise ValueError("Probe needs --config or both --base-url and --key")
    parsed = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    prefix = parsed.path.rstrip("/")
    origin = f"{parsed.scheme}://{parsed.netloc}"

    def request(method, path, body=None, headers=None):
        raw = json.dumps(body).encode() if body is not None else None
        values = {"Host": parsed.netloc, **(headers or {})}
        if body is not None:
            values["Content-Type"] = "application/json"
        connection.request(method, prefix + path, raw, values)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    status, _, raw = request("POST", "/bootstrap", {},
                             {"X-Local-Key": Path(key_path).read_text().strip()})
    if status != 200:
        raise RuntimeError(f"bootstrap returned {status}")
    ticket_url = json.loads(raw)["url"]
    ticket = urllib.parse.urlsplit(ticket_url).query
    status, headers, _ = request("GET", "/open?" + ticket)
    if status != 303:
        raise RuntimeError(f"ticket exchange returned {status}")
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    auth = {"Cookie": cookie}
    checks = []
    for path in ("/api/status", "/", "/label", "/api/label/tasks"):
        status, response_headers, raw = request("GET", path, headers=auth)
        if status != 200:
            raise RuntimeError(f"{path} returned {status}")
        if path == "/api/status" and json.loads(raw).get("service_error") is not None:
            raise RuntimeError("Service reports an error")
        if path == "/":
            golden = Path(__file__).with_name("app.html").read_text().replace("__APP_PREFIX__", json.dumps(prefix)).encode()
            if raw != golden:
                raise RuntimeError("Main page differs from golden HTML")
        if path == "/label" and hashlib.sha256(raw).hexdigest() != _sha256(Path(__file__).with_name("label.html")):
            raise RuntimeError("Label page differs from golden HTML")
        checks.append({"path": path, "status": status,
                       "sha256": hashlib.sha256(raw).hexdigest()})
        if path == "/api/label/tasks":
            tasks = json.loads(raw)
    if len(tasks) != 28:
        raise RuntimeError(f"Expected 28 label tasks, got {len(tasks)}")
    task = next((row for row in tasks if row["state"] != "locked"), None)
    if not task:
        raise RuntimeError("No readable label task")
    status, headers, raw = request("GET", f'/api/label/audio/{task["task"]}',
                                   headers={**auth, "Range": "bytes=0-1023"})
    if status != 206 or len(raw) != 1024:
        raise RuntimeError("Range audio probe failed")
    checks.append({"path": "/api/label/audio/<task>", "status": status,
                   "bytes": len(raw)})
    status, _, _ = request("GET", "/api/label/tasks")
    if status != 403:
        raise RuntimeError("Unsigned task API did not return 403")
    connection.close()
    return {"ok": True, "tasks": len(tasks), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init")
    init_parser.add_argument("--packet", required=True)
    init_parser.add_argument("--labels", required=True)
    init_parser.add_argument("--retention-frozen-sha256", required=True)
    trash = commands.add_parser("trash-after-ack")
    trash.add_argument("--run", required=True); trash.add_argument("--labels", required=True)
    trash.add_argument("--trash-root")
    check = commands.add_parser("probe")
    check.add_argument("--config"); check.add_argument("--base-url"); check.add_argument("--key")
    args = parser.parse_args(argv)
    if args.command == "init":
        result = initialize(args.packet, args.labels, args.retention_frozen_sha256)
    elif args.command == "trash-after-ack":
        result = {"moved": trash_after_ack(args.run, args.labels, args.trash_root)}
    else:
        result = probe(args.config, args.base_url, args.key)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

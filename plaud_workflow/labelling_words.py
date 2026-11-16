"""Blind transcript-reference tasks for the Plaud labelling service."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import secrets
import stat
import subprocess
import unicodedata
from pathlib import Path

OPENSSL = "/opt/homebrew/opt/openssl@3/bin/openssl"
OPENSSL_REALPATH = "/opt/homebrew/Cellar/openssl@3/3.6.4/bin/openssl"
OPENSSL_SHA256 = "67a83dd6d6d747d50c5d296dffb23e32bae9a2c588c93ae2d77e4c607b455c72"
RECIPIENT_CN = "CN=plaud-word-ref-diar-20261006"
OID_NAMES = ["id-smime-ct-authEnvelopedData", "commonName", "rsaesOaep",
             "sha256", "mgf1", "sha256", "pkcs7-data", "aes-256-gcm"]
SEED = 20261006
STRATA = ("English", "Mandarin", "Cantonese/code-switch")
CAP_SECONDS = 180 * 60
MAX_PLAINTEXT = 262144
MAX_CIPHERTEXT = 300000
TASK_RE = re.compile(r"^w[0-9a-f]{12}$")

LANGUAGE_ALIASES = {
    "en": "English", "eng": "English", "english": "English",
    "zh": "Mandarin", "cmn": "Mandarin", "mandarin": "Mandarin",
    "普通话": "Mandarin", "國語": "Mandarin", "国语": "Mandarin",
    "普通話": "Mandarin", "yue": "Cantonese", "cantonese": "Cantonese",
    "粤语": "Cantonese", "粵語": "Cantonese", "广东话": "Cantonese",
    "廣東話": "Cantonese",
}


class WordConflict(Exception): pass
class PayloadTooLarge(Exception): pass


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path, value, mode=0o400):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    temp = path.with_name("." + path.name + ".tmp")
    temp.unlink(missing_ok=True)
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temp, mode); os.replace(temp, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _exclusive(path, raw, mode):
    path = Path(path)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(path, mode)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True); raise


def _regular(path):
    try: value = os.lstat(path)
    except FileNotFoundError: return False
    return stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode)


def _append_ledger(labels, event):
    path = Path(labels) / "words/ledger.jsonl"
    raw = json.dumps(event, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def normalize_languages(values):
    if not isinstance(values, list): return []
    result = []
    for value in values:
        item = (LANGUAGE_ALIASES.get(value.strip().casefold(), "Other")
                if isinstance(value, str) else "Other")
        if item not in result: result.append(item)
    return result


def stratum_for(languages, cantonese_code_switch):
    if "Cantonese" in languages or cantonese_code_switch:
        return "Cantonese/code-switch"
    if "Mandarin" in languages: return "Mandarin"
    if "English" in languages: return "English"
    return None


def provenance_inputs(provenance):
    provenance = provenance if isinstance(provenance, dict) else {}
    languages = normalize_languages(provenance.get("languages"))
    strata = provenance.get("strata_human")
    if not isinstance(strata, dict): strata = provenance
    code_switch = (bool(strata.get("cantonese")) or bool(strata.get("code_switch"))
                   or bool(strata.get("cantonese_code_switch"))
                   or "Cantonese" in languages)
    return {"languages": languages, "cantonese_code_switch": code_switch,
            "stratum": stratum_for(languages, code_switch)}


def encrypt_argv(recipient):
    return [OPENSSL, "cms", "-encrypt", "-binary", "-aes-256-gcm",
            "-recip", str(recipient), "-keyopt", "rsa_padding_mode:oaep",
            "-keyopt", "rsa_oaep_md:sha256", "-keyopt", "rsa_mgf1_md:sha256",
            "-outform", "DER"]


def v1_argv(path):
    return [OPENSSL, "asn1parse", "-inform", "DER", "-in", str(path)]


def v2_argv(path):
    return [OPENSSL, "cms", "-cmsout", "-print", "-noout", "-inform", "DER",
            "-in", str(path)]


def _run(argv, **kwargs):
    try:
        return subprocess.run(argv, capture_output=True, timeout=10,
                              env={"LC_ALL": "C"}, **kwargs)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Pinned OpenSSL operation failed") from error


def _check_openssl(record):
    if record.get("path") != OPENSSL:
        raise RuntimeError("Pinned OpenSSL path changed")
    try: realpath = str(Path(OPENSSL).resolve(strict=True))
    except FileNotFoundError as error:
        raise RuntimeError("Pinned OpenSSL is unavailable") from error
    if realpath != record.get("realpath") or _sha256(OPENSSL) != record.get("sha256"):
        raise RuntimeError("Pinned OpenSSL identity changed")


def encrypt_bytes(plaintext, recipient, openssl_record):
    if len(plaintext) > MAX_PLAINTEXT: raise PayloadTooLarge("Reference is too large")
    _check_openssl(openssl_record)
    result = _run(encrypt_argv(recipient), input=plaintext)
    if (result.returncode != 0 or result.stderr != b"" or not result.stdout
            or len(result.stdout) > MAX_CIPHERTEXT or result.stdout[:1] != b"\x30"):
        raise RuntimeError("Reference encryption failed")
    return result.stdout


def _accepted(result):
    if result.returncode != 0 or result.stderr != b"":
        raise ValueError("CMS validation failed")
    return result.stdout.decode("utf-8", "strict")


def validate_ciphertext(path, openssl_record):
    path = Path(path)
    if not _regular(path) or not 0 < path.stat().st_size <= MAX_CIPHERTEXT:
        raise ValueError("CMS file is invalid")
    if path.read_bytes()[:1] != b"\x30": raise ValueError("CMS is not DER")
    _check_openssl(openssl_record)
    parsed = _accepted(_run(v1_argv(path)))
    objects = [line.rsplit(":", 1)[-1].strip() for line in parsed.splitlines()
               if "OBJECT" in line]
    if objects != OID_NAMES: raise ValueError("CMS algorithms do not match the pin")
    printed = _accepted(_run(v2_argv(path)))
    issuers = [line.strip().split("issuer:", 1)[1].strip()
               for line in printed.splitlines() if "issuer:" in line]
    serials = [line.strip().split("serialNumber:", 1)[1].strip()
               for line in printed.splitlines() if "serialNumber:" in line]
    if issuers != [RECIPIENT_CN] or serials != ["1"]:
        raise ValueError("CMS recipient does not match")
    return True


def _word_paths(labels):
    root = Path(labels) / "words"
    return root, root / "private/word-map.json", root / "init.json"


def _build_plan(packet, labels):
    from . import labelling
    state = labelling.load_state(packet, labels)
    originals = [row["task"] for row in state["repeat_map"]["queue"]
                 if row["kind"] == "original"]
    by_family = {}
    for task in originals:
        info = state["manifest"].get(task)
        if info is not None and info["family"] not in by_family:
            by_family[info["family"]] = info
    rng = random.Random(SEED); candidates = {}
    for family in sorted(by_family):
        info = by_family[family]
        starts = list(range(0, max(0, int(math.floor(info["seconds"])) - 59)))
        rng.shuffle(starts); candidates[family] = starts
    plan = [{"family": state["manifest"][task]["family"], "source_task": task,
             "candidate_starts": candidates[state["manifest"][task]["family"]]}
            for task in originals if state["manifest"][task]["family"] in candidates]
    return plan


def create_word_map(packet, labels):
    _root, map_path, _init_path = _word_paths(labels)
    plan = _build_plan(packet, labels)
    value = {"schema": "plaud-word-map/1", "seed": SEED, "plan": plan,
             "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
             "walked": [], "slots": [], "shortfall": []}
    _write_json(map_path, value, 0o400)
    return value


def _load(labels):
    root, map_path, init_path = _word_paths(labels)
    if not _regular(init_path) or not _regular(map_path):
        raise KeyError("Transcript tasks are unavailable")
    init = json.loads(init_path.read_text())
    word_map = json.loads(map_path.read_text())
    if init.get("schema") != "plaud-word-tasks/1" or word_map.get("schema") != "plaud-word-map/1":
        raise ValueError("Transcript task metadata is invalid")
    if init.get("plan_sha256") != word_map.get("plan_sha256"):
        raise ValueError("Transcript task plan changed")
    return root, init, word_map


def _slot(word_map, task, chosen_only=True):
    if not TASK_RE.fullmatch(str(task)): raise KeyError("Unknown transcript task")
    for slot in word_map["slots"]:
        if slot.get("task") == task and (not chosen_only or slot.get("state") == "chosen"):
            return slot
    raise KeyError("Unknown transcript task")


def _certificate_spki(recipient):
    first = _run([OPENSSL, "x509", "-in", str(recipient), "-noout", "-pubkey"])
    if first.returncode != 0 or first.stderr != b"" or not first.stdout:
        raise ValueError("Recipient certificate is invalid")
    second = _run([OPENSSL, "pkey", "-pubin", "-outform", "DER"], input=first.stdout)
    if second.returncode != 0 or second.stderr != b"" or not second.stdout:
        raise ValueError("Recipient public key is invalid")
    return hashlib.sha256(second.stdout).hexdigest()


def initialize(packet, labels, recipient_pem, expect_cert_sha256,
               prov_contract_sha256, wordref_root=None):
    packet = Path(packet).resolve(); labels = Path(labels).resolve()
    words = labels / "words"
    if words.exists(): raise FileExistsError("Transcript task root already exists")
    if not re.fullmatch(r"[0-9a-f]{64}", expect_cert_sha256 or ""):
        raise ValueError("Recipient certificate digest is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", prov_contract_sha256 or ""):
        raise ValueError("Provenance contract digest is invalid")
    recipient_pem = bytes(recipient_pem)
    if hashlib.sha256(recipient_pem).hexdigest() != expect_cert_sha256:
        raise ValueError("Recipient certificate digest mismatch")
    openssl_record = {"path": OPENSSL, "realpath": OPENSSL_REALPATH,
                      "sha256": OPENSSL_SHA256}
    _check_openssl(openssl_record)
    version = _run([OPENSSL, "version"])
    if version.returncode != 0 or version.stderr != b"" or not version.stdout.startswith(b"OpenSSL 3.6.4"):
        raise RuntimeError("Pinned OpenSSL version changed")
    words.mkdir(mode=0o700); wordref = None; wordref_created = False
    try:
        for name in ("private", "drafts"): (words / name).mkdir(mode=0o700)
        _exclusive(words / "recipient.pem", recipient_pem, 0o400)
        spki = _certificate_spki(words / "recipient.pem")
        wordref = (Path(wordref_root).resolve() if wordref_root else
                   labels.parent / "diar-20261006-wordref")
        wordref.mkdir(mode=0o700); wordref_created = True; os.chmod(wordref, 0o700)
        (words / "ledger.jsonl").touch(mode=0o600)
        word_map = create_word_map(packet, labels)
        marker = secrets.token_hex(16)
        init = {"schema": "plaud-word-tasks/1", "created_utc": _utc(),
                "key_id": "wordref-diar-20261006",
                "recipient_cert_sha256": expect_cert_sha256,
                "recipient_spki_sha256": spki,
                "prov_contract_sha256": prov_contract_sha256,
                "plan_sha256": word_map["plan_sha256"], "marker": marker,
                "openssl": dict(openssl_record, version=version.stdout.decode().strip()),
                "wordref_root": str(wordref), "cap_minutes": 180}
        _write_json(words / "init.json", init, 0o600)
        synthetic = _canonical({"self_test": True, "marker": marker})
        cipher = encrypt_bytes(synthetic, words / "recipient.pem", init["openssl"])
        witness = words / ".selftest.cms"
        _exclusive(witness, cipher, 0o600)
        try: validate_ciphertext(witness, init["openssl"])
        finally: witness.unlink(missing_ok=True)
        select(packet, labels)
        return {"plan_sha256": init["plan_sha256"],
                "recipient_spki_sha256": spki, "wordref_root": str(wordref)}
    except BaseException:
        # Init is all-or-nothing and is only allowed when no word work exists.
        import shutil
        shutil.rmtree(words, ignore_errors=True)
        if wordref_created:
            try: wordref.rmdir()
            except OSError: pass
        raise


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _repeat_window(state, source_task):
    for slot in state["repeat_map"]["slots"]:
        if slot.get("source_task") == source_task:
            start = float(slot["window_start_s"])
            return start, start + 60.0
    return None


def _word_task_id(plan_sha256, stratum, number):
    raw = f"{plan_sha256}:{stratum}:{number}".encode("utf-8")
    return "w" + hashlib.sha256(raw).hexdigest()[:12]


def select(packet, labels):
    """Walk submitted originals in frozen queue order and fill 5 slots/stratum."""
    from . import labelling
    labels = Path(labels); state = labelling.load_state(packet, labels)
    _root, _init, word_map = _load(labels)
    walked = {row["family"] for row in word_map["walked"]}
    changed = False
    for plan in word_map["plan"]:
        if plan["family"] in walked: continue
        task = plan["source_task"]
        rttm_path = labels / "submitted" / f"{task}.rttm"
        prov_path = labels / "submitted" / f"{task}.prov.json"
        if not _regular(rttm_path) or not _regular(prov_path): break
        provenance = json.loads(prov_path.read_text())
        inputs = provenance_inputs(provenance); stratum = inputs["stratum"]
        rows = labelling.validate_rttm(
            rttm_path.read_text(), task, state["manifest"][task]["seconds"],
            bool(provenance.get("no_speech")))
        repeat = _repeat_window(state, task); chosen = None; scorable = 0.0
        for start in plan["candidate_starts"]:
            window = (float(start), float(start) + 60.0)
            if repeat and _overlaps(window, repeat): continue
            score = labelling._scorable_window(rows, *window)
            if score >= 30.0:
                chosen = float(start); scorable = score; break
        record = {"family": plan["family"], "source_task": task,
                  "prov_sha256": _sha256(prov_path),
                  "languages": inputs["languages"],
                  "cantonese_code_switch": inputs["cantonese_code_switch"],
                  "stratum": stratum, "eligible": bool(stratum and chosen is not None)}
        if stratum and chosen is not None:
            number = 1 + sum(slot.get("stratum") == stratum
                             and slot.get("state") == "chosen"
                             for slot in word_map["slots"])
            if number <= 5:
                word_map["slots"].append({
                    "slot": number,
                    "task": _word_task_id(word_map["plan_sha256"], stratum, number),
                    "family": plan["family"], "source_task": task, "stratum": stratum,
                    "stratum_inputs": {"prov_sha256": record["prov_sha256"],
                                        "languages": inputs["languages"],
                                        "cantonese_code_switch": inputs["cantonese_code_switch"]},
                    "window_source_start_s": chosen, "scorable_s": scorable,
                    "state": "chosen", "active_seconds": 0.0})
        word_map["walked"].append(record); walked.add(plan["family"]); changed = True
    done = all(_regular(labels / "submitted" / f'{row["source_task"]}.rttm')
               for row in word_map["plan"])
    if done and not word_map["shortfall"]:
        for stratum in STRATA:
            count = sum(slot.get("stratum") == stratum and slot.get("state") == "chosen"
                        for slot in word_map["slots"])
            if count < 5:
                shortfall = {"stratum": stratum, "eligible": count, "needed": 5}
                word_map["shortfall"].append(shortfall)
                for number in range(count + 1, 6):
                    word_map["slots"].append({
                        "slot": number,
                        "task": _word_task_id(word_map["plan_sha256"], stratum, number),
                        "stratum": stratum, "state": "void", "shortfall": shortfall})
                changed = True
    if changed:
        _root, map_path, _init_path = _word_paths(labels)
        _write_json(map_path, word_map, 0o400)
        _append_ledger(labels, {"type": "word_map_updated", "at": _utc(),
                                "sha256": _sha256(map_path)})
    return word_map


def on_original_submitted(packet, labels):
    if (Path(labels) / "words/init.json").exists(): select(packet, labels)


def _detail(packet, labels, task):
    from . import labelling
    labels = Path(labels); state = labelling.load_state(packet, labels)
    _root, _init, word_map = _load(labels); slot = _slot(word_map, task)
    source = slot["source_task"]; start = float(slot["window_source_start_s"])
    provenance = json.loads((labels / "submitted" / f"{source}.prov.json").read_text())
    rows = labelling.validate_rttm(
        (labels / "submitted" / f"{source}.rttm").read_text(), source,
        state["manifest"][source]["seconds"], bool(provenance.get("no_speech")))
    segments = []; noscore = []
    ordered = sorted(rows, key=lambda row: (row["start"], row["end"], row.get("speaker") or ""))
    for row in ordered:
        left = max(start, row["start"]); right = min(start + 60.0, row["end"])
        if right <= left: continue
        if row["kind"] == "speaker":
            segments.append({"seg": f"s{len(segments) + 1}", "speaker": row["speaker"],
                             "start": left - start, "end": right - start})
        else: noscore.append([left - start, right - start])
    draft_path = labels / "words/drafts" / f"{task}.json"
    draft = json.loads(draft_path.read_text()) if _regular(draft_path) else None
    return {"task": task, "seconds": 60, "audio_task": source,
            "window_start_s": start, "segments": segments,
            "noscore": noscore, "draft": draft}


def get_task(packet, labels, task):
    return _detail(packet, labels, task)


def list_tasks(packet, labels):
    _root, init, word_map = _load(labels); wordref = Path(init["wordref_root"])
    result = []
    for slot in word_map["slots"]:
        if slot.get("state") != "chosen": continue
        task = slot["task"]
        state = ("submitted" if _regular(wordref / f"{task}.cms") else
                 "draft" if _regular(Path(labels) / "words/drafts" / f"{task}.json")
                 else "todo")
        result.append({"task": task, "seconds": 60, "state": state})
    return result


def _require_writable(labels, task, init, word_map):
    slot = _slot(word_map, task)
    if (Path(labels) / "SEALED").exists(): raise WordConflict("Labels are sealed")
    if _regular(Path(init["wordref_root"]) / f"{task}.cms"):
        raise WordConflict("Transcript task is already submitted")
    return slot


def _active_total(labels, word_map, current_task, replacement):
    total = float(replacement)
    if not math.isfinite(total) or total < 0: raise ValueError("Invalid active time")
    for slot in word_map["slots"]:
        if slot.get("task") != current_task:
            total += float(slot.get("active_seconds", 0))
    for path in (Path(labels) / "words/drafts").glob("*.json"):
        task = path.stem
        if task == current_task: continue
        if any(slot.get("task") == task and slot.get("active_seconds", 0) > 0
               for slot in word_map["slots"]):
            continue
        total += float(json.loads(path.read_text()).get("active_seconds", 0))
    if not math.isfinite(total) or total < 0: raise ValueError("Invalid active time")
    if total >= CAP_SECONDS:
        raise WordConflict("Transcript time reached 180 minutes; ask the operator")
    return total


def _normalize_body(detail, body):
    if not isinstance(body, dict):
        raise ValueError("Transcript submission must be an object")
    if body.get("attestation_audio_only") is not True:
        raise ValueError("Audio-only attestation is required")
    active = body.get("active_seconds")
    if isinstance(active, bool): raise ValueError("Invalid active time")
    try: active = float(active)
    except (TypeError, ValueError): raise ValueError("Invalid active time")
    if not math.isfinite(active) or active < 0:
        raise ValueError("Invalid active time")
    incoming = body.get("segments")
    if not isinstance(incoming, list):
        raise ValueError("Transcript segments are required")
    by_id = {}
    for item in incoming:
        if not isinstance(item, dict) or not isinstance(item.get("seg"), str):
            raise ValueError("Transcript segment is invalid")
        if item["seg"] in by_id:
            raise ValueError("Transcript segment is duplicated")
        unclear = item.get("unclear") is True
        text = item.get("text", "")
        if not isinstance(text, str): raise ValueError("Transcript text is invalid")
        text = unicodedata.normalize("NFC", text)
        if any(unicodedata.category(char) == "Cc" and char != "\n" for char in text):
            raise ValueError("Transcript text contains a control character")
        if len(text) > 2000 or (not unclear and len(text) < 1):
            raise ValueError("Transcript segment must contain words or be unclear")
        by_id[item["seg"]] = {"text": text, "unclear": unclear}
    expected = [row["seg"] for row in detail["segments"]]
    if set(by_id) != set(expected):
        raise ValueError("Transcript segments do not match")
    segments = [dict(row, text=by_id[row["seg"]]["text"],
                     unclear=by_id[row["seg"]]["unclear"])
                for row in detail["segments"]]
    return {"segments": segments, "active_seconds": active,
            "attestation_audio_only": True,
            "page_sha256": str(body.get("page_sha256", ""))[:64]}


def save_draft(packet, labels, task, body):
    from . import labelling
    labels = Path(labels)
    with labelling._locked(labels):
        _root, init, word_map = _load(labels)
        _require_writable(labels, task, init, word_map)
        detail = _detail(packet, labels, task); value = _normalize_body(detail, body)
        _active_total(labels, word_map, task, value["active_seconds"])
        draft = dict(value, marker=init["marker"], task=task, saved_utc=_utc())
        _write_json(labels / "words/drafts" / f"{task}.json", draft, 0o600)
        _append_ledger(labels, {"type": "word_draft_saved", "task": task,
                                "at": draft["saved_utc"]})
        return {"task": task, "state": "draft", "saved": True}


def _reference_document(labels, init, slot, detail, value):
    module = Path(__file__)
    return {"schema": "plaud-word-ref/1", "marker": init["marker"],
            "task": slot["task"], "slot": slot["slot"], "family": slot["family"],
            "stratum": slot["stratum"],
            "window_source_start_s": slot["window_source_start_s"],
            "segments": value["segments"], "noscore": detail["noscore"],
            "annotator": "owner", "submitted_utc": _utc(),
            "active_seconds": value["active_seconds"],
            "attestation_audio_only": True,
            "page_sha256": _sha256(module.with_name("label.html")),
            "module_sha256": _sha256(module)}


def submit(packet, labels, task, body):
    from . import labelling
    labels = Path(labels)
    with labelling._locked(labels):
        _root, init, word_map = _load(labels)
        slot = _require_writable(labels, task, init, word_map)
        detail = _detail(packet, labels, task); value = _normalize_body(detail, body)
        _active_total(labels, word_map, task, value["active_seconds"])
        document = _reference_document(labels, init, slot, detail, value)
        plaintext = _canonical(document)
        if len(plaintext) > MAX_PLAINTEXT:
            raise PayloadTooLarge("Reference is too large")
        cipher = encrypt_bytes(plaintext, labels / "words/recipient.pem", init["openssl"])
        wordref = Path(init["wordref_root"])
        if not wordref.is_dir() or wordref.is_symlink():
            raise RuntimeError("Reference store is unavailable")
        cipher_path = wordref / f"{task}.cms"
        try: _exclusive(cipher_path, cipher, 0o400)
        except FileExistsError as error:
            raise WordConflict("Transcript task is already submitted") from error
        (labels / "words/drafts" / f"{task}.json").unlink(missing_ok=True)
        slot["active_seconds"] = value["active_seconds"]
        _root, map_path, _init_path = _word_paths(labels)
        _write_json(map_path, word_map, 0o400)
        _append_ledger(labels, {"type": "word_map_updated", "at": _utc(),
                                "sha256": _sha256(map_path)})
        event = {"type": "word_ref_sealed", "task": task, "slot": slot["slot"],
                 "ciphertext_sha256": hashlib.sha256(cipher).hexdigest(),
                 "ciphertext_bytes": len(cipher),
                 "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(), "at": _utc()}
        _append_ledger(labels, event)
        return {"task": task, "state": "submitted", "saved": True}


def scan_plaintext(labels, extra_roots=()):
    labels = Path(labels); _root, init, _word_map = _load(labels)
    roots = [labels, Path(init["wordref_root"]), *(Path(p) for p in extra_roots)]
    marker = init["marker"].encode("ascii"); hits = []; scanned = 0
    init_path = labels / "words/init.json"
    for root in roots:
        if not root.exists(): continue
        for directory, names, files in os.walk(root, followlinks=False):
            names[:] = [name for name in names
                        if not (Path(directory) / name).is_symlink()]
            for name in files:
                path = Path(directory) / name
                if path == init_path or not _regular(path): continue
                if path.stat().st_size > 8 * 1024 * 1024: continue
                scanned += 1; raw = path.read_bytes()
                if marker in raw or (path.suffix == ".json"
                                     and b'"plaud-word-ref/1"' in raw):
                    hits.append(str(path))
    return {"roots": [str(path) for path in roots], "files_scanned": scanned,
            "hits": sorted(hits), "scanned_at": _utc()}


def _sealed_events(labels):
    result = {}
    for line in (Path(labels) / "words/ledger.jsonl").read_text().splitlines():
        value = json.loads(line)
        if value.get("type") == "word_ref_sealed": result[value["task"]] = value
    return result


def seal_inventory(packet, labels, extra_scan_roots=()):
    labels = Path(labels); _root, init, word_map = _load(labels)
    drafts = labels / "words/drafts"
    if any(drafts.iterdir()): raise ValueError("Transcript drafts remain")
    wordref = Path(init["wordref_root"]); events = _sealed_events(labels)
    chosen = [slot for slot in word_map["slots"] if slot.get("state") == "chosen"]
    wanted = {f'{slot["task"]}.cms' for slot in chosen}
    actual = {path.name for path in wordref.iterdir()}
    if actual != wanted: raise ValueError("Transcript reference inventory differs")
    rows = []
    for slot in word_map["slots"]:
        if slot.get("state") == "void":
            rows.append({key: slot[key] for key in ("slot", "task", "stratum", "state")})
            continue
        prov_path = labels / "submitted" / (slot["source_task"] + ".prov.json")
        current = provenance_inputs(json.loads(prov_path.read_text()))
        expected_inputs = slot["stratum_inputs"]
        observed_inputs = {"prov_sha256": _sha256(prov_path),
                           "languages": current["languages"],
                           "cantonese_code_switch": current["cantonese_code_switch"]}
        if observed_inputs != expected_inputs or current["stratum"] != slot["stratum"]:
            raise ValueError("Transcript stratum changed after selection")
        path = wordref / f'{slot["task"]}.cms'; event = events.get(slot["task"])
        if (not _regular(path) or stat.S_IMODE(os.lstat(path).st_mode) != 0o400
                or not event or _sha256(path) != event.get("ciphertext_sha256")
                or path.stat().st_size != event.get("ciphertext_bytes")):
            raise ValueError("Transcript reference does not match its ledger")
        validate_ciphertext(path, init["openssl"])
        row = {key: slot[key] for key in ("slot", "task", "family", "stratum",
                                          "stratum_inputs", "window_source_start_s",
                                          "scorable_s")}
        row.update(state="sealed", ciphertext_name=path.name,
                   ciphertext_sha256=event["ciphertext_sha256"],
                   ciphertext_bytes=event["ciphertext_bytes"],
                   plaintext_sha256=event["plaintext_sha256"])
        rows.append(row)
    scan = scan_plaintext(labels, extra_scan_roots)
    if scan["hits"]: raise ValueError("Transcript plaintext residue found")
    used = sum(float(slot.get("active_seconds", 0)) for slot in chosen)
    return {"key_id": init["key_id"],
            "recipient_cert_sha256": init["recipient_cert_sha256"],
            "recipient_spki_sha256": init["recipient_spki_sha256"],
            "plan_sha256": init["plan_sha256"],
            "word_map_sha256": _sha256(labels / "words/private/word-map.json"),
            "cap": {"minutes": 180, "used_s": used},
            "prov_contract_sha256": init["prov_contract_sha256"],
            "slots": rows, "shortfall": word_map["shortfall"], "scan": scan}


def handle(handler, app, route, body=None):
    """Dispatch routes after web.py's existing authentication checks."""
    from . import labelling
    try:
        packet, labels = labelling._roots(app)
        if not (Path(labels) / "words/init.json").exists():
            raise KeyError("Not found")
        if handler.command in ("GET", "HEAD"):
            if route == "/api/label/words":
                handler.respond(list_tasks(packet, labels)); return
            prefix = "/api/label/words/"
            if route.startswith(prefix):
                task = route[len(prefix):]
                if not task or "/" in task or task in (".", ".."):
                    raise KeyError("Unknown transcript task")
                handler.respond(get_task(packet, labels, task)); return
            raise KeyError("Not found")
        if handler.command == "POST":
            prefix = "/api/label/words/"
            if not route.startswith(prefix): raise KeyError("Not found")
            tail = route[len(prefix):]
            if tail.endswith("/draft"): task = tail[:-6]; action = "draft"
            elif tail.endswith("/submit"): task = tail[:-7]; action = "submit"
            else: raise KeyError("Not found")
            if not TASK_RE.fullmatch(task): raise KeyError("Unknown transcript task")
            result = (save_draft(packet, labels, task, body) if action == "draft"
                      else submit(packet, labels, task, body))
            handler.respond(result); return
        raise KeyError("Not found")
    except PayloadTooLarge:
        handler.respond({"error": "Transcript reference is too large"}, 413)
    except WordConflict as error:
        handler.respond({"error": str(error)}, 409)
    except KeyError:
        handler.respond({"error": "Not found"}, 404)
    except ValueError as error:
        handler.respond({"error": str(error)}, 400)
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception:
        handler.respond({"error": "Transcript service could not complete request"}, 500)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage blind Plaud transcript tasks")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--packet", required=True); init.add_argument("--labels", required=True)
    init.add_argument("--recipient", required=True)
    init.add_argument("--expect-cert-sha256", required=True)
    init.add_argument("--prov-contract-sha256", required=True)
    init.add_argument("--wordref-root")
    args = parser.parse_args(argv)
    if args.recipient != "-": raise ValueError("Recipient certificate must arrive on stdin")
    import sys
    recipient = sys.stdin.buffer.read(65537)
    if len(recipient) > 65536: raise ValueError("Recipient certificate is too large")
    result = initialize(args.packet, args.labels, recipient, args.expect_cert_sha256,
                        args.prov_contract_sha256, args.wordref_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

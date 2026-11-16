import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_wav(path, seconds=125):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(10)
        out.writeframes(b"\0\0" * (seconds * 10))


def make_real_shape_packet(packet, task_count=25):
    """Build the packet shape emitted by the diarization packet builder."""
    (packet / "audio").mkdir(parents=True)
    tasks = []
    for number in range(task_count):
        task = f"opaque{number:02d}"
        audio = packet / "audio" / f"{task}.wav"
        make_wav(audio)
        tasks.append({
            "audio": f"audio/{task}.wav",
            "audio_sha256": sha(audio),
            "excerpt_s": 125.0,
            "excerpt_start_s": float(number),
            "family": f"family{number:02d}",
            "long_session_region": bool(number % 2),
            "opaque_id": task,
        })
    manifest = {
        "contains_provider_output": False,
        "created_utc": "2026-10-27T00:00:00Z",
        "tasks": tasks,
    }
    (packet / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class LabellingFixture(unittest.TestCase):
    def setUp(self):
        from plaud_workflow import labelling
        self.labelling = labelling
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet = self.root / "packet"
        self.manifest = make_real_shape_packet(self.packet)
        self.labels = self.root / "labels"

    def tearDown(self):
        self.temp.cleanup()

    def init(self):
        self.labelling.initialize(self.packet, self.labels, "a" * 64)
        return json.loads((self.labels / "private/repeat-map.json").read_text())

    def submit(self, task, duration=60):
        return self.labelling.save_submission(self.packet, self.labels, task, {
            "rttm": f"SPEAKER {task} 1 0.000 {duration:.3f} <NA> <NA> spk1 <NA> <NA>\n",
            "languages": ["yue"], "strata_human": {"cantonese": True, "code_switch": False},
            "attestations": {"blind": True}, "no_speech": False, "active_seconds": 3,
            "playback_seconds": 2, "edits": 1, "session_hash_prefix": "0123456789ab"})


class RealManifestCli(LabellingFixture):
    def test_real_shape_init_creates_25_originals_and_3_locked_repeats(self):
        self.assertEqual(set(self.manifest),
                         {"contains_provider_output", "created_utc", "tasks"})
        self.assertIs(type(self.manifest["contains_provider_output"]), bool)
        self.assertIs(type(self.manifest["created_utc"]), str)
        self.assertEqual(len(self.manifest["tasks"]), 25)
        self.assertEqual(
            set(self.manifest["tasks"][0]),
            {"audio", "audio_sha256", "excerpt_s", "excerpt_start_s", "family",
             "long_session_region", "opaque_id"})
        field_types = {
            "audio": str, "audio_sha256": str, "excerpt_s": float,
            "excerpt_start_s": float, "family": str,
            "long_session_region": bool, "opaque_id": str,
        }
        for task in self.manifest["tasks"]:
            self.assertEqual({key: type(value) for key, value in task.items()}, field_types)
        completed = subprocess.run(
            [sys.executable, "-m", "plaud_workflow.labelling", "init",
             "--packet", str(self.packet), "--labels", str(self.labels),
             "--retention-frozen-sha256", "a" * 64],
            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["tasks"], 28)
        repeat_map = json.loads((self.labels / "private/repeat-map.json").read_text())
        self.assertEqual(len(repeat_map["queue"]), 28)
        self.assertEqual(len(repeat_map["slots"]), 3)
        self.assertEqual({slot["state"] for slot in repeat_map["slots"]}, {"locked"})


class F17StateMachine(LabellingFixture):
    def test_original_and_unlocked_repeat_share_submission_path(self):
        repeat_map = self.init()
        originals = [r["task"] for r in repeat_map["queue"] if r["kind"] == "original"]
        for task in originals[:6]:
            self.submit(task)
        state = self.labelling.load_state(self.packet, self.labels)
        repeat = next(s for s in state["repeat_map"]["slots"]
                      if s.get("source_task") and self.labelling.task_state(state, s["task"]) != "locked")
        self.labelling.save_draft(self.packet, self.labels, repeat["task"], {"rttm": ""})
        self.assertEqual(self.submit(repeat["task"])["state"], "submitted")
        path = self.labels / "submitted" / f'{repeat["task"]}.rttm'
        self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        rp = json.loads(path.with_suffix(".prov.json").read_text())
        op = json.loads((self.labels / "submitted" / f'{originals[0]}.prov.json').read_text())
        self.assertEqual(set(rp), set(op))
        self.assertEqual(rp["audio_sha256"], repeat["audio_sha256"])

    def test_locked_submitted_and_sealed_are_conflicts(self):
        repeat_map = self.init()
        locked = repeat_map["slots"][0]["task"]
        with self.assertRaises(self.labelling.LabelConflict):
            self.labelling.save_draft(self.packet, self.labels, locked, {"rttm": ""})
        originals = [r["task"] for r in repeat_map["queue"] if r["kind"] == "original"]
        self.submit(originals[0])
        with self.assertRaises(self.labelling.LabelConflict):
            self.submit(originals[0])
        (self.labels / "SEALED").touch()
        with self.assertRaises(self.labelling.LabelConflict):
            self.labelling.save_draft(self.packet, self.labels, originals[1], {"rttm": ""})


class F18Validation(LabellingFixture):
    def test_rttm_grammar_bounds_and_no_speech(self):
        task = next(r["task"] for r in self.init()["queue"] if r["kind"] == "original")
        with self.assertRaises(ValueError):
            self.labelling.validate_rttm(
                f"SPEAKER {task} 1 124 2 <NA> <NA> spk1 <NA> <NA>\n", task, 125, False)
        with self.assertRaises(ValueError):
            self.labelling.validate_rttm("", task, 125, False)
        self.assertEqual(self.labelling.validate_rttm("", task, 125, True), [])
        rows = self.labelling.validate_rttm(
            f";; note\nNOSCORE {task} 1 1 2\nSPEAKER {task} 1 0 3 <NA> <NA> spk2 <NA> <NA>\n",
            task, 125, False)
        self.assertEqual(len(rows), 2)
        with self.assertRaises(ValueError):
            self.labelling.validate_rttm(
                f"SPEAKER {task} 1 0 3 <NA> <NA> spk2 <NA> <NA>\n"
                f"SPEAKER {task} 1 2 3 <NA> <NA> spk2 <NA> <NA>\n",
                task, 125, False)


class F26RepeatSelection(LabellingFixture):
    def test_seeded_selection_is_deterministic_and_low_speech_is_skipped(self):
        first = self.init()
        other = self.root / "labels2"
        self.labelling.initialize(self.packet, other, "a" * 64)
        second = json.loads((other / "private/repeat-map.json").read_text())
        self.assertEqual(first, second)
        originals = [r["task"] for r in first["queue"] if r["kind"] == "original"]
        self.submit(originals[0], duration=29)
        for task in originals[1:7]:
            self.submit(task)
        chosen = [s.get("source_task") for s in
                  self.labelling.load_state(self.packet, self.labels)["repeat_map"]["slots"]]
        self.assertNotIn(originals[0], chosen)


class F31InventoryAndAck(LabellingFixture):
    def test_inventory_and_ack_verified_trash(self):
        repeat_map = self.init()
        originals = [r["task"] for r in repeat_map["queue"] if r["kind"] == "original"]
        for task in originals:
            self.submit(task)
        state = self.labelling.load_state(self.packet, self.labels)
        for slot in state["repeat_map"]["slots"]:
            self.assertEqual(self.labelling.task_state(state, slot["task"]), "todo")
            self.submit(slot["task"])
        inventory = self.labelling.seal_inventory(self.packet, self.labels)
        self.assertEqual(len(inventory["tasks"]), 28)
        self.assertEqual(sum(row["kind"] == "repeat" for row in inventory["tasks"]), 3)
        self.assertEqual(inventory["bundle_members"], 57)
        (self.labels / "SEALED").touch()
        run = self.root / "run"; run.mkdir()
        seal = run / "labels-seal.json"; seal.write_text(json.dumps({"artifact": "labels-seal"}))
        ack = {"seal_sha256": sha(seal), "files": [], "ok": True,
               "acked_at": "2026-10-27T00:00:00Z", "actor": "custodian",
               "bundle_sha256": "b" * 64, "reference_index_sha256": "c" * 64}
        (run / "custody-ack.json").write_text(json.dumps(ack))
        (run / "labels-ack.json").write_text(
            json.dumps({"custody_ack_sha256": sha(run / "custody-ack.json"),
                        "state": "LABELS_ACKED"}))
        moved = self.labelling.trash_after_ack(run, self.labels, self.root / "Trash")
        self.assertEqual({Path(p).name for p in moved}, {"submitted", "drafts"})
        self.assertFalse((self.labels / "submitted").exists())
        self.assertTrue((self.labels / "private/repeat-map.json").exists())


class F16Confinement(LabellingFixture):
    def first(self):
        repeat_map = self.init()
        task = next(row["task"] for row in repeat_map["queue"]
                    if row["kind"] == "original")
        state = self.labelling.load_state(self.packet, self.labels)
        return state, task, self.labelling.task_info(state, task)

    def test_hash_mismatch_and_unknown_task_fail_closed(self):
        state, task, info = self.first()
        Path(info["audio"]).write_bytes(b"changed")
        with self.assertRaises(KeyError):
            self.labelling._audio_path(state, info)
        with self.assertRaises(KeyError):
            self.labelling.save_draft(self.packet, self.labels, "unknown", {})

    def test_symlink_audio_fails_closed(self):
        state, task, info = self.first()
        audio = Path(info["audio"]); outside = self.root / "outside.wav"
        outside.write_bytes(audio.read_bytes()); audio.unlink(); audio.symlink_to(outside)
        with self.assertRaises(KeyError):
            self.labelling._audio_path(state, info)


class N08N09UiContract(unittest.TestCase):
    def test_responsive_editor_has_required_controls_without_repeat_disclosure(self):
        html = (Path(__file__).resolve().parents[1] / "plaud_workflow/label.html").read_text()
        for marker in ('name="viewport"', '@media(max-width:700px)', '<option>.5</option>',
                       '<option>1.5</option>', 'id="zoom"', 'onpointerdown', 'NOSCORE',
                       'Digit[1-9]', '36000', 'Save draft', 'Submit final'):
            self.assertIn(marker, html)
        self.assertNotIn('window_start', html)
        self.assertNotIn('source_task', html)
        self.assertNotIn('family', html)

if __name__ == "__main__":
    unittest.main()

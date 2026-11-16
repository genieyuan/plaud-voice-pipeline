import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_labelling import LabellingFixture


class WordFixture(LabellingFixture):
    def setUp(self):
        super().setUp()
        from plaud_workflow import labelling_words
        self.words = labelling_words
        self.init()
        self.wordref = self.root / "wordref"
        self.wordref.mkdir(mode=0o700)
        words = self.labels / "words"
        (words / "private").mkdir(parents=True, mode=0o700)
        (words / "drafts").mkdir(mode=0o700)
        (words / "ledger.jsonl").touch(mode=0o600)
        self.marker = "b" * 32
        word_map = self.words.create_word_map(self.packet, self.labels)
        self.words._write_json(words / "init.json", {
            "schema": "plaud-word-tasks/1", "marker": self.marker,
            "key_id": "wordref-diar-20261006",
            "plan_sha256": word_map["plan_sha256"], "wordref_root": str(self.wordref),
            "cap_minutes": 180, "recipient_cert_sha256": "d" * 64,
            "recipient_spki_sha256": "e" * 64,
            "prov_contract_sha256": "f" * 64,
            "openssl": {"path": self.words.OPENSSL, "realpath": self.words.OPENSSL_REALPATH,
                        "sha256": self.words.OPENSSL_SHA256, "version": "OpenSSL 3.6.4"}}, 0o600)

    def submit_original(self, task, languages, code_switch=False, seconds=125):
        self.labelling.save_submission(self.packet, self.labels, task, {
            "rttm": f"SPEAKER {task} 1 0.000 {seconds:.3f} <NA> <NA> spk1 <NA> <NA>\n",
            "languages": languages,
            "strata_human": {"cantonese": False, "code_switch": code_switch},
            "attestations": {"blind": True}, "no_speech": False,
            "active_seconds": 1, "playback_seconds": 1, "edits": 1,
            "session_hash_prefix": "fixture"})


class F00AndF34b(unittest.TestCase):
    def test_normalization_matches_p1_and_unknown_is_other(self):
        from plaud_workflow.labelling_words import normalize_languages
        self.assertEqual(normalize_languages(["yue", "en"]),
                         ["Cantonese", "English"])
        self.assertEqual(normalize_languages(["???"]), ["Other"])
        self.assertEqual(normalize_languages(["English", "Mandarin"]),
                         ["English", "Mandarin"])


class F36Selection(unittest.TestCase):
    def test_stratum_table(self):
        from plaud_workflow.labelling_words import stratum_for
        cases = [
            ((["Cantonese"], False), "Cantonese/code-switch"),
            ((["English", "Mandarin"], True), "Cantonese/code-switch"),
            ((["English", "Mandarin"], False), "Mandarin"),
            ((["English"], False), "English"),
            ((["Other"], False), None), (([], False), None),
        ]
        for args, expected in cases:
            with self.subTest(args=args): self.assertEqual(stratum_for(*args), expected)

    def test_inventory_and_unrelated_provenance_cannot_change_stratum(self):
        from plaud_workflow.labelling_words import provenance_inputs
        base = {"languages": ["yue", "en"], "strata_human": {},
                "audio_sha256": "a", "asr_languages": ["English"],
                "system_output": "SECRET"}
        changed = dict(base, asr_languages=["Mandarin"], system_output="DIFFERENT",
                       recording_name="canary")
        self.assertEqual(provenance_inputs(base), provenance_inputs(changed))
        self.assertEqual(provenance_inputs(base)["stratum"], "Cantonese/code-switch")


class F36SelectionIntegration(WordFixture):
    def test_plan_is_deterministic_and_four_cantonese_yields_void_shortfall(self):
        first = self.words._build_plan(self.packet, self.labels)
        self.assertEqual(first, self.words._build_plan(self.packet, self.labels))
        repeat_path = self.labels / "private/repeat-map.json"
        repeat_map = __import__("json").loads(repeat_path.read_text())
        repeat_map["family_plan"] = []
        self.labelling._atomic_json(repeat_path, repeat_map, 0o400)
        word_map = __import__("json").loads(
            (self.labels / "words/private/word-map.json").read_text())
        for index, plan in enumerate(word_map["plan"]):
            languages = (["Cantonese"] if index < 4 else
                         ["Mandarin"] if index < 9 else ["English"])
            self.submit_original(plan["source_task"], languages)
        result = __import__("json").loads(
            (self.labels / "words/private/word-map.json").read_text())
        self.assertIn({"stratum": "Cantonese/code-switch", "eligible": 4, "needed": 5},
                      result["shortfall"])
        cantonese = [slot for slot in result["slots"]
                     if slot["stratum"] == "Cantonese/code-switch"]
        self.assertEqual(sum(slot["state"] == "chosen" for slot in cantonese), 4)
        self.assertEqual(sum(slot["state"] == "void" for slot in cantonese), 1)


class F37Initialize(LabellingFixture):
    def test_init_creates_pinned_private_layout(self):
        from types import SimpleNamespace
        from plaud_workflow import labelling_words as words
        self.init(); recipient = b"fixture certificate"
        cert_sha = __import__("hashlib").sha256(recipient).hexdigest()
        version = SimpleNamespace(returncode=0, stderr=b"",
                                  stdout=b"OpenSSL 3.6.4 30 Sep 2026\n")
        wordref = self.root / "wordref"
        with mock.patch.object(words, "_check_openssl"):
            with mock.patch.object(words, "_run", return_value=version):
                with mock.patch.object(words, "_certificate_spki", return_value="e" * 64):
                    with mock.patch.object(words, "encrypt_bytes", return_value=b"\x30cipher"):
                        with mock.patch.object(words, "validate_ciphertext", return_value=True):
                            result = words.initialize(self.packet, self.labels, recipient,
                                                      cert_sha, "f" * 64, wordref)
        self.assertEqual(result["recipient_spki_sha256"], "e" * 64)
        self.assertEqual((self.labels / "words").stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.labels / "words/init.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.labels / "words/recipient.pem").stat().st_mode & 0o777, 0o400)
        word_map = self.labels / "words/private/word-map.json"
        self.assertEqual(word_map.stat().st_mode & 0o777, 0o400)
        self.assertFalse((self.labels / "words/.selftest.cms").exists())


class F37StateMachine(WordFixture):
    def chosen(self):
        state = self.labelling.load_state(self.packet, self.labels)
        rows = []
        for row in state["repeat_map"]["queue"]:
            if row["kind"] != "original": continue
            self.submit_original(row["task"], ["English"])
            rows = self.words.list_tasks(self.packet, self.labels)
            if rows: break
        self.assertTrue(rows)
        return rows[0]["task"]

    def valid_body(self, detail, active=1):
        return {"segments": [{"seg": row["seg"], "text": "heard words", "unclear": False}
                             for row in detail["segments"]],
                "active_seconds": active, "attestation_audio_only": True,
                "page_sha256": "1" * 64}

    def test_todo_draft_submitted_modes_and_no_plaintext_ledger(self):
        task = self.chosen(); detail = self.words.get_task(self.packet, self.labels, task)
        body = self.valid_body(detail)
        self.assertEqual(self.words.save_draft(self.packet, self.labels, task, body)["state"], "draft")
        draft = self.labels / "words/drafts" / f"{task}.json"
        self.assertEqual(draft.stat().st_mode & 0o777, 0o600)
        self.assertIn(self.marker, draft.read_text())
        with mock.patch.object(self.words, "encrypt_bytes", return_value=b"\x30cipher"):
            self.assertEqual(self.words.submit(self.packet, self.labels, task, body)["state"], "submitted")
        self.assertFalse(draft.exists())
        cipher = self.wordref / f"{task}.cms"
        self.assertEqual(cipher.stat().st_mode & 0o777, 0o400)
        ledger = (self.labels / "words/ledger.jsonl").read_text()
        self.assertNotIn("heard words", ledger)
        with self.assertRaises(self.words.WordConflict):
            self.words.submit(self.packet, self.labels, task, body)

    def test_validation_and_180_minute_cap(self):
        task = self.chosen(); detail = self.words.get_task(self.packet, self.labels, task)
        body = self.valid_body(detail, 180 * 60)
        with self.assertRaisesRegex(self.words.WordConflict, "180 minutes"):
            self.words.save_draft(self.packet, self.labels, task, body)
        body["active_seconds"] = 179.99 * 60
        self.words.save_draft(self.packet, self.labels, task, body)
        for bad in (math.nan, math.inf, -1):
            body["active_seconds"] = bad
            with self.assertRaises(ValueError):
                self.words.save_draft(self.packet, self.labels, task, body)

    def test_encryption_failure_keeps_draft_and_writes_nothing(self):
        task = self.chosen(); detail = self.words.get_task(self.packet, self.labels, task)
        body = self.valid_body(detail); self.words.save_draft(self.packet, self.labels, task, body)
        with mock.patch.object(self.words, "encrypt_bytes", side_effect=RuntimeError("openssl failed")):
            with self.assertRaises(RuntimeError): self.words.submit(self.packet, self.labels, task, body)
        self.assertTrue((self.labels / "words/drafts" / f"{task}.json").exists())
        self.assertFalse((self.wordref / f"{task}.cms").exists())

    def test_all_validation_failures_are_generic(self):
        task = self.chosen(); detail = self.words.get_task(self.packet, self.labels, task)
        base = self.valid_body(detail)
        cases = []
        empty = dict(base); empty["segments"] = [dict(base["segments"][0], text="")]
        cases.append(empty)
        huge = dict(base); huge["segments"] = [dict(base["segments"][0], text="x" * 2001)]
        cases.append(huge)
        control = dict(base); control["segments"] = [dict(base["segments"][0], text="canary\x00")]
        cases.append(control)
        unattested = dict(base, attestation_audio_only=False); cases.append(unattested)
        for active in (float("nan"), float("inf"), -1):
            cases.append(dict(base, active_seconds=active))
        for body in cases:
            with self.subTest(body=list(body)):
                with self.assertRaises(ValueError):
                    self.words.save_draft(self.packet, self.labels, task, body)

    def test_concurrent_submit_has_one_winner(self):
        import threading
        task = self.chosen(); detail = self.words.get_task(self.packet, self.labels, task)
        body = self.valid_body(detail); outcomes = []
        def run():
            try: outcomes.append(self.words.submit(self.packet, self.labels, task, body)["state"])
            except self.words.WordConflict: outcomes.append("conflict")
        with mock.patch.object(self.words, "encrypt_bytes", return_value=b"\x30cipher"):
            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
        self.assertCountEqual(outcomes, ["submitted", "conflict"])


class F39CipherBuilders(unittest.TestCase):
    def test_exact_encrypt_argv_and_size_limit(self):
        from plaud_workflow import labelling_words as words
        recipient = Path("/tmp/recipient.pem")
        expected = [words.OPENSSL, "cms", "-encrypt", "-binary", "-aes-256-gcm",
                    "-recip", str(recipient), "-keyopt", "rsa_padding_mode:oaep",
                    "-keyopt", "rsa_oaep_md:sha256", "-keyopt",
                    "rsa_mgf1_md:sha256", "-outform", "DER"]
        self.assertEqual(words.encrypt_argv(recipient), expected)
        with self.assertRaises(words.PayloadTooLarge):
            words.encrypt_bytes(b"x" * 262145, recipient, {"path": words.OPENSSL})

    def test_v1_and_v2_argv_are_frozen(self):
        from plaud_workflow import labelling_words as words
        path = Path("fixture.cms")
        self.assertEqual(words.v1_argv(path),
                         [words.OPENSSL, "asn1parse", "-inform", "DER", "-in", str(path)])
        self.assertEqual(words.v2_argv(path),
                         [words.OPENSSL, "cms", "-cmsout", "-print", "-noout",
                          "-inform", "DER", "-in", str(path)])

    def test_stray_stderr_is_rejected(self):
        from types import SimpleNamespace
        from plaud_workflow import labelling_words as words
        stray = SimpleNamespace(returncode=0, stderr=b"warning", stdout=b"\x30cipher")
        with mock.patch.object(words, "_check_openssl"):
            with mock.patch.object(words, "_run", return_value=stray):
                with self.assertRaises(RuntimeError):
                    words.encrypt_bytes(b"reference", Path("recipient.pem"),
                                        {"path": words.OPENSSL})

    def test_exact_success_tuple_is_accepted(self):
        from types import SimpleNamespace
        from plaud_workflow import labelling_words as words
        result = SimpleNamespace(returncode=0, stderr=b"", stdout=b"\x30cipher")
        with mock.patch.object(words, "_check_openssl"):
            with mock.patch.object(words, "_run", return_value=result):
                self.assertEqual(words.encrypt_bytes(b"reference", Path("recipient.pem"),
                                                     {"path": words.OPENSSL}), b"\x30cipher")

    def test_v1_exact_oid_list_and_v2_recipient(self):
        from types import SimpleNamespace
        from plaud_workflow import labelling_words as words
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ref.cms"; path.write_bytes(b"\x30cipher")
            oid_text = "\n".join(f"0:d=0 prim: OBJECT :{name}" for name in words.OID_NAMES)
            results = [SimpleNamespace(returncode=0, stderr=b"", stdout=oid_text.encode()),
                       SimpleNamespace(returncode=0, stderr=b"",
                                       stdout=b"issuer: CN=plaud-word-ref-diar-20261006\nserialNumber: 1\n")]
            with mock.patch.object(words, "_check_openssl"):
                with mock.patch.object(words, "_run", side_effect=results):
                    self.assertTrue(words.validate_ciphertext(path, {"path": words.OPENSSL}))


class F40PlaudSealHalf(WordFixture):
    def choose(self):
        state = self.labelling.load_state(self.packet, self.labels)
        for row in state["repeat_map"]["queue"]:
            if row["kind"] != "original": continue
            self.submit_original(row["task"], ["English"])
            tasks = self.words.list_tasks(self.packet, self.labels)
            if tasks: return tasks[0]["task"]
        self.fail("no eligible word task")

    def test_inventory_modes_digests_and_residue_scan(self):
        task = self.choose(); detail = self.words.get_task(self.packet, self.labels, task)
        body = {"segments": [{"seg": row["seg"], "text": "reference", "unclear": False}
                             for row in detail["segments"]],
                "active_seconds": 2, "attestation_audio_only": True}
        with mock.patch.object(self.words, "encrypt_bytes", return_value=b"\x30cipher"):
            self.words.submit(self.packet, self.labels, task, body)
        with mock.patch.object(self.words, "validate_ciphertext", return_value=True):
            inventory = self.words.seal_inventory(self.packet, self.labels)
        sealed = [row for row in inventory["slots"] if row["state"] == "sealed"]
        self.assertEqual(len(sealed), 1); self.assertEqual(sealed[0]["task"], task)
        self.assertEqual(inventory["scan"]["hits"], [])
        residue = self.labels / "stray.txt"; residue.write_text(self.marker)
        with mock.patch.object(self.words, "validate_ciphertext", return_value=True):
            with self.assertRaisesRegex(ValueError, "residue"):
                self.words.seal_inventory(self.packet, self.labels)

    def test_leftover_draft_refuses_without_state_change(self):
        task = self.choose(); detail = self.words.get_task(self.packet, self.labels, task)
        body = {"segments": [{"seg": row["seg"], "text": "draft", "unclear": False}
                             for row in detail["segments"]],
                "active_seconds": 1, "attestation_audio_only": True}
        self.words.save_draft(self.packet, self.labels, task, body)
        map_path = self.labels / "words/private/word-map.json"; before = map_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "drafts remain"):
            self.words.seal_inventory(self.packet, self.labels)
        self.assertEqual(map_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

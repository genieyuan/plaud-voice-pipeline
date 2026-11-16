import json

from tests.test_labelling_web import F16WebRoutes


class N12PreInit(F16WebRoutes):
    def test_words_route_is_404_before_init(self):
        cookie = self.login()
        status = self.request("/api/label/words", headers={"Cookie": cookie})[0]
        self.assertEqual(status, 404)


class F35WordRoutes(F16WebRoutes):
    def setUp(self):
        super().setUp()
        from plaud_workflow import labelling_words
        self.words = labelling_words; self.wordref = self.root / "wordref"
        self.wordref.mkdir(mode=0o700)
        words = self.labels / "words"
        (words / "private").mkdir(parents=True, mode=0o700)
        (words / "drafts").mkdir(mode=0o700)
        (words / "ledger.jsonl").touch(mode=0o600)
        word_map = self.words.create_word_map(self.packet, self.labels)
        self.words._write_json(words / "init.json", {
            "schema": "plaud-word-tasks/1", "marker": "f" * 32,
            "plan_sha256": word_map["plan_sha256"], "wordref_root": str(self.wordref),
            "cap_minutes": 180, "recipient_cert_sha256": "d" * 64,
            "recipient_spki_sha256": "e" * 64, "prov_contract_sha256": "f" * 64,
            "openssl": {"path": self.words.OPENSSL,
                        "realpath": self.words.OPENSSL_REALPATH,
                        "sha256": self.words.OPENSSL_SHA256}}, 0o600)

    def choose(self):
        state = self.labelling.load_state(self.packet, self.labels)
        for row in state["repeat_map"]["queue"]:
            if row["kind"] != "original": continue
            task = row["task"]
            self.labelling.save_submission(self.packet, self.labels, task, {
                "rttm": f"SPEAKER {task} 1 0 125 <NA> <NA> spk1 <NA> <NA>\n",
                "languages": ["English"], "strata_human": {},
                "attestations": {"blind": True}, "active_seconds": 1,
                "playback_seconds": 1, "edits": 1, "no_speech": False})
            tasks = self.words.list_tasks(self.packet, self.labels)
            if tasks: return tasks[0]["task"]
        self.fail("no eligible transcript task")

    def test_access_guards_and_exact_response_shape(self):
        task = self.choose()
        paths = ["/api/label/words", f"/api/label/words/{task}",
                 f"/api/label/words/{task}/draft", f"/api/label/words/{task}/submit"]
        self.assertEqual(self.request(paths[0])[0], 403)
        cookie = self.login(); auth = {"Cookie": cookie}
        self.assertEqual(self.request(paths[0], headers={**auth, "Host": "evil"})[0], 403)
        self.assertEqual(self.request(paths[0], headers={**auth,
                                                        "Sec-Fetch-Site": "cross-site"})[0], 403)
        status, _, raw = self.request(paths[0], headers=auth)
        self.assertEqual(status, 200)
        self.assertEqual(set(json.loads(raw)[0]), {"task", "seconds", "state"})
        status, _, raw = self.request(paths[1], headers=auth)
        self.assertEqual(status, 200)
        detail = json.loads(raw)
        self.assertEqual(set(detail), {"task", "seconds", "audio_task", "window_start_s",
                                       "segments", "noscore", "draft"})
        self.assertEqual(set(detail["segments"][0]),
                         {"seg", "speaker", "start", "end"})
        for bad in ("w000000000000", "..", "%2e%2e%2fx"):
            self.assertEqual(self.request("/api/label/words/" + bad, headers=auth)[0], 404)

    def test_refused_body_never_echoes_canary(self):
        task = self.choose(); cookie = self.login()
        headers = {"Cookie": cookie, "Origin": f"http://127.0.0.1:{self.port}"}
        canary = "NEVER-ECHO-THIS-WORD-REFERENCE"
        body = {"segments": [{"seg": "s1", "text": canary, "unclear": False}],
                "active_seconds": 1, "attestation_audio_only": False}
        status, _, raw = self.request(f"/api/label/words/{task}/submit",
                                      "POST", body, headers)
        self.assertEqual(status, 400); self.assertNotIn(canary.encode(), raw)

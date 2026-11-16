import http.client
import json
import threading
from http.server import ThreadingHTTPServer

from tests.test_labelling import LabellingFixture


class F16WebRoutes(LabellingFixture):
    def setUp(self):
        super().setUp(); self.init()
        from plaud_workflow.web import Application, handler
        self.config = {"root": str(self.root / "service"),
                       "database": str(self.root / "service/db.sqlite"),
                       "base_url": "http://127.0.0.1:1", "require_session": True,
                       "phase": "test", "notes": str(self.root / "notes"),
                       "label_packet": str(self.packet), "label_root": str(self.labels)}
        self.app = Application(self.config)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.app))
        self.port = self.server.server_port
        self.config["base_url"] = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        super().tearDown()

    def request(self, path, method="GET", data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        values = dict(headers or {}); body = json.dumps(data) if data is not None else None
        if body is not None: values["Content-Type"] = "application/json"
        connection.request(method, path, body, values); response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close(); return result

    def login(self):
        status, _, raw = self.request("/bootstrap", "POST", {}, {"X-Local-Key": self.app.key})
        self.assertEqual(status, 200)
        ticket = json.loads(raw)["url"].split("ticket=", 1)[1]
        status, headers, _ = self.request("/open?ticket=" + ticket)
        self.assertEqual(status, 303)
        return headers["Set-Cookie"].split(";", 1)[0]

    def test_access_audio_and_blind_shapes(self):
        self.assertEqual(self.request("/label")[0], 200)
        self.assertEqual(self.request("/api/label/tasks")[0], 403)
        cookie = self.login(); auth = {"Cookie": cookie}
        status, _, raw = self.request("/api/label/tasks", headers=auth)
        self.assertEqual(status, 200); tasks = json.loads(raw)
        self.assertEqual(len(tasks), 28)
        self.assertEqual(set(tasks[0]), {"task", "seconds", "state"})
        self.assertNotIn(b"family", raw); self.assertNotIn(b"transcript", raw)
        locked = next(row["task"] for row in tasks if row["state"] == "locked")
        self.assertEqual(self.request(f"/api/label/tasks/{locked}", headers=auth)[0], 409)
        original = next(row["task"] for row in tasks if row["state"] == "todo")
        status, _, detail = self.request(f"/api/label/tasks/{original}", headers=auth)
        self.assertEqual(status, 200)
        self.assertEqual(set(json.loads(detail)),
                         {"task", "seconds", "state", "draft", "peaks"})
        status, headers, raw = self.request(f"/api/label/audio/{original}",
                                            headers={**auth, "Range": "bytes=0-1023"})
        self.assertEqual((status, len(raw), headers["Accept-Ranges"]),
                         (206, 1024, "bytes"))
        for path in ("/api/label/tasks/nope", "/api/label/audio/nope", "/label/nope"):
            self.assertEqual(self.request(path, headers=auth)[0], 404)

    def test_post_state_and_security_guards(self):
        cookie = self.login(); origin = f"http://127.0.0.1:{self.port}"
        headers = {"Cookie": cookie, "Origin": origin}
        tasks = json.loads(self.request("/api/label/tasks", headers={"Cookie": cookie})[2])
        task = next(row["task"] for row in tasks if row["state"] == "todo")
        body = {"rttm": f"SPEAKER {task} 1 0 30 <NA> <NA> spk1 <NA> <NA>\n",
                "languages": [], "strata_human": {}, "attestations": {"blind": True},
                "active_seconds": 1, "playback_seconds": 1, "edits": 1,
                "session_hash_prefix": "abc", "no_speech": False}
        draft = f"/api/label/tasks/{task}/draft"
        submit = f"/api/label/tasks/{task}/submit"
        self.assertEqual(self.request(draft, "POST", body, headers)[0], 200)
        self.assertEqual(self.request(submit, "POST", body, headers)[0], 200)
        self.assertEqual(self.request(submit, "POST", body, headers)[0], 409)
        self.assertEqual(self.request(draft, "POST", body, headers)[0], 409)
        self.assertEqual(self.request(draft, "POST", body,
                                      {**headers, "Host": "evil.example"})[0], 403)
        self.assertEqual(self.request(draft, "POST", body,
                                      {**headers, "Sec-Fetch-Site": "cross-site"})[0], 403)

    def test_authenticated_release_probe(self):
        from plaud_workflow import labelling
        result = labelling.probe(
            base_url=f"http://127.0.0.1:{self.port}",
            key_path=self.root / "service/launcher.key")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tasks"], 28)
        self.assertEqual(result["checks"][-1]["bytes"], 1024)

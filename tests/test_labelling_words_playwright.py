import hashlib
import json
import unittest
import wave
from unittest import mock

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:  # optional dev dependency: pip install playwright && playwright install chromium
    raise unittest.SkipTest('playwright is not installed')

from tests.test_labelling_words_web import F35WordRoutes


class N11Playwright(F35WordRoutes):
    def make_browser_audio(self, audio_task):
        audio = self.packet / "audio" / f"{audio_task}.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(125 * 16000 * 2))

        manifest_path = self.packet / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["tasks"]:
            if row["opaque_id"] == audio_task:
                row["audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
                break
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        manifest_path.write_bytes(manifest_bytes)

        init_path = self.labels / "init.json"
        init = json.loads(init_path.read_text())
        init["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        self.labelling._atomic_json(init_path, init, 0o600)

    def choose(self):
        state = self.labelling.load_state(self.packet, self.labels)
        for row in state["repeat_map"]["queue"]:
            if row["kind"] != "original": continue
            task = row["task"]
            rttm = (f"SPEAKER {task} 1 0 125 <NA> <NA> spk1 <NA> <NA>\n"
                    f"SPEAKER {task} 1 10 100 <NA> <NA> spk2 <NA> <NA>\n")
            body = {"rttm": rttm, "languages": ["English"], "strata_human": {},
                    "attestations": {"blind": True}, "active_seconds": 1,
                    "playback_seconds": 1, "edits": 1, "no_speech": False}
            self.labelling.save_submission(self.packet, self.labels, task, body)
            tasks = self.words.list_tasks(self.packet, self.labels)
            if tasks: return tasks[0]["task"]
        self.fail("no eligible transcript task")

    def workflow(self, width):
        task = self.choose()
        detail = self.words.get_task(self.packet, self.labels, task)
        self.make_browser_audio(detail["audio_task"])
        base = f"http://127.0.0.1:{self.port}"
        console_errors = []
        http_errors = []
        with mock.patch.object(self.words, "encrypt_bytes", return_value=b"0cipher"):
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.on("console", lambda msg: console_errors.append(msg.text)
                        if msg.type == "error" else None)
                page.on("response", lambda response: http_errors.append((response.url, response.status))
                        if response.status >= 400 else None)
                ticket = self.app.ticket(self.app.key)
                page.goto(base + "/open?ticket=" + ticket)
                page.goto(base + "/label")
                page.wait_for_selector("#transcripts:not(.hidden)")
                page.locator("#word-tasks button", has_text=task).click()
                page.wait_for_selector("#word-editor:not(.hidden)")
                expect(page.locator(".word-segment")).to_have_count(2)
                self.assertEqual(page.locator(".word-overlap").count(), 2)
                fits = page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                self.assertTrue(fits)
                for _ in range(50):
                    if page.locator("#word-audio").evaluate("a => a.readyState") >= 1: break
                    page.wait_for_timeout(100)
                else: self.fail("word audio metadata did not load")
                page.evaluate("s => { const a=document.querySelector('#word-audio'); a.currentTime=0; a.dispatchEvent(new Event('timeupdate')); }",
                              detail["window_start_s"])
                current = page.locator("#word-audio").evaluate("a => a.currentTime")
                self.assertAlmostEqual(current, detail["window_start_s"], places=1)
                areas = page.locator(".word-segment textarea")
                page.locator("#word-attest").check()
                page.evaluate("""() => { const a=[...document.querySelectorAll('.word-segment textarea')]; a[0].value='first speaker words'; a[1].value='overlapping speaker words'; a[1].focus(); }""")
                page.locator("#word-heading").click()
                expect(page.locator("#word-status")).to_have_text("Draft saved")
                page.reload(); page.wait_for_selector("#transcripts:not(.hidden)")
                page.locator("#word-tasks button", has_text=task).click()
                page.wait_for_selector("#word-editor:not(.hidden)")
                self.assertEqual(page.locator(".word-segment textarea").nth(0).input_value(),
                                 "first speaker words")
                self.assertEqual(page.locator(".word-segment textarea").nth(1).input_value(),
                                 "overlapping speaker words")
                self.assertTrue(page.locator("#word-attest").is_checked())
                page.locator("#word-submit").click()
                expect(page.locator("#word-tasks")).to_contain_text("submitted")
                self.assertNotIn("SYSTEM-CANARY", page.locator("body").inner_text())
                self.assertEqual(console_errors, [], http_errors)
                browser.close()

    def test_375px_workflow(self):
        self.workflow(375)

    def test_1280px_workflow(self):
        self.workflow(1280)

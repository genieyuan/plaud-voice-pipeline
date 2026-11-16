"""CR-06 attribution rule and fallbacks: T-A1..A12, T-X1..X3, T-F1..F10 (+F9b).

Unit tests use generated tokens only. T-A12 and T-X1 ran against the owner's real
recordings and are not part of this copy.
"""
import concurrent.futures
import json
import math
import multiprocessing
import os
import random
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from plaud_workflow import attribution
from plaud_workflow.attribution import Fallback, attribute, flatten, trimmed_word_times
from plaud_workflow.processing import assemble_segments
from plaud_workflow.store import Store

REAL_VERIFY = attribution.verify_assets

from test_attribution_stage import (StageFixture, build_recording, events, fake, nemotron_turns,
                                    run_job)


def chunk(words, offset=0.0, lo=0.0, hi=150.0, per_segment=None):
    """words: [(start, end, text)] relative to the chunk offset; one ASR segment unless split."""
    groups = per_segment or [words]
    segs = [{'start': g[0][0], 'end': g[-1][1], 'text': ''.join(w[2] for w in g),
             'words': [{'start': a, 'end': b, 'word': t} for a, b, t in g]} for g in groups]
    return {'offset': offset, 'core_start': lo, 'core_end': hi, 'segments': segs}


def run(turns, chunks, duration=150.0, current=()):
    result = attribute(turns, chunks, duration, list(current))
    return result


class AttributionRule(unittest.TestCase):
    def test_A1_same_speaker_overlap_counts_once_and_loses_to_more_real_overlap(self):
        words = [(0.0, 2.0, ' a')]
        turns = [(0.0, 1.0, 'A'), (0.5, 1.5, 'A'), (0.0, 1.8, 'B')]
        U, rank = attribution.union_turns(attribution.validate_turns(turns, 150000))
        self.assertEqual(U['A'], [(0, 1500)])
        segs = run(turns, [chunk(words)])
        self.assertEqual([s['speaker'] for s in segs], ['SPEAKER_01'])
        self.assertEqual(attribution.owner(0, 2000, U, rank), 'B')

    def test_A2_exact_tie_goes_to_smaller_rank_and_is_deterministic(self):
        turns = [(1.0, 2.0, 'B'), (0.0, 1.0, 'A')]
        U, rank = attribution.union_turns(attribution.validate_turns(turns, 150000))
        self.assertEqual(attribution.owner(0, 2000, U, rank), 'A')
        same_start = [(0.0, 1.0, 'Z'), (0.0, 1.0, 'M')]
        U2, rank2 = attribution.union_turns(attribution.validate_turns(same_start, 150000))
        self.assertEqual(attribution.owner(0, 1000, U2, rank2), 'M')
        words = [(round(k * 0.3, 3), round(k * 0.3 + 0.3, 3), f' t{k}') for k in range(40)]
        rng = random.Random(7)
        turns = [(rng.uniform(0, 12), rng.uniform(0, 12), rng.choice('ABC')) for _ in range(30)]
        first = json.dumps(run(turns, [chunk(words)]))
        for _ in range(100):
            self.assertEqual(json.dumps(run(turns, [chunk(words)])), first)

    def test_A3_gap_word_takes_nearest_interval_equidistant_takes_smaller_rank(self):
        turns = [(0.0, 1.0, 'Z'), (3.0, 4.0, 'A')]
        U, rank = attribution.union_turns(attribution.validate_turns(turns, 150000))
        self.assertEqual(attribution.owner(1500, 1700, U, rank), 'Z')
        self.assertEqual(attribution.owner(2700, 2900, U, rank), 'A')
        self.assertEqual(attribution.owner(1900, 2100, U, rank), 'Z')  # rank (0,'Z') < (3000,'A')

    def test_A4_leading_and_trailing_words_take_nearest_interval(self):
        turns = [(5.0, 6.0, 'P'), (10.0, 11.0, 'Q')]
        words = [(0.0, 0.4, ' lead'), (5.0, 5.5, ' mid'), (10.2, 10.6, ' q'), (20.0, 20.4, ' trail')]
        segs = run(turns, [chunk(words, per_segment=[[w] for w in words])])
        by_text = {s['text']: s['speaker'] for s in segs}
        self.assertEqual(by_text[' lead'], by_text[' mid'])
        self.assertEqual(by_text[' trail'], by_text[' q'])
        self.assertNotEqual(by_text[' lead'], by_text[' trail'])

    def test_A5_zero_duration_mid_segment_word_attaches_to_previous_group(self):
        turns = [(0.0, 1.05, 'A'), (1.05, 3.0, 'B')]
        words = [(0.0, 1.0, ' w1'), (1.0, 1.0, ' z'), (1.1, 2.0, ' w3')]
        segs = run(turns, [chunk(words)])
        self.assertEqual([(s['text'], s['speaker']) for s in segs], [(' w1 z', 'SPEAKER_01'), (' w3', 'SPEAKER_02')])

    def test_A6_zero_duration_leading_word_is_carried_into_next_group(self):
        turns = [(0.0, 3.0, 'A')]
        words = [(0.5, 0.5, ' z'), (0.6, 1.0, ' w1')]
        segs = run(turns, [chunk(words)])
        self.assertEqual([(s['text'], s['start'], s['end']) for s in segs], [(' z w1', 0.6, 1.0)])

    def test_A7_run_of_untimed_words_between_speakers_attaches_to_previous_group(self):
        turns = [(0.0, 1.0, 'A'), (1.0, 2.0, 'B')]
        words = [(0.0, 1.0, ' w1'), (1.0, 1.0, ' u1'), (1.0, 1.0, ' u2'), (1.0, 1.0, ' u3'), (1.0, 2.0, ' w2')]
        segs = run(turns, [chunk(words)])
        self.assertEqual([(s['text'], s['speaker']) for s in segs],
                         [(' w1 u1 u2 u3', 'SPEAKER_01'), (' w2', 'SPEAKER_02')])

    def test_A8_core_with_untimed_text_and_no_timed_word_falls_back(self):
        turns = [(0.0, 5.0, 'A')]
        words = [(1.0, 1.0, ' u1'), (1.0, 1.0, ' u2')]
        self.assertEqual(run(turns, [chunk(words)]), Fallback('untimed-chunk'))
        # A timed word in another ASR segment of the same core does not rescue it (processing.py:117).
        two = chunk([], per_segment=[[(0.0, 0.5, ' ok')], words])
        self.assertEqual(run(turns, [two]), Fallback('untimed-chunk'))

    def test_A9_invalid_or_absent_turns_and_clipping(self):
        words = [(0.0, 1.0, ' w')]
        self.assertEqual(run([], [chunk(words)]), Fallback('no-valid-turns'))
        bad = [(float('nan'), 1.0, 'A'), (2.0, 2.0, 'A'), (3.0, 1.0, 'A'), (0.0, 1.0, ''), (0.0, 1.0, None),
               (0.0, 1.0, 7), (True, 1.0, 'A'), (0.0, float('inf'), 'A'), ('0', '1', 'A'), (200.0, 300.0, 'A')]
        self.assertEqual(run(bad, [chunk(words)], duration=10.0), Fallback('no-valid-turns'))
        self.assertEqual(attribution.validate_turns([(-5.0, 1000.0, 'A'), (9.9996, 20.0, 'B')], 10000),
                         [(0, 10000, 'A')])
        segs = run([(-5.0, 1000.0, 'A')], [chunk(words, hi=10.0)], duration=10.0)
        self.assertEqual([s['speaker'] for s in segs], ['SPEAKER_01'])

    def test_A10_core_edge_trimming_matches_assemble_segments(self):
        duration = 330.0
        rows = []
        for k in range(0, 660):
            a = round(k * 0.5, 3)
            rows.append((a, round(a + 0.45, 3), f' t{k}'))
        rows += [(149.8, 150.3, ' edgeA'), (149.6, 150.1, ' edgeB'), (299.9, 300.05, ' edgeC'), (329.8, 330.4, ' tail')]
        rows.sort()
        chunks = []
        for i, lo in enumerate(range(0, math.ceil(duration), 150)):
            hi = min(duration, lo + 150)
            start = max(0, lo - 15)
            end = min(duration, hi + 15)
            inside = [(round(a - start, 3), round(b - start, 3), t) for a, b, t in rows if a >= start and a < end]
            chunks.append(chunk(inside, offset=start, lo=lo, hi=hi,
                                per_segment=[inside[j:j + 7] for j in range(0, len(inside), 7)]))
        today = assemble_segments(chunks, [{'start': 0, 'end': duration, 'speaker': 'SPEAKER_01'}], duration)
        new = attribute([(0.0, duration, 'x')], chunks, duration, [{'start': 0, 'end': duration, 'speaker': 'SPEAKER_01'}])
        self.assertEqual([(s['id'], round(s['start'], 6), round(s['end'], 6), s['speaker'], s['text']) for s in new],
                         [(s['id'], round(s['start'], 6), round(s['end'], 6), s['speaker'], s['text']) for s in today])

    def test_A11_no_unassigned_and_labels_by_first_appearance(self):
        rng = random.Random(11)
        for trial in range(25):
            words = []
            t = 0.0
            for k in range(200):
                width = rng.choice([0.0, 0.2, 0.4, 0.7])
                words.append((round(t, 3), round(t + width, 3), f' t{k}'))
                t += rng.uniform(0.1, 0.8)
            words[0] = (words[0][0], round(words[0][0] + 0.3, 3), words[0][2])
            turns = [(rng.uniform(0, t), rng.uniform(0, t), str(rng.randint(0, 5))) for _ in range(40)]
            segs = run(turns, [chunk(words, per_segment=[words[j:j + 9] for j in range(0, 200, 9)])])
            if isinstance(segs, Fallback):
                self.assertIn(segs.reason, ('no-valid-turns', 'untimed-chunk'))
                continue
            labels = [s['speaker'] for s in segs]
            self.assertNotIn('UNASSIGNED', labels)
            order = list(dict.fromkeys(labels))
            self.assertEqual(order, [f'SPEAKER_{i + 1:02d}' for i in range(len(order))])


class TextAndTiming(unittest.TestCase):
    def test_X2_every_boundary_is_a_trimmed_word_time_of_its_chunk(self):
        rng = random.Random(3)
        words = [(round(k * 0.37, 3), round(k * 0.37 + rng.choice([0.0, 0.3, 0.33]), 3), f' t{k}') for k in range(400)]
        words[0] = (0.0, 0.3, ' t0')
        turns = [(rng.uniform(0, 148), rng.uniform(0, 148), rng.choice('ABCD')) for _ in range(50)]
        c = chunk(words, per_segment=[words[j:j + 11] for j in range(0, 400, 11)])
        segs = run(turns, [c])
        times = trimmed_word_times(c, 150000)
        for s in segs:
            self.assertIn(round(s['start'] * 1000), times)
            self.assertIn(round(s['end'] * 1000), times)

    def test_X3_untimed_words_never_create_a_boundary(self):
        turns = [(0.0, 1.05, 'A'), (1.05, 3.0, 'B')]
        words = [(0.0, 1.0, ' w1'), (1.05, 1.05, ' z1'), (1.07, 1.07, ' z2'), (1.1, 2.0, ' w3'), (2.5, 2.5, ' z3')]
        segs = run(turns, [chunk(words)])
        boundaries = {round(x * 1000) for s in segs for x in (s['start'], s['end'])}
        self.assertTrue(boundaries.isdisjoint({1050, 1070, 2500}))
        self.assertEqual(boundaries, {0, 1000, 1100, 2000})


# ------------------------------------------------------------------- T-F (each ends complete)

class Fallbacks(StageFixture, unittest.TestCase):
    def assert_fallback(self, reason, rid='r1', built=None, stage='attribute'):
        before = self.store.segments(rid)
        version = self.store.get_recording(rid)['version']
        job = run_job(self.c, self.store, rid, stage=stage)
        rec = self.store.get_recording(rid)
        self.assertEqual((job['stage'], job['state'], job['error']), ('compare', 'complete', None))
        self.assertEqual(rec['metadata'].get('speaker_source'), 'current-fallback')
        self.assertEqual(rec['metadata'].get('speaker_fallback_reason'), reason)
        self.assertEqual(rec['version'], version)
        self.assertEqual(self.store.segments(rid), before)
        self.assertEqual([e['reason'] for e in events(self.store, 'speaker_fallback', rid)], [reason])
        self.assertEqual(events(self.store, 'speaker_attribution', rid), [])
        self.assertFalse(self.wav(rid).exists())
        return rec

    def test_F1_asset_invalid(self):
        build_recording(self.root, self.store)
        with mock.patch('plaud_workflow.attribution.verify_assets',
                        side_effect=attribution.AssetInvalid('missing')):
            self.assert_fallback('asset-invalid')

    def test_F1_asset_invalid_real_verifier_on_fake_home(self):
        build_recording(self.root, self.store)
        with mock.patch('plaud_workflow.attribution.verify_assets', REAL_VERIFY):
            self.assert_fallback('asset-invalid')

    def test_F2_decode_failed(self):
        build_recording(self.root, self.store)
        self.c['runtimes']['ffmpeg'] = str(self.root / 'absent-ffmpeg')
        self.assert_fallback('decode-failed')
        build_recording(self.root, self.store, 'r2')
        from test_attribution_stage import fake_ffmpeg
        self.c['runtimes']['ffmpeg'] = str(fake_ffmpeg(self.root, fail=True))
        self.assert_fallback('decode-failed', 'r2')

    def test_F3_nemotron_exit(self):
        build_recording(self.root, self.store)
        self.set_fake(exit='3')
        self.assert_fallback('nemotron-exit')

    def test_F4_nemotron_timeout_kills_the_group_and_leaves_no_orphan(self):
        info = self.root / 'info.json'
        build_recording(self.root, self.store)
        self.set_fake(sleep='30', info=str(info))
        self.c['nemotron'].update(deadline_floor_s=1, deadline_ratio=0)
        started = time.monotonic()
        self.assert_fallback('nemotron-timeout')
        self.assertLess(time.monotonic() - started, 10)
        pid = json.loads(info.read_text())['pid']
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        with self.assertRaises(ProcessLookupError):
            os.killpg(pid, 0)

    def test_F5_no_valid_turns(self):
        for i, controls in enumerate(({'mode': 'invalid'}, {'mode': 'empty'}, {'mode': 'missing'},
                                      {'turns': [{'start': 3, 'end': 1, 'speaker': 1}]})):
            with self.subTest(controls=controls):
                build_recording(self.root, self.store, f'n{i}')
                self.set_fake(**controls)
                self.assert_fallback('no-valid-turns', f'n{i}')

    def test_F6_words_missing(self):
        build_recording(self.root, self.store)
        (self.root / 'work' / 'r1' / 'chunk-0000' / 'asr.json').unlink()
        self.assert_fallback('words-missing')

    def test_F7_untimed_chunk(self):
        build_recording(self.root, self.store)
        path = self.root / 'work' / 'r1' / 'chunk-0000' / 'asr.json'
        data = json.loads(path.read_text())
        data['segments'].append({'start': 39.5, 'end': 39.5, 'text': ' u',
                                 'words': [{'start': 39.5, 'end': 39.5, 'word': ' u'}]})
        path.write_text(json.dumps(data))
        self.assert_fallback('untimed-chunk')

    def test_F8_text_invariant(self):
        build_recording(self.root, self.store)
        real = attribution.current_segments

        def mismatch(*a):
            segs = real(*a)
            segs[0] = dict(segs[0], text=segs[0]['text'] + ' extra')
            return segs
        with mock.patch.object(attribution, 'current_segments', side_effect=mismatch):
            self.assert_fallback('text-invariant')

    def test_F9_named_before_attribution_keeps_the_name(self):
        build_recording(self.root, self.store)
        seg = self.store.segments('r1')[0]
        pid = self.store.profile('Synthetic Person')
        self.store.identify('r1', seg['id'], pid, 'segment', 1, 'k1', seg['start'], min(seg['end'], seg['start'] + 1), None)
        rec = self.assert_fallback('named-before-attribution')
        self.assertEqual(self.store.segments('r1')[0]['profile_id'], pid)
        self.assertEqual(rec['version'], 2)

    def test_F9b_name_saved_while_nemotron_runs_gives_version_changed(self):
        build_recording(self.root, self.store)
        barrier = self.root / 'barrier'
        self.set_fake(turns=nemotron_turns(40.0), barrier=str(barrier))
        seg = self.store.segments('r1')[0]
        pid = self.store.profile('Synthetic Person')

        def name_during_run():
            deadline = time.time() + 20
            while not Path(str(barrier) + '.waiting').exists() and time.time() < deadline:
                time.sleep(0.02)
            other = Store(self.root / 'state.sqlite')
            try:
                other.identify('r1', seg['id'], pid, 'segment', 1, 'k1', seg['start'],
                               min(seg['end'], seg['start'] + 1), None)
            finally:
                other.close()
            barrier.write_text('')
        helper = threading.Thread(target=name_during_run)
        helper.start()
        try:
            before = [dict(s, profile_id=None if s['id'] != seg['id'] else s['profile_id']) for s in self.store.segments('r1')]
            job = run_job(self.c, self.store, 'r1')
        finally:
            helper.join(30)
        rec = self.store.get_recording('r1')
        self.assertEqual((job['state'], rec['metadata']['speaker_source'], rec['metadata']['speaker_fallback_reason']),
                         ('complete', 'current-fallback', 'version-changed'))
        rows = self.store.segments('r1')
        self.assertEqual(rows[0]['profile_id'], pid)
        self.assertEqual([(s['id'], s['speaker'], s['text']) for s in rows], [(s['id'], s['speaker'], s['text']) for s in before])
        self.assertEqual(rec['version'], 2)
        self.assertFalse(self.wav().exists())

    def test_F10_internal_exception_class(self):
        build_recording(self.root, self.store)
        with mock.patch('plaud_workflow.attribution.attribute', side_effect=ZeroDivisionError('boom')):
            self.assert_fallback('internal-ZeroDivisionError')

    def test_F_service_safety_net_never_reaches_the_retry_path(self):
        build_recording(self.root, self.store)
        with mock.patch('plaud_workflow.attribution.stage', side_effect=MemoryError()):
            self.assert_fallback('internal-MemoryError')


if __name__ == '__main__':
    unittest.main()

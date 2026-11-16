import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plaud_workflow.processing import suggest_speakers
from plaud_workflow.store import Store


class SuggestionContract(unittest.TestCase):
    """CR-05 D-A / Gate 2 F03. A person named once is offered in later recordings."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        self.s = Store(d / 'state.sqlite')
        self.s.recording('r1', 'Earlier', 'h', '/a.wav', 100)
        self.s.transcript('r1', [{'id': 'a', 'start': 0, 'end': 20, 'speaker': 'SPEAKER_01', 'text': 'hello'}])
        self.p = self.s.profile('Person A')
        self.s.identify('r1', 'a', self.p, 'segment', 1, 'k1', 0, 5, [1.0, 0.0])
        self.s.recording('r2', 'Later', 'h2', '/b.wav', 100)
        self.s.transcript('r2', [
            {'id': 'x', 'start': 0, 'end': 20, 'speaker': 'SPEAKER_01', 'text': 'same person'},
            {'id': 'y', 'start': 30, 'end': 50, 'speaker': 'SPEAKER_02', 'text': 'different person'},
            {'id': 'z', 'start': 60, 'end': 80, 'speaker': 'UNASSIGNED', 'text': 'uncertain'}])
        self.c = {'runtimes': {}, 'models': {}}

    def tearDown(self):
        self.s.close(); self.tmp.cleanup()

    def run_with(self, vec):
        with patch('plaud_workflow.processing.embedding', return_value=vec):
            return suggest_speakers(self.c, 'r2', self.s)

    def test_known_voice_is_offered_in_a_later_recording(self):
        out = self.run_with([1.0, 0.0])
        self.assertIn('SPEAKER_01', out)
        self.assertEqual(out['SPEAKER_01'][0][0], 'Person A')
        self.assertTrue(any(g['speaker'] == 'SPEAKER_01' for g in self.s.suggestions('r2')))

    def test_a_suggestion_never_writes_an_identity(self):
        self.run_with([1.0, 0.0])
        self.assertTrue(all(s['profile_id'] is None for s in self.s.segments('r2')),
                        'suggestion must not set profile_id — only a human Save may')

    def test_unlike_voice_is_not_offered(self):
        self.assertEqual(self.run_with([-1.0, 0.0]), {})

    def test_unassigned_turns_are_never_suggested_for(self):
        self.run_with([1.0, 0.0])
        self.assertNotIn('UNASSIGNED', {g['speaker'] for g in self.s.suggestions('r2')})

    def test_confirmed_speaker_is_never_rescored(self):
        # Design Decision 1 invariant: a segment whose profile_id is set
        # never re-enters scoring, so neither a correction nor a new enrolment can
        # relabel an already-confirmed segment.
        self.s.identify('r2', 'x', self.p, 'segment', 1, 'k2', 0, 5, [1.0, 0.0])
        seen = []
        def spy(c, src, a, b):
            seen.append((a, b)); return [1.0, 0.0]
        with patch('plaud_workflow.processing.embedding', side_effect=spy):
            suggest_speakers(self.c, 'r2', self.s)
        self.assertFalse(any(a == 0 for a, _ in seen),
                         'confirmed SPEAKER_01 segment was re-scored')

    def test_rerun_replaces_rather_than_accumulates(self):
        self.run_with([1.0, 0.0]); first = len(self.s.suggestions('r2'))
        self.run_with([1.0, 0.0])
        self.assertEqual(len(self.s.suggestions('r2')), first)

    def test_no_profiles_means_no_suggestions(self):
        empty = Store(Path(self.tmp.name) / 'empty.sqlite')
        empty.recording('r', 'n', 'h', '/c.wav', 50)
        empty.transcript('r', [{'id': 'a', 'start': 0, 'end': 20, 'speaker': 'SPEAKER_01', 'text': 'x'}])
        with patch('plaud_workflow.processing.embedding', return_value=[1.0, 0.0]):
            self.assertEqual(suggest_speakers(self.c, 'r', empty), {})
        empty.close()

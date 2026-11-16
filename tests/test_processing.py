import unittest
from plaud_workflow.processing import group_speakers, assemble_segments, validate_summary


class ProcessingContract(unittest.TestCase):
    def test_chunk_local_names_do_not_define_global_identity(self):
        chunks=[{'offset':0,'turns':[{'start':0,'end':8,'local_speaker':0,'embedding':[1,0]}]},
                {'offset':150,'turns':[{'start':0,'end':8,'local_speaker':1,'embedding':[1,0]},
                                      {'start':9,'end':15,'local_speaker':0,'embedding':[0,1]}]}]
        turns=group_speakers(chunks)
        self.assertEqual(turns[0]['speaker'],turns[1]['speaker'])
        self.assertNotEqual(turns[0]['speaker'],turns[2]['speaker'])

    def test_word_grouping_splits_speaker_change_without_losing_zero_time_text(self):
        a=[{'offset':0,'core_start':0,'core_end':10,'segments':[{'start':0,'end':8,'text':'hello yes again',
            'words':[{'start':0,'end':3,'word':'hello'},{'start':4,'end':7,'word':' yes'},
                     {'start':7,'end':7,'word':' again'}]}]}]
        t=[{'start':0,'end':3.5,'speaker':'S01'},{'start':4,'end':8,'speaker':'S02'}]
        s=assemble_segments(a,t,10)
        self.assertEqual([x['speaker'] for x in s],['S01','S02'])
        self.assertEqual(''.join(x['text'] for x in s),'hello yes again')
        self.assertTrue(all(0<=x['start']<x['end']<=10 for x in s))

    def test_summary_cannot_cite_invented_segments(self):
        with self.assertRaises(ValueError):validate_summary([{'text':'x','segments':['invented']}],{'a'})

    def test_fragmented_local_cluster_cannot_mix_distinct_voices(self):
        chunks=[{'offset':0,'turns':[{'start':0,'end':8,'local_speaker':0,'embedding':[1,0]},
                                   {'start':9,'end':17,'local_speaker':0,'embedding':[0,1]},
                                   {'start':18,'end':19,'local_speaker':9,'embedding':[.7,.7]}]}]
        t=group_speakers(chunks)
        self.assertNotEqual(t[0]['speaker'],t[1]['speaker'])
        self.assertEqual(t[2]['speaker'],'UNASSIGNED')

    def test_leading_zero_duration_text_is_preserved(self):
        chunks=[{'offset':0,'core_start':0,'core_end':10,'segments':[{'words':[
            {'start':0,'end':0,'word':'Hello '},{'start':0,'end':2,'word':'there'}]}]}]
        result=assemble_segments(chunks,[],10)
        self.assertEqual(result[0]['text'],'Hello there')

    def test_extractive_summary_cannot_publish_model_written_claims(self):
        import tempfile
        from unittest.mock import patch
        from plaud_workflow.processing import experimental_extractive_summary,select_passages
        source=[{'id':'s','start':0,'end':8,'text':"A synthetic option is possible, not an agreement to buy."}]
        with tempfile.TemporaryDirectory() as d,patch('plaud_workflow.processing.local_json',return_value={'ids':['0'],'invented_claim':'We decided to buy.'}):
            items=experimental_extractive_summary({'summary_model':'fixture'},source,d)
        self.assertEqual(items[0]['text'],source[0]['text'])
        self.assertEqual(items[0]['category'],'Topic')
        self.assertEqual(items[0]['segments'],['s'])
        with patch('plaud_workflow.processing.local_json',return_value={'ids':['missing']}):
            with self.assertRaises(ValueError):select_passages({},[{'id':'0','text':'Source'}],1)

    def test_unaccepted_summary_cannot_reach_publication(self):
        from unittest.mock import patch
        from plaud_workflow.processing import summarize
        with patch('plaud_workflow.processing.local_json') as inference:
            with self.assertRaisesRegex(RuntimeError,'accuracy has not passed'):
                summarize({},[{'text':'Source'}],None)
            inference.assert_not_called()


class UnknownIsNotAVote(unittest.TestCase):
    """A reviewed example: a confident 6.36s SPEAKER_01 turn lost its words
    to two unknown scraps that overlapped each other."""

    def assemble(self,turns):
        from plaud_workflow.processing import assemble_segments
        chunks=[{'offset':0,'core_start':0,'core_end':600,'segments':[{'words':[
            {'start':460.76,'end':461.40,'word':'synthetic words'}]}]}]
        return assemble_segments(chunks,turns,600)

    def test_overlapping_unknown_scraps_do_not_outvote_a_named_turn(self):
        out=self.assemble([{'start':457.14,'end':463.50,'speaker':'SPEAKER_01'},
                           {'start':460.68,'end':461.22,'speaker':'UNASSIGNED'},
                           {'start':460.73,'end':461.20,'speaker':'UNASSIGNED'}])
        self.assertEqual(out[0]['speaker'],'SPEAKER_01')

    def test_unknown_still_wins_where_it_genuinely_covers_the_word(self):
        # A speaker clipping the very edge of the word must not claim it.
        out=self.assemble([{'start':455.0,'end':460.85,'speaker':'SPEAKER_01'},
                           {'start':460.80,'end':462.0,'speaker':'UNASSIGNED'}])
        self.assertEqual(out[0]['speaker'],'UNASSIGNED')

    def test_two_named_speakers_are_still_decided_on_coverage(self):
        # The word runs 460.76-461.40: SPEAKER_01 covers 0.24s of it, SPEAKER_02 0.50s.
        out=self.assemble([{'start':455.0,'end':461.0,'speaker':'SPEAKER_01'},
                           {'start':460.9,'end':470.0,'speaker':'SPEAKER_02'}])
        self.assertEqual(out[0]['speaker'],'SPEAKER_02')

    def test_a_word_no_turn_covers_is_still_unknown(self):
        out=self.assemble([{'start':100.0,'end':200.0,'speaker':'SPEAKER_01'}])
        self.assertEqual(out[0]['speaker'],'UNASSIGNED')

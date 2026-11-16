import unittest
from dataclasses import dataclass

import numpy as np

from plaud_workflow.qwen_asr_worker import (AlignmentBreakdown, aligned_words, looping,
                                            realign_tail, restore_punctuation)


@dataclass
class Word:
    text: str
    start_time: float
    end_time: float


class FakeAligner:
    """Stands in for Qwen3-ForcedAligner: returns whatever it was told to."""
    def __init__(self,result):self.result=result;self.calls=[]
    def generate(self,speech,text,language):
        self.calls.append((len(speech),text,language));return self.result


class AlignmentEndsAtTheBreakdown(unittest.TestCase):
    """Qwen3-ForcedAligner does not clamp its timestamps to the audio it was given
    (QwenLM/Qwen3-ASR#197), and the library's LIS repair interpolates every word
    after the first bad one - so a word past the end ends the alignment."""

    def items(self,spans):
        return [{'text':f'w{i}','start_time':a,'end_time':b} for i,(a,b) in enumerate(spans)]

    def test_nothing_after_a_word_past_the_end_is_kept(self):
        # The last word lands back inside the audio, but it was interpolated from
        # the one that escaped: keeping it would assert a time nobody measured.
        timed,unused=aligned_words(self.items([(0,1),(1,2),(9,14),(3,4)]),10,2)
        self.assertEqual([(w['start'],w['end']) for w in timed],[(0,1),(1,2)])
        self.assertEqual([w['text'] for w in unused],['w2','w3'])

    def test_breakdown_inside_the_core_is_raised_not_swallowed(self):
        # Core runs to 8s but alignment gave up at 2s: the next chunk starts at 8s
        # and will never transcribe 2-8s, so this cannot be dropped quietly.
        with self.assertRaises(AlignmentBreakdown) as caught:
            aligned_words(self.items([(0,1),(1,2),(3,14)]),10,8)
        self.assertEqual(caught.exception.index,2)
        self.assertEqual(caught.exception.settled,2)

    def test_a_clean_alignment_still_uses_every_word(self):
        timed,unused=aligned_words(self.items([(0,1),(1,2),(2,3)]),10,10)
        self.assertEqual(len(timed),3);self.assertEqual(unused,[])


class RunawayDecoding(unittest.TestCase):
    def test_a_repetition_loop_is_recognised(self):
        self.assertTrue(looping('. Yeah'*400))
        self.assertTrue(looping('哈！'*700))

    def test_ordinary_speech_is_not(self):
        # Real transcripts repeat too; the worst sub-section in the library reaches
        # 26.5% coverage, so agreement and filler must stay well clear.
        self.assertFalse(looping('Yeah. '*3+'So this synthetic fixture reads one long and ordinary sentence, '
                                 'and then it carries on with other plain words for a while. '*4))
        self.assertFalse(looping('好的，好的，好的。'+'这是一个合成的测试句子，用来检查普通说话不会被当成循环。'*6))

    def test_a_short_line_is_never_called_a_loop(self):
        self.assertFalse(looping('OK. OK.'))


class RetryingABrokenTail(unittest.TestCase):
    def setUp(self):
        self.samples=np.zeros(160000,dtype=np.int16)   # 10s at 16 kHz
        self.items=[{'text':'a','start_time':0,'end_time':1},
                    {'text':'b','start_time':1,'end_time':2},
                    {'text':'c','start_time':9,'end_time':14}]
        self.broke=AlignmentBreakdown(2,2.)

    def test_a_placeable_tail_comes_back_on_the_original_timeline(self):
        model=FakeAligner([Word('c',1.,1.5)])
        spliced=realign_tail(model,self.samples,16000,'English',self.items,self.broke,10.)
        self.assertEqual([w['text'] for w in spliced],['a','b','c'])
        self.assertEqual((spliced[2]['start_time'],spliced[2]['end_time']),(3.,3.5))
        self.assertEqual(model.calls[0][1],'c')

    def test_a_retry_that_rewrites_the_words_is_refused(self):
        model=FakeAligner([Word('different',1.,1.5)])
        self.assertIsNone(realign_tail(model,self.samples,16000,'English',self.items,self.broke,10.))

    def test_a_retry_that_overshoots_again_is_refused(self):
        model=FakeAligner([Word('c',1.,90.)])
        self.assertIsNone(realign_tail(model,self.samples,16000,'English',self.items,self.broke,10.))

    def test_no_audio_left_means_no_retry(self):
        model=FakeAligner([Word('c',0.,.1)])
        self.assertIsNone(realign_tail(model,self.samples,16000,'English',self.items,
                                       AlignmentBreakdown(2,10.),10.))
        self.assertEqual(model.calls,[])


class SplicedTailStillMatchesTheSource(unittest.TestCase):
    def test_restore_punctuation_accepts_the_spliced_sequence(self):
        # The retry must not disturb the lexical contract: same words, same order.
        items=[{'text':'Done','start_time':0,'end_time':1},{'text':'Next','start_time':9,'end_time':14}]
        model=FakeAligner([Word('Next',0.5,1.5)])
        spliced=realign_tail(model,np.zeros(160000,dtype=np.int16),16000,'English',
                             items,AlignmentBreakdown(1,1.),10.)
        restored=restore_punctuation('Done. Next?',spliced)
        self.assertEqual(''.join(w['display_word'] for w in restored),'Done. Next?')
        timed,unused=aligned_words(restored,10,10)
        self.assertEqual(len(timed),2);self.assertEqual(unused,[])


if __name__=='__main__':unittest.main()

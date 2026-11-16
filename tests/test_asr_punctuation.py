import unittest
from plaud_workflow.qwen_asr_worker import restore_punctuation,aligned_words
from plaud_workflow.processing import passages


class ASRPunctuation(unittest.TestCase):
    def test_source_punctuation_and_timestamps_survive_alignment(self):
        for text,words in [
            ('“Three apples per crate?” No, pears.',['Three','apples','per','crate','No','pears']),
            ("I don't think so. Already done!",['I','dont','think','so','Already','done']),
            ('不是三个，是两个。好吗？',['不是','三','个','是','两','个','好','吗']),
        ]:
            with self.subTest(text=text):
                items=[{'text':w,'start_time':i,'end_time':i+.5} for i,w in enumerate(words)]
                restored=restore_punctuation(text,items)
                self.assertEqual(''.join(w['display_word'] for w in restored),text)
                self.assertEqual([{k:v for k,v in w.items() if k!='display_word'} for w in restored],items)
                timed,unused=aligned_words(restored,len(words))
                self.assertEqual(''.join(w['word'] for w in timed),text)
                self.assertEqual([(w['start'],w['end']) for w in timed],[(i,i+.5) for i in range(len(words))])
                self.assertEqual(unused,[])
                self.assertTrue(all('display_word' not in w for w in items))

    def test_mismatched_words_fail_without_rewriting_the_source(self):
        with self.assertRaisesRegex(ValueError,'words differ'):
            restore_punctuation('Already done.',[{'text':'Not done','start_time':0,'end_time':1}])

    def test_invalid_overlap_text_is_not_reattached_to_the_core(self):
        items=[{'text':'Done','start_time':1,'end_time':2},
               {'text':'Next','start_time':9,'end_time':12}]
        timed,unused=aligned_words(restore_punctuation('Done. Next?',items),10,8)
        self.assertEqual(timed,[{'start':1,'end':2,'word':'Done. '}])
        self.assertEqual(unused[0]['display_word'],'Next?')

    def test_summary_passage_keeps_question_and_reply_on_separate_lines(self):
        rows=[{'id':'a','start':0,'end':1,'text':'Per crate?'},
              {'id':'b','start':1,'end':2,'text':'No, pears.'}]
        p=passages(rows)[0]
        self.assertEqual(p['text'],'Per crate?\nNo, pears.')
        self.assertEqual(p['segments'],['a','b'])

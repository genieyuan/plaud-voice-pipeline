import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from plaud_workflow.summary_contract import validate,confirmed_source_labels
from plaud_workflow.summary_contract import sections,verify_section_coverage
from plaud_workflow.summary_worker import verify
from plaud_workflow.processing import assemble_segments,summarize


class SummaryContract(unittest.TestCase):
    def test_time_sections_preserve_every_source_passage_once(self):
        source=[{'id':str(i),'start':i*30,'end':(i+1)*30,'text':'Meeting content'} for i in range(64)]
        groups=sections(source)
        self.assertEqual(len(groups),4)
        self.assertEqual([s['id'] for g in groups for s in g],[s['id'] for s in source])
        self.assertEqual(groups[-1][-1]['id'],'63')

    def test_long_meeting_budget_preserves_end_without_global_merge(self):
        c={'summary_commissioned':True,'summary_url':'local','summary_model':'expected','summary_model_path':'path','summary_config_sha256':'h'}
        source=[{'id':str(i),'start':i*60,'end':i*60+20,'text':'Substantive content in this interval.'} for i in range(82)]
        jobs=[]
        def run(args,**kwargs):
            job=json.loads(Path(args[-1]).read_text());jobs.append(job)
            refs=[r['id'] for r in job['source'] if r['id'] not in (job.get('context_ids') or [])]
            Path(job['output']).write_text(json.dumps([{'category':'Topic','text':'A section summary','segments':[refs[-1]]}]))
        with tempfile.TemporaryDirectory() as d,patch('plaud_workflow.processing.subprocess.run',side_effect=run):
            items=summarize(c,source,d)
            self.assertEqual(len(items),10);self.assertEqual(len(jobs),10)
            self.assertTrue(all(job['limit']==1 and not job['claims'] for job in jobs))
            self.assertIn('81',items[-1]['segments'])

    def test_twelve_input_sections_merge_only_overflow_and_preserve_all_sources(self):
        c={'summary_commissioned':True,'summary_url':'local','summary_model':'expected','summary_model_path':'path','summary_config_sha256':'h'}
        source=[{'id':str(i),'start':i*100,'end':i*100+20,'text':'A substantive source section.'} for i in range(12)]
        jobs=[]
        def run(args,**kwargs):
            job=json.loads(Path(args[-1]).read_text());jobs.append(job)
            refs=[g[0] for g in job['required_groups']] if job['claims'] else [job['source'][-1]['id']]
            Path(job['output']).write_text(json.dumps([{'category':'Topic','text':'Supported section content','segments':refs}]))
        with tempfile.TemporaryDirectory() as d,patch('plaud_workflow.summary_contract.sections',side_effect=lambda rows:[[r] for r in rows]),patch('plaud_workflow.processing.subprocess.run',side_effect=run):
            items=summarize(c,source,d)
            self.assertEqual(len(items),10)
            self.assertEqual({sid for item in items for sid in item['segments']},{s['id'] for s in source})
            merges=[j for j in jobs if j['claims']]
            self.assertEqual(len(merges),1);self.assertEqual(len(merges[0]['required_groups']),3)
            self.assertEqual(len(jobs),13)

    def test_late_topic_cannot_be_hidden_by_beginning_and_closing_citations(self):
        groups=[[{'id':str(i)}] for i in range(4)]
        with self.assertRaisesRegex(ValueError,'sections: 3'):
            verify_section_coverage([{'segments':['0','1','3']}],groups)
        verify_section_coverage([{'segments':['0','1']},{'segments':['2','3']}],groups)

    def test_context_only_item_is_retained_in_audit_without_hiding_owned_content(self):
        from plaud_workflow.summary_contract import owned_claims
        previous={'segments':['34'],'text':'Previous-section feedback'}
        own={'segments':['38'],'text':'Current-section framework'}
        self.assertEqual(owned_claims([previous,own],[{'id':'38'}]),([own],[previous]))
        with self.assertRaisesRegex(ValueError,'only cited adjacent'):owned_claims([previous],[{'id':'38'}])

    def test_large_section_cannot_exceed_input_or_call_bounds(self):
        source=[{'id':str(i),'start':i,'end':i+1,'text':'x'*20000} for i in range(13)]
        with self.assertRaisesRegex(ValueError,'twelve'):sections(source)

    def test_adjacent_context_counts_toward_leaf_input_limit(self):
        c={'summary_commissioned':True,'summary_url':'local','summary_model':'expected','summary_model_path':'path','summary_config_sha256':'h'}
        source=[{'id':str(i),'start':i*500,'end':i*500+20,'text':'x'*15000} for i in range(2)]
        def run(args,**kwargs):
            job=json.loads(Path(args[-1]).read_text())
            Path(job['output']).write_text(json.dumps([{'category':'Topic','text':'A claim','segments':[job['source'][0]['id']]}]))
        with tempfile.TemporaryDirectory() as d,patch('plaud_workflow.processing.subprocess.run',side_effect=run) as worker:
            with self.assertRaisesRegex(ValueError,'including adjacent context'):summarize(c,source,d)
            self.assertEqual(worker.call_count,1)

    def test_actual_invented_person_failure_rejected(self):
        source=[{'id':'s','text':'Source'}]
        for text in ['Another speaker responds positively to the different perspective.',
                     'The speaker requests a website review.']:
            with self.assertRaisesRegex(ValueError,'person narration'):
                validate([{'category':'Topic','text':text,'segments':['s']}],source)

    def test_subject_topic_not_confused_with_human_narration(self):
        item={'category':'Topic','text':'Speaker recognition requires reliable samples.','segments':['s']}
        self.assertEqual(validate([item],[{'id':'s'}]),[item])

    def test_saved_correction_changes_provenance_only(self):
        item={'category':'Action','text':'A website review is requested.','segments':['s']}
        source={'s':{'profile_name':'Person A'}}
        self.assertEqual(confirmed_source_labels(item,source)['confirmed_names'],['Person A'])
        source['s']['profile_name']='Person B'
        self.assertEqual(confirmed_source_labels(item,source)['confirmed_names'],['Person B'])
        self.assertEqual(item['text'],'A website review is requested.')
        with self.assertRaisesRegex(ValueError,'Profile name'):
            validate([{**item,'text':'Person A will review the website.'}],list({'id':i,**v} for i,v in source.items()),['Person A'])

    def test_oversized_and_foreign_citations_rejected(self):
        for item in [{'category':'Topic','text':'x'*601,'segments':['s']},
                     {'category':'Topic','text':'Claim','segments':['missing']}]:
            with self.assertRaises(ValueError):validate([item],[{'id':'s'}])

    def test_shared_service_busy_or_model_drift_fails_before_inference(self):
        c={'summary_url':'http://127.0.0.1:8000/v1/chat/completions','summary_model':'expected','summary_model_path':'/expected'}
        with patch('plaud_workflow.summary_worker.get',return_value={'active_requests':1,'waiting_requests':0}):
            with self.assertRaisesRegex(RuntimeError,'busy'):verify(c)
        with patch('plaud_workflow.summary_worker.get',side_effect=[{'active_requests':0,'waiting_requests':0},{'models':[{'id':'different'}]}]):
            with self.assertRaisesRegex(RuntimeError,'identity'):verify(c)

    def test_zero_only_text_cannot_silently_disappear(self):
        with self.assertRaisesRegex(ValueError,'unaligned'):
            assemble_segments([{'offset':0,'core_start':0,'core_end':10,'segments':[{'words':[
                {'start':1,'end':1,'word':'Unaligned speech'}]}]}],[],10)

    def test_invalid_later_overlap_is_retained_without_fabricating_core_times(self):
        from plaud_workflow.qwen_asr_worker import aligned_words
        valid={'text':'core','start_time':40,'end_time':41}
        tail={'text':'ask','start_time':179.52,'end_time':228.32}
        words,unused=aligned_words([valid,tail],180,165)
        self.assertEqual(words,[{'start':40,'end':41,'word':' core'}])
        self.assertEqual(unused,[tail])
        for bad in [tail,{**tail,'start_time':164},{**tail,'start_time':float('nan')}]:
            with self.assertRaises(ValueError):aligned_words([bad],180,180 if bad is tail else 165)

    def test_decoder_subdivisions_must_be_contiguous_and_inside_the_audio(self):
        from plaud_workflow.qwen_asr_worker import decoding_parts
        parts=[{'start':0,'end':30,'text':''},{'start':30,'end':60,'text':'Speech'}]
        self.assertEqual(decoding_parts({'segments':parts},60),(parts,60))
        # A hole, a decode that does not start at the beginning, nothing at all, and a
        # nonsense boundary are all still refused - none of them can be located.
        for bad in [[parts[1]],[],[{**parts[0],'end':float('nan')}],
                    [parts[0],{'start':35,'end':60,'text':'Speech'}]]:
            with self.assertRaises(ValueError):decoding_parts({'segments':bad},60)

    def test_a_decode_that_stops_early_reports_how_far_it_got(self):
        from plaud_workflow.qwen_asr_worker import decoding_parts
        # Owner rule: audio the models could not read is archived, not dropped.
        # So stopping early is a span to keep, not a reason to lose the chunk - the
        # caller records it. On the 89-minute recording a loop in the first 30-second
        # section ate the token budget and 130 seconds were never read at all.
        parts=[{'start':0,'end':34.53,'text':'.'*1541}]
        self.assertEqual(decoding_parts({'segments':parts},180),(parts,34.53))

    def test_invalid_model_output_cannot_be_checkpointed(self):
        c={'summary_commissioned':True,'summary_url':'local','summary_model':'expected','summary_model_path':'path','summary_config_sha256':'h'}
        source=[{'id':'s','start':0,'end':8,'text':'A website review is requested.'}]
        def run(args,**kwargs):
            job=json.loads(Path(args[-1]).read_text())
            Path(job['output']).write_text(json.dumps([{'category':'Action','text':'Invented','segments':['wrong']}]))
        with tempfile.TemporaryDirectory() as d,patch('plaud_workflow.processing.subprocess.run',side_effect=run):
            with self.assertRaises(ValueError):summarize(c,source,d)
            self.assertFalse((Path(d)/'content-summary/direct.complete.json').exists())

    def test_reprocessing_preserves_source_version_and_removes_stale_summary(self):
        from plaud_workflow.store import Store,Conflict
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d)/'db');store.recording('r','Meeting','hash','source',20)
            store.transcript('r',[{'id':'old','start':0,'end':8,'text':'Old text'}])
            store.set_summary('r',[{'text':'Old claim','segments':['old']}])
            store.transcript('r',[{'id':'new','start':0,'end':8,'text':'Corrected text'}])
            self.assertEqual(store.get('transcript_archive:r:1')['segments'][0]['text'],'Old text')
            self.assertEqual(store.get_recording('r')['summary'],[])
            self.assertEqual(store.pending_publications(),['r'])
            p=store.profile('Known');store.identify('r','new',p,'segment',2,'label',0,5,[1,0])
            with self.assertRaises(Conflict):store.transcript('r',[{'id':'x','start':0,'end':8,'text':'Changed'}])
            self.assertEqual(store.segments('r')[0]['profile_id'],p);store.close()


if __name__=='__main__':unittest.main()

"""Offline ASR and alignment in a bounded, disposable MLX process."""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
import dataclasses
import gc
import json
import math
import sys
import unicodedata
import wave
from collections import Counter
from pathlib import Path

ASR_TOKEN_LIMIT=1536
REPEAT_SHINGLE=8      # characters of cycle to look for in a sub-section
REPEAT_COVERAGE=.5    # of that sub-section's text before it is a decoder loop
DROPPABLE_CORE=2.     # seconds of core speech the worker may give up on its own


class AlignmentBreakdown(ValueError):
    """The aligner placed a word outside the audio it was given.

    Qwen3-ForcedAligner predicts a timestamp class (0-4999, one per 80 ms, so it can
    address 400 seconds) independently of how much audio it was handed, and nothing
    upstream clamps it - QwenLM/Qwen3-ASR#197, "Predicted times are not clamped to
    audio duration". From the first word that lands outside, the library's LIS
    monotonicity repair interpolates every later one, so the alignment has ended
    rather than stumbled: nothing after that word is evidence of anything.
    """
    def __init__(self,index,settled):
        super().__init__('Alignment outside original chunk; raw results retained')
        self.index=index;self.settled=settled


def looping(text):
    """True when one short cycle covers most of the text - a decoder repetition loop.

    Measured over all 906 ASR sub-sections in the library on 2026-09-19: the worst
    healthy one reaches 26.5% coverage and p99 is 11.3%, against 100% for the loop
    that cost a 140-minute recording its transcript. Half is empty space between them.
    """
    if len(text)<REPEAT_SHINGLE*4:return False
    shingles=Counter(text[i:i+REPEAT_SHINGLE] for i in range(len(text)-REPEAT_SHINGLE+1))
    return shingles.most_common(1)[0][1]*REPEAT_SHINGLE>=REPEAT_COVERAGE*len(text)


def realign_tail(model,samples,sample_rate,language,items,broke,duration):
    """Re-align the words the first pass could not place, on the audio still left.

    One retry, same model, no new tuning: handed just the tail, the timestamp
    classifier sees a window it can address. Returns the spliced items - the same
    words in the same order, so restore_punctuation's lexical contract still holds -
    or None when the retry cannot be trusted or the leftover audio cannot hold them.
    """
    import numpy as np
    tail=items[broke.index:];window=duration-broke.settled
    if not tail or window<=0:return None
    speech=samples[round(broke.settled*sample_rate):round(duration*sample_rate)].astype(np.float32)/32768.
    retried=[dataclasses.asdict(w) for w in
             model.generate(speech,text=' '.join(w['text'] for w in tail),language=language)]
    # Never accept a retry that changed the words: the spliced list has to stay the
    # exact lexical sequence restore_punctuation will be asked to match again.
    if [w['text'] for w in retried]!=[w['text'] for w in tail]:return None
    if any(not 0<=float(w['start_time'])<=float(w['end_time'])<=window+.02 for w in retried):return None
    return items[:broke.index]+[{**w,'start_time':float(w['start_time'])+broke.settled,
                                 'end_time':float(w['end_time'])+broke.settled} for w in retried]


def save(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.chmod(0o600);tmp.replace(path)


def restore_punctuation(text,items):
    """Recover display text from ASR without changing alignment words or times.

    Alignment strips punctuation. Only attach the original character spans when
    the complete lexical sequence matches; never guess missing words.
    """
    def normalized(value):
        return ''.join(c for x in value for c in unicodedata.normalize('NFKC',x).casefold() if c.isalnum())
    lexical=[];positions=[]
    for pos,char in enumerate(text):
        chars=normalized(char);lexical.extend(chars);positions.extend([pos]*len(chars))
    lengths=[len(normalized(w['text'])) for w in items]
    if ''.join(lexical)!=''.join(normalized(w['text']) for w in items) or not lexical:
        raise ValueError('ASR and alignment words differ; raw text retained')
    result=[];consumed=0;boundary=0
    for w,n in zip(items,lengths):
        if not n:
            result.append({**w,'display_word':''});continue
        consumed+=n
        if consumed<len(positions) and positions[consumed]==positions[consumed-1]:
            raise ValueError('Alignment splits a source character; raw text retained')
        end=positions[consumed] if consumed<len(positions) else len(text)
        result.append({**w,'display_word':text[boundary:end]});boundary=end
    return result


def aligned_words(items,duration,core_end=None):
    core_end=duration if core_end is None else core_end
    if not 0<=core_end<=duration:raise ValueError('Invalid owned audio interval')
    words=[];settled=0.
    for i,w in enumerate(items):
        a=float(w['start_time']);b=float(w['end_time'])
        if not math.isfinite(a+b) or not 0<=a<=b<=duration+.02:
            # Only later overlap is dispensable: the next core transcribes that
            # audio itself. Keep the exact rejected result as explicit evidence;
            # never clamp an invalid core word onto an invented timestamp. The
            # alignment stops here either way - a word placed past the end means
            # the rest were interpolated from it, not read off the audio.
            if math.isfinite(a+b) and core_end<duration and core_end<=a:
                return words,list(items[i:])
            raise AlignmentBreakdown(i,settled)
        words.append({'start':a,'end':min(b,duration),'word':w.get('display_word',' '+w['text'])})
        settled=min(b,duration)
    return words,[]


def decoding_parts(raw,duration):
    """Check the decoder's subdivisions are contiguous and inside the audio it was given.

    How far they reach is deliberately not checked. A decode that stops early has not
    read the rest, which makes that span unreadable audio - archived, by the owner's
    rule - and not a reason to lose the chunk or the recording.
    Returns the parts and the point the decoder actually reached.
    """
    parts=raw.get('segments',[]);end=0
    for part in parts:
        a=float(part['start']);b=float(part['end'])
        if not math.isfinite(a+b) or abs(a-end)>.001 or not a<b<=duration+1:
            raise ValueError('ASR subdivisions do not cover the original chunk')
        end=b
    if not parts:raise ValueError('ASR produced no subdivisions for the original chunk')
    return parts,end


def run(job):
    import numpy as np
    import mlx.core as mx
    from mlx_audio.stt import load
    with wave.open(job['audio']) as wav:
        sample_rate=wav.getframerate();duration=wav.getnframes()/sample_rate
        samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16)
    if not len(samples) or np.max(np.abs(samples.astype(np.int32)))==0:
        save(job['raw_asr'],{'text':'','reason':'digital silence'})
        save(job['raw_alignment'],{'items':[]})
        save(job['output'],{'language':None,'segments':[]});return
    model=load(job['model'])
    result=model.generate(job['audio'],language=job.get('language'),max_tokens=ASR_TOKEN_LIMIT,
                          batch_size=1,temperature=0,chunk_duration=190,verbose=False)
    raw=dataclasses.asdict(result);save(job['raw_asr'],raw)
    if raw.get('generation_tokens',0)>=ASR_TOKEN_LIMIT:
        # One bounded recovery for long-context repetition, inside the same
        # parent's 180-second wall deadline. Never publish the truncated attempt.
        initial=raw
        result=model.generate(job['audio'],language=job.get('language'),max_tokens=ASR_TOKEN_LIMIT,
                              batch_size=1,temperature=0,chunk_duration=30,verbose=False)
        raw=dataclasses.asdict(result);raw['prior_attempt']=initial
        raw['recovery']='one short-section decode after output limit';save(job['raw_asr'],raw)
    core_end=min(job.get('core_end',duration),duration)
    parts,decoded_to=decoding_parts(raw,duration)
    dropped=[]
    if decoded_to<core_end-.001:
        # The decoder stopped inside the core - on the 89-minute recording a loop in
        # the first 30-second section ate the whole token budget and the remaining 130
        # seconds were never read at all. That audio is kept, not written off.
        dropped.append({'start':decoded_to,'end':core_end,'seconds':core_end-decoded_to,
                        'reason':'decoder stopped before the end of the chunk','text':''})
    text=raw.get('text','').strip()
    lang=raw.get('language')
    if isinstance(lang,list):lang=next((x for x in lang if x and x!='None'),None)
    del model;gc.collect();mx.clear_cache()
    if not text:
        save(job['raw_alignment'],{'items':[]})
        save(job['output'],{'language':lang,'segments':[]});return
    model=load(job['aligner'])
    words=[];unused=[];raw_parts=[];all_items=[]
    # Both raw artefacts exist for every chunk that gets this far, including one whose
    # every section was unreadable and so never reaches the loop below. The chunk's
    # completion manifest hashes them by name; a missing file is a FileNotFoundError
    # at the end of an otherwise finished chunk.
    save(job['raw_alignment'],{'items':[],'parts':[],'language':lang})
    for part in parts:
        if not part['text'].strip():continue
        if looping(part['text']):
            # A runaway section costs the audio it covers, not the chunk and not the
            # recording. Record the span so the gap is visible rather than inferred.
            dropped.append({'start':part['start'],'end':part['end'],'seconds':part['end']-part['start'],
                            'reason':'decoder repetition loop','text':part['text']});continue
        offset=part['start'];end=min(duration,part['end']);part_duration=end-offset
        section=samples[round(offset*sample_rate):round(end*sample_rate)]
        speech=job['audio'] if len(parts)==1 else section.astype(np.float32)/32768.
        part_lang=part.get('language');part_lang=part_lang if part_lang and part_lang!='None' else lang or 'English'
        aligned=model.generate(speech,text=part['text'],language=part_lang)
        items=[dataclasses.asdict(w) for w in aligned]
        raw_parts.append({'offset':offset,'duration':part_duration,'items':items,'language':part_lang})
        all_items.extend({**w,'start_time':w['start_time']+offset,'end_time':w['end_time']+offset} for w in items)
        save(job['raw_alignment'],{'items':all_items,'parts':raw_parts,'language':lang})
        owned_end=min(part_duration,max(0,core_end-offset))
        try:
            part_words,part_unused=aligned_words(restore_punctuation(part['text'],items),part_duration,owned_end)
        except AlignmentBreakdown as broke:
            spliced=realign_tail(model,section,sample_rate,part_lang,items,broke,part_duration)
            if spliced is not None:
                del all_items[len(all_items)-len(items):]
                items=spliced;raw_parts[-1]['items']=items;raw_parts[-1]['retried_tail_from']=broke.settled
                all_items.extend({**w,'start_time':w['start_time']+offset,'end_time':w['end_time']+offset} for w in items)
                save(job['raw_alignment'],{'items':all_items,'parts':raw_parts,'language':lang})
                part_words,part_unused=aligned_words(restore_punctuation(part['text'],items),part_duration,owned_end)
            elif owned_end-broke.settled<DROPPABLE_CORE:
                # The retry had nowhere to put these: the ASR assigned them to this
                # section but they sound in the next one's audio, and the section is
                # already over. Under two seconds - twice the worst boundary error
                # measured across 906 sections - is a smaller loss than the recording.
                restored=restore_punctuation(part['text'],items)
                part_words,_=aligned_words(restored[:broke.index],part_duration)
                part_unused=list(restored[broke.index:])
                dropped.append({'start':offset+broke.settled,'end':offset+owned_end,
                                'seconds':owned_end-broke.settled,'reason':'alignment ended inside the chunk core',
                                'text':''.join(w.get('display_word',w['text']) for w in part_unused)})
            else:raise
        words.extend({**w,'start':w['start']+offset,'end':w['end']+offset} for w in part_words)
        unused.extend({'offset':offset,**w} for w in part_unused)
    if not words or not any(w['end']>w['start'] for w in words):
        # Nothing readable came out of this chunk. That is a gap when we can say which
        # audio it was - the archive keeps that audio, and the neighbouring chunks are
        # unaffected - and a defect when we cannot, because then it failed for a reason
        # nobody recorded.
        if not dropped:raise ValueError('ASR text has no usable alignment; raw text retained as unaligned')
        words=[]
    save(job['output'],{'language':lang,'unused_overlap_alignment':unused,'dropped_audio':dropped,
         'segments':[{'start':0,'end':duration,'text':text,'words':words}] if words else []})


if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))

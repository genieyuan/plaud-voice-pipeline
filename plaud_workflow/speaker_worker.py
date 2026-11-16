"""Pinned local embeddings and per-chunk anonymous speaker turns."""
import json
import sys
import wave
import numpy as np
import sherpa_onnx


def main(job):
    with wave.open(job['audio']) as f:
        assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2)
        audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
    ec=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=job['models']['speaker']['path'],num_threads=4,provider='cpu')
    extractor=sherpa_onnx.SpeakerEmbeddingExtractor(ec)
    def embed(samples):
        if len(samples)<4000 or np.sqrt(np.mean(samples*samples))<1e-5:return None
        stream=extractor.create_stream();stream.accept_waveform(sample_rate=16000,waveform=samples);stream.input_finished()
        if not extractor.is_ready(stream):return None
        vector=np.asarray(extractor.compute(stream),dtype=float)
        if not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-12:return None
        return (vector/np.linalg.norm(vector)).tolist()
    if job['mode']=='embedding':return {'embedding':embed(audio)}
    cfg=sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=job['models']['segmentation']['path']),num_threads=4,provider='cpu'),
        embedding=ec,clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1,threshold=.5),min_duration_on=.3,min_duration_off=.5)
    if not cfg.validate():raise ValueError('Invalid speaker configuration')
    rows=sherpa_onnx.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time()
    turns=[]
    for row in rows:
        start=max(0.,float(row.start));end=min(len(audio)/16000,float(row.end))
        if end>start:
            turns.append({'start':start,'end':end,'local_speaker':int(row.speaker),
                          'embedding':embed(audio[int(start*16000):int(min(end,start+5)*16000)])})
    return {'turns':turns}


if __name__=='__main__':print(json.dumps(main(json.load(open(sys.argv[1]))),allow_nan=False))

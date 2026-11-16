# plaud-voice-pipeline

A local, single-user pipeline that takes recordings from a PLAUD voice recorder account,
transcribes them on your own Mac, works out who spoke when, lets a person name the voices,
and publishes one Markdown note per recording into a notes folder (Obsidian-compatible).
Nothing is sent to a cloud ASR or LLM. Audio, transcripts and voice profiles stay on the machine.

It was built for one household on macOS / Apple silicon. This repository holds only the code,
the test suite and synthetic fixtures. It does not include models, recordings or any data.

## What it does, end to end

Each recording moves through a job with these stages: `download → transcribe → attribute →
suggest → publish → compare`. Every stage is restartable, and the job state lives in SQLite.

1. **Intake (`intake.py`)**. The service polls the PLAUD account through the PLAUD command-line
   client (`plaud me`, `plaud files`, `plaud file <id>`, `plaud audio <id>`), with output parsed
   from the CLI's text format. Before reading anything it checks that the signed-in account is
   the configured `account_id`. The first run records what already exists as a baseline and
   queues only recordings that arrive later. `service backfill` queues the older library on
   purpose. Audio comes from a short-lived signed URL that stays in memory. The download is
   bounded at 4 GiB, verified, and stored content-addressed by SHA-256 with mode 0600.
2. **Transcribe (`processing.py`, `asr_worker.py`, `qwen_asr_worker.py`, `speaker_worker.py`)**.
   The recording is cut into 150-second chunks with 15 seconds of overlap on each side. Every
   chunk runs in a fresh worker process, in its own interpreter, with offline flags and bounded
   threads:
   - ASR with word timestamps: either Whisper, or Qwen3-ASR plus Qwen3-ForcedAligner on MLX
     (`asr_backend: "qwen3"`). The Qwen path restores source punctuation onto aligned words. It
     also detects decoder repetition loops and aligner overshoot, and retries a broken tail
     once.
   - Per-chunk anonymous speaker turns and voice embeddings (sherpa-onnx, pyannote segmentation
     plus a speaker-embedding model).

   Turns are grouped into recording-wide anonymous speakers (`SPEAKER_01`, …) by embedding
   prototypes. Words are then assigned to turns by coverage. Audio the models could not turn
   into timed text is never dropped silently. `archive.py` cuts those spans out of the original
   recording. If `archive_remote` is set, it pushes them over `ssh` to a NAS, checks the SHA-256
   on arrival, and only then removes the local cut.
3. **Attribute: the new speaker-labelling stage (`attribution.py`)**. When
   `speaker_attribution` is `"new_mix"`, this stage replaces the speaker labels (never the
   words) with labels from NVIDIA's Nemotron-3 Diarization model (preset `v3-offline`). The
   model runs locally through a `nemo-speech` binary. The stage works like this:
   - **Pinned runtime.** The runtime home has to match a manifest with SHA-256 pins for the
     binary, its dylibs and the GGUF weights. It also checks exact symlink targets and allows no
     extra files. A dyld load proof confirms that only the runtime's own libraries and system
     libraries load. `runtime-deploy` assembles a home through a staging directory and swaps it
     in by rename, keeping the previous home as a backup.
   - **Isolated run.** The audio is decoded to 16 kHz mono. The binary is launched directly
     (no shell or wrapper), with a minimal environment (no proxy variables), lowered priority
     and a deadline. The process is reaped once and its peak RSS is recorded.
   - **Deterministic alignment to word timings.** Nemotron's turns are validated at integer
     milliseconds, and same-speaker intervals are unioned. Every timed word goes to the speaker
     whose unioned intervals overlap it most. If no interval overlaps, it goes to the nearest
     interval. Ties are broken by first appearance. Adjacent words from the same speaker are
     grouped, and labels are renumbered `SPEAKER_01…` in order of first speech. Segment
     boundaries are always real word edges.
   - **Text-invariant guard.** The concatenated text bytes of the new segments must equal what
     the existing assembly produces from the same chunks. The stage may move speaker
     boundaries but can never add, drop or reorder a character.
   - **Fail-open fallback.** Any problem leaves the existing labels in place, stamps the
     recording `current-fallback` with a reason code, and lets the job continue. Reason codes:
     `asset-invalid`, `decode-failed`, `nemotron-timeout`, `nemotron-exit`, `no-valid-turns`,
     `words-missing`, `untimed-chunk`, `text-invariant`, `named-before-attribution`,
     `version-changed`, `internal-*`.
   - **Atomic commit.** A success replaces the transcript in one SQLite transaction. That
     transaction checks the version, archives the previous revision, and records provenance
     (`speaker_source`, the manifest digest, wall time and peak memory). A recording that
     already carries a human-confirmed name is never re-labelled.

   With the setting absent or `"current"`, the stage only advances the job.
4. **Speaker naming and profiles (`store.py`, `media.py`, `samples.py`, `processing.py`,
   `review.py`)**. Names come only from a person pressing Save in the web UI. A confirmed
   stretch of speech (at most 5 seconds) becomes an enrolment sample, and its embedding is
   added to that person's voice profile. The `suggest` stage compares anonymous speakers in new
   recordings against the profiles and offers likely names above a similarity floor
   (`suggest_floor`). A suggestion never writes a label, and a confirmed segment is never
   re-scored. `review.py` proposes names for unassigned stretches with its evidence and records
   whether the reviewer agreed.
5. **Note publishing (`publication.py`)**. Each recording gets one Markdown note with YAML
   frontmatter in the configured `notes` folder. The pipeline owns only the block between
   `<!-- plaud-workflow:begin -->` and `<!-- plaud-workflow:end -->`. That block holds a link to
   the local timestamped transcript, the confirmed speakers and the transcript revision.
   Anything written outside the block is preserved. If someone edited the managed block or
   deleted the note, publication stops with a conflict instead of overwriting. Publishing goes
   through a retryable outbox, and the file is replaced atomically.
6. **Compare (`compare.py`)**. This optional evaluation stage fetches PLAUD's own transcript
   through the CLI and aligns it with the local one (CJK-aware tokens). It stages agreements
   and differences for a person to judge, and never scores which side is right. It fails open:
   if the reference is missing or unreachable, the event is recorded and the recording is still
   complete. (The divergence classifier it uses is not part of this copy. See *Not included*.)

**Local web app (`web.py`, `app.html`)**. The web app serves the transcript, the audio player,
speaker naming, comparison verdicts and reconciliation. It serves on loopback, or on the
machine's own Tailscale address only when `require_session` is on. Requests must pass Host,
Origin and Sec-Fetch-Site checks. Login uses a single-use ticket minted with a local launcher
key (`launcher.py` installs a per-user LaunchAgent). `labelling.py` / `labelling_words.py` /
`label.html` are blind annotation tools for building evaluation labels.

### Re-labelling existing recordings (`relabel.py`)

The `attribute` stage applies to new recordings. To move recordings that were already
processed onto Nemotron labels, use:

```
python -m plaud_workflow.relabel batch   --config runtime/config.json (--recording RID ... | --limit N) [--out-dir D]
python -m plaud_workflow.relabel undo    --config runtime/config.json --journal D/journal.json --recording RID [...]
python -m plaud_workflow.relabel recover --config runtime/config.json --journal D/journal.json
python -m plaud_workflow.relabel report  --config runtime/config.json --journal D/journal.json [--out F]
```

- **Backup first.** It refuses below 20 GB of free space. It then writes an online SQLite
  backup, a per-table snapshot of every recording with saved names
  (`tools/cr06_snapshot.py`), and a digest of all named rows into a fresh
  `data/backups/relabel/<timestamp>/`.
- **Only unnamed recordings.** Each forward step calls the same `attribution.stage()`, then
  `suggest`, then the single-recording publish. All of this runs under the work lock it shares
  with the running service. A whole-batch time box stops selection between recordings,
  and a lock timeout bounds each wait for the work lock. An optional web probe confirms the UI
  stays up.
- **Stop and reverse.** A failed check, fallback, exception or publish conflict stops the
  selection and reverses every completed recording in reverse order. Each reversal is one
  SQLite transaction, and the note replace is its last act. A recording a person changed
  during the run is left alone and reported.
- **Journal.** The journal is fsynced before each mutation and at each phase, so `recover` can
  finish or reverse an interrupted run after a crash. Running it twice is harmless. `undo`
  reverses chosen recordings later. Reports hold counts, digests and verdicts, never transcript
  text.

`scripts/relabel_rehearsal.py` drives the real CLI end to end on a synthetic root. The root
has generated tokens, a fake Nemotron and a fake ffmpeg. The script injects a failure, runs a
clean batch plus an undo, and SIGKILLs a reversal mid-commit before running `recover` twice.

## Requirements

- macOS on Apple silicon (Metal is the default Nemotron device, and the launcher uses
  `launchd`). The core service is standard-library Python 3.12+ plus numpy. It was developed on
  3.14.
- `ffmpeg` / `ffprobe`.
- The PLAUD command-line client (Node), installed under `runtime/plaud-cli/` and signed in to
  your account.
- Worker interpreters and models, which you provide yourself (see `requirements-workers.txt`):
  - Whisper (`openai-whisper`, `torch`) or Qwen3-ASR + Qwen3-ForcedAligner (`mlx`,
    `mlx-audio`).
  - sherpa-onnx with a pyannote segmentation model and a speaker-embedding model.
  - For the attribute stage: a `nemo-speech` build and the `Nemotron-3-Diarization` q8_0 GGUF.
    Their SHA-256 pins are in `attribution.py`. A different build needs new pins.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.json config.local.json        # gitignored; fill in every <placeholder>
.venv/bin/python -c 'from plaud_workflow.config import initialize; print(initialize())'
.venv/bin/python -m plaud_workflow.service check # validates paths, model hashes, Nemotron runtime
.venv/bin/python -m plaud_workflow.launcher install && .venv/bin/python -m plaud_workflow.launcher open
```

`initialize()` writes `runtime/config.json`, which is the file the service reads. Everything
specific to a household stays in `config.local.json`: the PLAUD account id, the notes folder,
the archive host, and the runtime and model paths and hashes. To turn on Nemotron labels:

```
python -m plaud_workflow.attribution runtime-deploy --source-bin <nemo-speech build/bin> \
    --source-model <Nemotron-3-Diarization.q8_0.gguf> --home runtime/nemotron --backup-dir data/backups/cr06
python -m plaud_workflow.attribution manifest --home runtime/nemotron --config-merge runtime/config.json
# then set "speaker_attribution": "new_mix" in runtime/config.json
```

Other service commands: `python -m plaud_workflow.service {serve,open,status,retry,backfill,archive,compare,check}`.

## Tests

```
.venv/bin/python -m pytest tests -q
```

All fixtures are synthetic: generated tokens, fake executables, temp databases, and
macOS `say` voices for the end-to-end timing test. Tests that need the real Nemotron runtime
skip when it is absent. Point `PLAUD_EVAL_ROOT` at a folder holding
`work/diarization-eval-20261006/runtime` to run them. The browser tests skip without
Playwright.

## Not included

- Models and binaries: Whisper, Qwen3-ASR, the aligner, sherpa-onnx models, `nemo-speech`,
  and the Nemotron GGUF.
- Any data: recordings, transcripts, voice profiles, the SQLite state, notes, evaluation
  sets, and labelling packets. `data/`, `runtime/`, `work/` and `out/` are gitignored.
- The PLAUD CLI itself.
- The evaluation package and divergence classifier that `compare.py` imports lazily
  (`tools_eval_diff`, `evaluation.filters`). Without them the compare stage records the
  comparison as unavailable, and the five comparison-artefact tests in `tests/test_compare.py`
  and `tests/test_recovery.py` fail.
- The design documents and tech specs that the code comments cite (CR-05, CR-06, relabel v5).

## Checking a copy before publishing

`tools/blocklist_scan.py` scans the working tree, every blob and path in git history, and
every commit message. It looks for private terms (from a local, gitignored list), private
network addresses, home-directory paths, e-mail addresses and secret-looking strings. It
exits non-zero on any finding.

## Licence

MIT. See `LICENSE`.

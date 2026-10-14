# Audio Revision Studio

## Setup (once, by whoever hosts it)
1. Install Python 3.10+ and ffmpeg (`sudo apt install ffmpeg` / `brew install ffmpeg`).
2. `pip install -r requirements.txt`
3. Set the key: `export GEMINI_API_KEY=your_key` (or paste it in the sidebar).
4. Run: `streamlit run app.py` -> open the URL shown.
   To host as a website: push this folder to GitHub and deploy on Streamlit Community Cloud
   (`packages.txt` installs ffmpeg; add GEMINI_API_KEY under Secrets/env).

## Manifest format (your builder must write this next to every audio file)
[{"id":0,"text":"full sentence","start_ms":0,"end_ms":5200,"voice":"Charon"}, ...]
Milliseconds, in the order of the final audio.

## Hard-coding the API key
Open app.py, find API_KEY = "" at the top and paste the key between the quotes
(models, default voice and dialect are right below it). Do not upload that file publicly.
Alternatives: env var GEMINI_API_KEY, or GEMINI_API_KEY in Streamlit secrets.

## Getting sentence times (no manifest needed)
Upload the audio plus the script .txt, then pick an alignment method:
- **Whisper (local, recommended)** - transcribes the audio with faster-whisper and matches the words to
  your script, so it tolerates dialect/spelling differences and ASR mistakes. Models: base / small /
  medium / large-v3 (bigger = better but needs more RAM; "small" fits Streamlit Community Cloud).
  The first run downloads the model.
- **Gemini** - the old method (timestamps from an LLM; less reliable on long audio).

### Methods you can compare
| Method | Install | Notes |
|---|---|---|
| Whisper transcribe + match | `pip install faster-whisper` (in requirements.txt) | Transcribes, then matches words to the script |
| Forced align: stable-ts | `pip install stable-ts` (pulls torch) | Forces your exact script onto the audio |
| Forced align: CTC (MMS) | `pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git` (needs torch) | Meta MMS model, language code `ara` |
| Gemini | API key | LLM timestamps, least reliable on long audio |

Run as many as you like on the same file: the **Compare** panel shows time taken, flagged sentences, how much
the methods disagree, and - after you tap-sync 8 "Spot-check" sentences - each method's error against what
you actually heard. Load any result into the table afterwards.
If pip fails to install torch on a very new Python (e.g. 3.14), create a venv with Python 3.11/3.12.

Sentences that could not be located are marked **estimated**. You do NOT have to fix them to continue:
- press **Use this manifest** anyway - flagged sentences are locked for regeneration (you can still
  listen to them), everything else works normally;
- or fix them first with **Tap-sync**: play the audio, press **Start** when a sentence begins and
  **End** when it ends (**B** = end of this one and start of the next when there is no pause).
  Keys: S / E / B / K skip / Z undo / P play. Press Done in the tool, then Apply taps to the table.
  A reaction offset (default 200 ms) is subtracted from your taps and the result is snapped to the
  nearest pause. You can tap only the flagged sentences, or any range.
- the "Re-align from a sentence" box reuses the Whisper transcript (instant).

## Reviser steps
1. Upload the audio plus EITHER a manifest JSON OR the script .txt. With the script: align it
   (see above), optionally fix flagged sentences, press Use this manifest.
2. Press play on the player; the transcript follows the audio. Click the waveform to jump.
3. When you hear a mispronounced word, click it (orange). Double-click replays the sentence.
4. On the right press Queue regeneration (optionally type the correct pronunciation first).
   It runs in the background - keep listening and queue more fixes (up to 3 run at once).
   Sentences with an unfinished fix show ⏳, accepted ones show ✎.
   Sentences with unverified timing are locked unless you tick 'I listened and the timing is right'.
5. In the Fix queue, each finished fix shows original vs regenerated. Accept, Again (optionally
   with a new correction) or discard. Undo is in the sidebar.
6. Press Submit revisions -> final edit. Select a region on the waveform and press Delete (or
   Keep only selection), zoom with the wheel, Undo/Redo, then press Apply edits.
7. Prepare download for the final audio + updated manifest.

## Folders
app.py, engine.py, player_component/, editor_component/, tapsync_component/ (new), requirements.txt, packages.txt

## Verify audio page (new)
Open **Verify audio** in the sidebar menu of the site (file: `pages/1_Verify_audio.py`, engine: `verify_engine.py`).
Upload the final audio and the script .txt, press **Run verification**:
- Whisper listens to the audio (the script is NOT given to it, so it cannot guess words that were never spoken).
- Every script word is compared with what was heard: **red = not heard**, **yellow = heard differently**
  (the heard word is shown small next to it), blue = heard but not in the script.
- Numbers: words in script, exact %, present % (exact + changed), missing / changed words, and sentence counts
  (fully heard / changed / words missing / whole sentence missing).
- Pick any problem sentence to listen to it with 1.5 s of context.
- **Generate the missing sentences**: Gemini TTS creates the sentences that are missing entirely, matched to the
  neighbouring tone and loudness; each one is re-checked with Whisper before you tick it for insertion, and is
  inserted at the gap where it belongs. Then run verification again to confirm.
- Downloads: audio, manifest JSON (same format as the main editor plus verification fields) and a CSV report.
Limits: Whisper itself can drop or mis-hear words (dialect, names, numbers written as digits), so listen to red/yellow
spots before regenerating. Sentences with only some words missing or changed are fixed in the main editor.

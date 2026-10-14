import copy, difflib, io, json, os, re, statistics, sys, threading, time, types as _pytypes
import numpy as np
import librosa
from pydub import AudioSegment
from pydub import effects
from pydub.silence import detect_leading_silence, detect_silence, detect_nonsilent
from google import genai
from google.genai import types

VOICES = ["Charon", "Puck", "Kore", "Zephyr", "Fenrir", "Leda", "Orus", "Aoede",
          "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
          "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
          "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
          "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"]


def client(key):
    return genai.Client(api_key=key)


def to_np(seg):
    x = np.array(seg.set_channels(1).get_array_of_samples(), dtype=np.float32)
    return x / float(1 << (8 * seg.sample_width - 1))


def wav_bytes(seg):
    b = io.BytesIO()
    seg.export(b, format="wav")
    return b.getvalue()


def load_manifest(raw):
    data = json.loads(raw)
    segs = data["segments"] if isinstance(data, dict) else data
    for i, s in enumerate(segs):
        s.setdefault("id", i)
    return segs


def find_segments(manifest, a_ms, b_ms):
    hits = [(min(s["end_ms"], b_ms) - max(s["start_ms"], a_ms), s) for s in manifest]
    return [s for ov, s in sorted(hits, key=lambda h: -h[0]) if ov > 0]


def _json(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def _parts(r):
    cand = r.candidates[0] if r.candidates else None
    content = getattr(cand, "content", None)
    return list(content.parts) if content is not None and content.parts else []


def _why(r):
    cand = r.candidates[0] if r.candidates else None
    fb = getattr(r, "prompt_feedback", None)
    return (f"finish_reason={getattr(cand, 'finish_reason', None)}, "
            f"block_reason={getattr(fb, 'block_reason', None)}")


def _retry(fn, n=3):
    err = None
    for k in range(n):
        try:
            return fn()
        except RuntimeError as ex:
            err = ex
            time.sleep(1.5 * (k + 1))
    raise err


def _gen_text(c, model, contents):
    r = c.models.generate_content(model=model, contents=contents)
    t = "".join(p.text for p in _parts(r) if getattr(p, "text", None))
    if not t:
        raise RuntimeError(f"Gemini returned no text ({_why(r)})")
    return t


def _gen_audio(c, model, voice, prompt, temperature=1.0):
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    r = c.models.generate_content(model=model, config=cfg, contents=prompt)
    for p in _parts(r):
        d = getattr(getattr(p, "inline_data", None), "data", None)
        if d:
            return d
    raise RuntimeError(f"TTS returned no audio ({_why(r)}). Try again, another voice/model, "
                       f"or a shorter style prompt.")


def diagnose(c, model, clip, text, dialect, word=None, wi=None, correction=None):
    """Return {'text': rewritten sentence, ...}. With `word` only that word is fixed;
    without it the whole sentence is respelled phonetically."""
    if word:
        rule = (f"Rewrite the sentence so a TTS engine pronounces it exactly as a native "
                f"{dialect} speaker would say it. Change only the problem word (spelling / "
                f"diacritics); keep every other word identical.")
        task = (f'The reviewer heard the word "{word}" (word number {wi + 1} of the sentence) '
                f"pronounced wrongly in this audio. ")
        task += (f"It should be pronounced: {correction}. " if correction else
                 "Decide how to respell it so the TTS says it correctly. ")
        shape = '{"word": "the word you changed", "text": "full sentence"}'
    else:
        rule = (f"Rewrite the WHOLE sentence phonetically, as close as possible to how a native "
                f"{dialect} speaker would actually say it, so a TTS engine reproduces that "
                f"pronunciation. Use spelling and diacritics that match the spoken sound. Keep the "
                f"meaning and word order; words already pronounced correctly may stay unchanged.")
        task = ("Listen to the audio of this sentence. One or more words may be mispronounced - "
                "find ALL of them. ")
        if correction:
            task += f"The reviewer's note on the correct pronunciation: {correction}. "
        shape = '{"words": ["every word you changed"], "text": "full rewritten sentence"}'
    prompt = f"Sentence: {text}\n{task}{rule}\nReturn JSON only: {shape}"
    part = types.Part.from_bytes(data=wav_bytes(clip), mime_type="audio/wav")
    return _json(_retry(lambda: _gen_text(c, model, [part, prompt])))


def _frames_db(x, sr):
    h = max(1, int(sr * 0.01))
    n = len(x) // h
    f = x[:n * h].reshape(n, h)
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9), h


def _noise_like(x, sr):
    """Broadband hiss / click (flat spectrum, high centroid) instead of speech."""
    seg = x[:int(sr * 0.1)]
    if len(seg) < 64:
        return False
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    flat = float(np.exp(np.mean(np.log(sp))) / np.mean(sp))
    return flat > 0.3 or float((fr * sp).sum() / sp.sum()) > 3500


def _tail_cut(x, sr, mode):
    """Sample index where a trailing burst (noise/click after a pause) begins; len(x) if none."""
    max_run, min_gap, any_short = (0.7, 0.12, False) if mode == "normal" else (1.0, 0.06, True)
    db, h = _frames_db(x, sr)
    if len(db) < 20:
        return len(x)
    ref = np.percentile(db, 90)
    idx = np.flatnonzero(db > max(ref - 30, -50))
    if len(idx) < 2:
        return len(x)
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > 8:                      # merge pauses shorter than 80 ms
            runs.append((s, p))
            s = i
        p = i
    runs.append((s, p))
    if len(runs) < 2:
        return len(x)
    (s, e), prev_end = runs[-1], runs[-2][1]
    length, gap = (e - s + 1) * 0.01, (s - prev_end) * 0.01
    odd = _noise_like(x[s * h:], sr) or db[s:e + 1].mean() > ref + 1 or (any_short and length <= 0.5)
    if length <= max_run and gap >= min_gap and odd:
        return min(prev_end + 1 + 4, s) * h  # keep 40 ms after the last real sound
    return len(x)


def clean_edges(seg, mode="normal"):
    """Remove the noise burst Gemini TTS can add after the speech (and a click before it).
    mode: 'off' | 'normal' | 'aggressive'."""
    if mode == "off" or len(seg) < 800:
        return seg
    x, sr = to_np(seg), seg.frame_rate
    end = _tail_cut(x, sr, mode)
    x2 = x[:end]
    start = len(x2) - _tail_cut(x2[::-1].copy(), sr, mode)
    out = seg[int(start / sr * 1000): int(end / sr * 1000)]
    if start > 0:
        out = out.fade_in(10)
    if end < len(x):
        out = out.fade_out(30)
    return out


def _trim(seg, keep=80):
    s = detect_leading_silence(seg, -45)
    e = detect_leading_silence(seg.reverse(), -45)
    return seg[max(0, s - keep): len(seg) - max(0, e - keep)]


def tts(c, model, voice, text, style, like, temperature=1.0, tail="normal"):
    prompt = (f"Say the following transcript aloud exactly as written, in this style: {style}. Speak at a clear, full, consistent volume.\n\n"
              f"TRANSCRIPT:\n{text}") if style else f"TRANSCRIPT:\n{text}"
    pcm = _retry(lambda: _gen_audio(c, model, voice, prompt, temperature))
    seg = AudioSegment(data=pcm, sample_width=2, frame_rate=24000, channels=1)
    seg = _trim(clean_edges(seg, tail)).set_frame_rate(like.frame_rate).set_channels(like.channels)
    return seg.set_sample_width(like.sample_width)


def speech_db(seg):
    """Loudness of the speech only (pauses ignored)."""
    try:
        ch = detect_nonsilent(seg, min_silence_len=120, silence_thresh=seg.dBFS - 18)
        if ch:
            return sum((seg[a:b] for a, b in ch), AudioSegment.empty()).dBFS
    except Exception:
        pass
    return seg.dBFS


def _soft_limit(x, knee=0.6, ceil=0.98):
    ax = np.abs(x)
    y = x.copy()
    o = ax > knee
    y[o] = np.sign(x[o]) * (knee + (ceil - knee) * np.tanh((ax[o] - knee) / (ceil - knee)))
    return y


def match_loudness(seg, target_db, boost=0.0):
    """Raise the speech level to target_db (+boost). Peaks are soft-limited instead of the
    whole clip being turned down, so the boost really takes effect."""
    seg = seg.set_sample_width(2)
    x = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    goal = target_db + boost
    gain = goal - speech_db(seg)
    out = seg
    for _ in range(5):
        y = _soft_limit(x * 10 ** (min(gain, 40.0) / 20))
        out = seg._spawn((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
        err = goal - speech_db(out)
        if abs(err) < 0.3:
            break
        gain += err
    return out


def _stats(seg, text):
    x, sr = to_np(seg), seg.frame_rate
    f0 = librosa.pyin(x, fmin=60, fmax=450, sr=sr)[0]
    f0 = f0[~np.isnan(f0)]
    return dict(f0=float(np.median(f0)) if len(f0) else np.nan, sdb=speech_db(seg),
                cps=len(text.replace(" ", "")) / max(seg.duration_seconds, 0.1))


def global_speech_db(audio, n=8, chunk=8000):
    """Typical speech level of the whole recording (sampled), used as a safe reference."""
    step = max(len(audio) // n, chunk)
    parts = [audio[i:i + chunk] for i in range(0, max(len(audio) - chunk, 1), step)][:n]
    return speech_db(sum(parts[1:], parts[0])) if parts else speech_db(audio)


def compare(new, text, refs, fallback_db=None):
    """Compare the new clip with reference clips [(seg, text)]. References that are too short
    (wrong time slots) are ignored; loudness is kept near the recording's overall level."""
    good = [(g, t) for g, t in refs if len(g) >= 700]
    s = _stats(new, text)
    if good:
        r = [_stats(g, t) for g, t in good]
        f0r = np.nanmedian([x["f0"] for x in r])
        pitch = abs(12 * np.log2(s["f0"] / f0r)) if not np.isnan(s["f0"] * f0r) else 0.0
        pace = abs(np.log2(s["cps"] / np.median([x["cps"] for x in r])))
        target = float(np.median([x["sdb"] for x in r]))
    else:
        pitch = pace = 0.0
        target = fallback_db if fallback_db is not None else s["sdb"]
    if fallback_db is not None:
        target = float(np.clip(target, fallback_db - 4, fallback_db + 4))
    return dict(pitch_semitones=round(pitch, 2), pace_log2=round(pace, 2), target_db=target,
                score=pitch / 2 + pace / 0.25, ok=pitch <= 2 and pace <= 0.25)


def neighbours(audio, manifest, seg):
    i = manifest.index(seg)
    return [(audio[m["start_ms"]:m["end_ms"]], m["text"])
            for m in manifest[max(0, i - 2): i + 3] if m is not seg]


def regenerate_refs(c, model, voice, text, style, like, refs, tries=1, boost=0.0, temperature=1.0,
                    fallback_db=None, tail="normal"):
    best = None
    for _ in range(tries):
        new = tts(c, model, voice, text, style, like, temperature, tail)
        m = compare(new, text, refs, fallback_db)
        new = match_loudness(new, m["target_db"], boost)
        if best is None or m["score"] < best[1]["score"]:
            best = (new, m)
        if m["ok"]:
            break
    return best


def regenerate(c, model, voice, text, style, audio, manifest, seg, tries=1, boost=0.0, temperature=1.0):
    like = audio[seg["start_ms"]:seg["end_ms"]]
    refs = neighbours(audio, manifest, seg) + [(like, seg["text"])]
    return regenerate_refs(c, model, voice, text, style, like, refs, tries, boost, temperature,
                           global_speech_db(audio))


def run_job(job):
    """Runs in a background thread; only mutates the job dict."""
    try:
        c = client(job["key"])
        job["status"] = "rewriting"
        d = diagnose(c, job["text_model"], job["clip"], job["base_text"], job["dialect"],
                     job["word"], job["wi"], job["corr"])
        job["status"] = "generating"
        new, m = regenerate_refs(c, job["tts_model"], job["voice"], d["text"], job["style"],
                                 job["like"], job["refs"], 1, job["boost"], job["temperature"],
                                 job["fallback_db"], job.get("tail", "normal"))
        job.update(new=new, m=m, new_text=d["text"], new_wav=wav_bytes(new),
                   old_wav=wav_bytes(job["clip"]), status="ready")
    except Exception as ex:
        job.update(status="failed", error=str(ex))


def splice(audio, manifest, seg, new, text, fade=15):
    """Replace the segment's audio; everything after it moves by the length difference."""
    a, b = seg["start_ms"], seg["end_ms"]
    new = new.fade_in(fade).fade_out(fade)
    out = audio[:a] + new + audio[b:]
    delta = len(new) - (b - a)
    mf = copy.deepcopy(manifest)
    k = next(i for i, m in enumerate(mf) if m["id"] == seg["id"])
    mf[k]["end_ms"], mf[k]["tts_text"] = b + delta, text
    mf[k]["rev"] = mf[k].get("rev", 0) + 1
    for m in mf[k + 1:]:
        m["start_ms"] += delta
        m["end_ms"] += delta
    return out, mf


def cut(audio, manifest, a, b):
    """Delete [a, b) ms from the audio and keep the manifest in sync."""
    a, b = max(0, int(a)), min(len(audio), int(b))
    if b <= a:
        return audio, manifest
    d, mf = b - a, []
    for m in copy.deepcopy(manifest):
        s, e = m["start_ms"], m["end_ms"]
        if e > a:
            if s >= b:
                m["start_ms"], m["end_ms"] = s - d, e - d
            else:
                ns, ne = (s if s < a else a), (e - d if e >= b else a)
                if ne <= ns:
                    continue
                m["start_ms"], m["end_ms"] = ns, ne
        mf.append(m)
    return audio[:a] + audio[b:], mf


def bad_slots(rows):
    """Sentence numbers whose time slot is too short for their text (wrong alignment)."""
    out = []
    for r in rows:
        d = (r["end_ms"] - r["start_ms"]) / 1000
        if d < 0.15 or len(r["text"].replace(" ", "")) / d > 40:
            out.append(r["id"] + 1)
    return out


def normalize(rows, total, check_slots=True):
    """Sort by id, make sentences contiguous. Returns (rows, ids_with_problems).
    check_slots=False leaves out the "too little time for the text" check (those rows are
    flagged with flagged() instead of blocking the user)."""
    rows = copy.deepcopy(sorted(rows, key=lambda r: r["id"]))
    probs = []
    for r in rows:
        r["start_ms"], r["end_ms"] = int(r["start_ms"]), int(r["end_ms"])
        if r["end_ms"] <= r["start_ms"]:
            probs.append(r["id"] + 1)
    for p, r in zip(rows, rows[1:]):
        if r["start_ms"] < p["end_ms"] - 50:
            probs.append(r["id"] + 1)
        bnd = (p["end_ms"] + r["start_ms"]) // 2 if r["start_ms"] < p["end_ms"] else r["start_ms"]
        p["end_ms"] = r["start_ms"] = bnd
    rows[0]["start_ms"], rows[-1]["end_ms"] = 0, total
    if check_slots:
        probs += bad_slots(rows)
    return rows, sorted(set(probs))


def split_script(text):
    parts = re.split(r"(?<=[.!?؟…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def build_manifest(c, model, audio, sentences, win_s=240, batch=40, progress=None):
    """Align script sentences to audio. Returns (rows, warning_or_None).
    Sentences that cannot be located are skipped, given estimated times and marked
    status='estimated' so the user can fix them by hand."""
    total = len(audio)
    sil = detect_silence(audio, min_silence_len=250, silence_thresh=audio.dBFS - 16)
    mids = [(a + b) // 2 for a, b in sil]

    def snap(ms):
        if not mids:
            return int(ms)
        m = min(mids, key=lambda x: abs(x - ms))
        return int(m if abs(m - ms) <= 1200 else ms)

    def report(done, pos):
        if progress:
            progress(min(done / n, 1.0), f"Aligned {done}/{n} sentences "
                     f"({100 * done // n}%) - audio position {pos / 1000:.0f}s of {total / 1000:.0f}s")

    n = len(sentences)
    found, pos, idx, fails, stopped = {}, 0, 0, 0, False
    report(0, 0)
    while idx < n and pos < total - 500:
        got, start = [], 0
        for attempt in range(2):  # 2nd try: look further back, wider window
            start = max(0, pos - (4000 if attempt == 0 else 10000)) if pos else 0
            clip = audio[start: start + int(win_s * (1 + 0.5 * attempt)) * 1000]
            listing = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences[idx: idx + batch]))
            prompt = ("The audio is a recording of a script. The clip starts at 0 seconds and may "
                      "begin with a few words from an earlier sentence - ignore those. Below are "
                      "the next script sentences in order (the speech may differ slightly in "
                      "dialect, match by meaning).\nFor each sentence that is FULLY spoken in this "
                      "clip, give its start and end time in seconds (decimal). Stop at the first "
                      "sentence that is not fully inside the clip.\n" + listing +
                      '\nReturn JSON only: {"items":[{"i":0,"start":0.0,"end":4.2}]}')
            part = types.Part.from_bytes(data=wav_bytes(clip), mime_type="audio/wav")
            try:
                items = sorted(_json(_retry(lambda: _gen_text(c, model, [part, prompt])))["items"],
                               key=lambda d: d["i"])
            except Exception:
                items = []
            got, prev_end, clip_s = [], 0.0, len(clip) / 1000
            for k, it in enumerate(items):
                try:
                    s0, e0 = float(it["start"]), float(it["end"])
                    chars = len(sentences[idx + k].replace(" ", ""))
                    ok = (it["i"] == k and 0 <= s0 < e0 <= clip_s + 0.5 and s0 >= prev_end - 0.5
                          and 2.0 <= chars / (e0 - s0) <= 40.0)  # plausible speaking speed
                except Exception:
                    ok = False
                if not ok:
                    break
                prev_end = e0
                got.append(it)
            if got:
                break
        if not got:  # skip this sentence, keep going (stop after 3 misses in a row)
            fails += 1
            idx += 1
            report(idx, pos)
            if fails >= 3:
                stopped = True
                break
            continue
        fails = 0
        for it in got:
            found[idx + it["i"]] = (start + float(it["start"]) * 1000, start + float(it["end"]) * 1000)
        idx += len(got)
        end_ms = found[idx - 1][1]
        pos = snap(end_ms) if snap(end_ms) > pos else int(end_ms)
        report(idx, pos)

    rows = _rows_from_found(found, sentences, total, snap)
    return rows, _missing_msg(found, n, stopped)


def _missing_msg(found, n, stopped=False):
    missing = [k + 1 for k in range(n) if k not in found]
    if not missing:
        return None
    shown = ", ".join(map(str, missing[:15])) + (" ..." if len(missing) > 15 else "")
    return (f"Aligned {len(found)} of {n} sentences. Not located: {shown}"
            + (" (stopped after 3 misses in a row)." if stopped else ".")
            + " Rows marked 'estimated' are rough guesses and stay locked for regeneration - "
              "fix them with the tap-sync tool or by hand.")


def _rows_from_found(found, sentences, total, snap):
    """found: {sentence_index: (start_ms, end_ms)} for located sentences. Boundaries are snapped
    between located neighbours; text-length estimates fill the gaps (status='estimated')."""
    n = len(sentences)
    b, i = [0], 0
    while i < n:
        if i in found:
            if i + 1 == n:
                b.append(total)
            elif i + 1 in found:
                b.append(snap((found[i][1] + found[i + 1][0]) / 2))
            else:
                b.append(snap(found[i][1]))
            i += 1
        else:
            j = i
            while j < n and j not in found:
                j += 1
            run_end = total if j == n else snap(found[j][0])
            w = [max(len(s), 1) for s in sentences[i:j]]
            acc = 0
            for x in w:
                acc += x
                b.append(int(b[i] + (run_end - b[i]) * acc / sum(w)))
            i = j
    for k in range(1, len(b)):
        b[k] = max(b[k], b[k - 1] + 1)
    return [dict(id=k, text=sentences[k], start_ms=int(b[k]), end_ms=int(b[k + 1]),
                 status="ok" if k in found else "estimated") for k in range(n)]


def _snapper(audio, tol=1200):
    """Returns snap(ms): moves a time to the middle of the nearest pause (within tol ms)."""
    sil = detect_silence(audio, min_silence_len=250, silence_thresh=audio.dBFS - 16)
    mids = [(a + b) // 2 for a, b in sil]

    def snap(ms):
        if not mids:
            return int(ms)
        m = min(mids, key=lambda x: abs(x - ms))
        return int(m if abs(m - ms) <= tol else ms)
    return snap


# ---------------------------------------------------------------- Whisper alignment
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
                         "ؤ": "و", "ئ": "ي", "گ": "ك", "چ": "ج", "پ": "ب", "ڤ": "ف"})
_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_PUNCT = re.compile(r"[^\w\s]", re.U)
_WH = {}


def _norm(w):
    w = _DIAC.sub("", w).translate(_AR_MAP)
    return _PUNCT.sub("", w).lower().strip()


def _whisper_store():
    """Process-wide store that survives Streamlit reruns AND reloads of this module."""
    m = sys.modules.get("_rev_whisper_store")
    if m is None:
        m = _pytypes.ModuleType("_rev_whisper_store")
        m.models, m.lock = {}, threading.Lock()
        sys.modules["_rev_whisper_store"] = m
    return m


def get_whisper(model_size, progress=None):
    """Load a Whisper model once per process. Looks in the local cache first (no network, no
    download); only downloads when the files are not on disk yet. Optional env var
    WHISPER_MODEL_DIR sets the cache folder (use a path with plain English letters)."""
    store = _whisper_store()
    with store.lock:
        if model_size in store.models:
            return store.models[model_size]
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper is not installed. Add `faster-whisper` to "
                               "requirements.txt and restart the app.")
        kw = dict(device="cpu", compute_type="int8", download_root=os.getenv("WHISPER_MODEL_DIR") or None)
        if progress:
            progress(0.0, f"Loading the Whisper '{model_size}' model from disk…")
        try:
            model = WhisperModel(model_size, local_files_only=True, **kw)
        except Exception:
            if progress:
                progress(0.0, f"Downloading the Whisper '{model_size}' model (needed only once)…")
            model = WhisperModel(model_size, **kw)
        store.models[model_size] = model
        return model


def whisper_words(audio, model_size="small", language="ar", prompt=None, progress=None):
    """Transcribe with faster-whisper. Returns [(word, start_ms, end_ms)]."""
    model = get_whisper(model_size, progress)
    x = to_np(audio.set_frame_rate(16000))
    segs, info = model.transcribe(
        x, language=language or None, word_timestamps=True, vad_filter=True, beam_size=5,
        condition_on_previous_text=False, initial_prompt=prompt)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append((w.word.strip(), int(w.start * 1000), int(w.end * 1000)))
        if progress:
            progress(min(s.end / max(info.duration, 1e-6), 1.0),
                     f"Transcribed {s.end:.0f}s of {info.duration:.0f}s")
    return words


def align_words(sentences, words, block=120):
    """Match script words to recognised words (order-preserving, tolerant of ASR mistakes and
    spelling differences). Returns {sentence_index: (start_ms, end_ms)} for located sentences."""
    toks = [[t for t in (_norm(w) for w in s.split()) if t] for s in sentences]
    A = [(_norm(w), a, b) for w, a, b in words if _norm(w)]
    n, total_tok = len(sentences), sum(len(t) for t in toks)
    if not A or not total_tok:
        return {}
    tok_a = [x[0] for x in A]
    ratio = len(A) / total_tok
    match, si, j = {}, 0, 0
    while si < n and j < len(A):
        blk, owner, sj = [], [], si
        while sj < n and len(blk) < block:
            for p, t in enumerate(toks[sj]):
                blk.append(t)
                owner.append((sj, p))
            sj += 1
        if blk:
            lo, hi = max(0, j - 15), min(len(A), j + 3 * len(blk) + 30)
            sm = difflib.SequenceMatcher(None, blk, tok_a[lo:hi], autojunk=False)
            ops = [o for o in sm.get_opcodes() if o[0] == "equal" or
                   (o[0] == "replace" and o[2] - o[1] == o[4] - o[3] <= 4)]
            solid = [o for o in ops if o[0] == "equal" and o[2] - o[1] >= 3]
            if solid:
                # single stray words far from any solid run are noise (common words like
                # "في" match anywhere) - keep only matches near/between solid runs
                first_b, last_b = lo + solid[0][3], lo + solid[-1][4]
                keep = [o for o in ops if o in solid or (first_b - 12 <= lo + o[3] and lo + o[4] <= last_b + 12)]
                for tag, a0, a1, b0, b1 in keep:
                    for d in range(a1 - a0):
                        match[owner[a0 + d]] = lo + b0 + d
                j = last_b if last_b > j else j
            else:
                j += int(len(blk) * ratio)
        si = sj
    glob = (A[-1][2] - A[0][1]) / len(A)  # average ms per word, pauses included
    found, prev_end = {}, 0
    for k, ts in enumerate(toks):
        pts = sorted((p, match[(k, p)]) for p in range(len(ts)) if (k, p) in match)
        if not pts or len(pts) < min(2, len(ts)) or len(pts) / len(ts) < 0.35:
            continue
        (p0, a0), (p1, a1) = pts[0], pts[-1]
        t0, t1 = A[a0][1], A[a1][2]
        per = (t1 - t0) / (p1 - p0 + 1) if p1 > p0 else glob
        per = min(max(per, 0.4 * glob), 2 * glob)
        s0 = max(t0 - p0 * per, prev_end)
        e0 = t1 + (len(ts) - 1 - p1) * per
        if e0 <= s0:
            continue
        found[k] = (int(s0), int(e0))
        prev_end = e0
    return found


def build_manifest_whisper(audio, sentences, words):
    """Whisper path. Returns (rows, warning_or_None) like build_manifest()."""
    found = align_words(sentences, words)
    rows = _rows_from_found(found, sentences, len(audio), _snapper(audio))
    return rows, _missing_msg(found, len(sentences))


# ---------------------------------------------------------------- tap-sync
def flagged(rows):
    """ids of sentences whose timing cannot be trusted (estimated / too little time)."""
    bad = {i - 1 for i in bad_slots(rows)}
    return sorted(r["id"] for r in rows
                  if r.get("status") in ("estimated", "unverified")
                  or (r["id"] in bad and r.get("status") not in ("tapped", "edited")))


def apply_taps(rows, taps, audio, tol=500):
    """taps: {sentence_id: {"s": start_ms, "e": end_ms}} (either may be missing).
    A boundary between sentence k-1 and k is the mean of the 'end' tap of k-1 and the 'start'
    tap of k, then snapped to the middle of a nearby pause."""
    rows = copy.deepcopy(sorted(rows, key=lambda r: r["id"]))
    pos = {r["id"]: k for k, r in enumerate(rows)}
    cand = {}
    for sid, t in taps.items():
        k = pos.get(int(sid))
        if k is None:
            continue
        if t.get("s") is not None:
            cand.setdefault(k, []).append(float(t["s"]))
        if t.get("e") is not None:
            cand.setdefault(k + 1, []).append(float(t["e"]))
        rows[k]["status"] = "tapped"
    snap = _snapper(audio, tol)
    for k, v in cand.items():
        bnd = snap(sum(v) / len(v))
        if k < len(rows):
            rows[k]["start_ms"] = bnd
        if k > 0:
            rows[k - 1]["end_ms"] = bnd
    return rows

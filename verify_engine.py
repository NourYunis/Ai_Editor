"""verify_engine.py - checks that an audio file really contains every word of its script.

Pipeline: Whisper transcription (NO script hints, so it cannot "hear" words that were never
spoken) -> locate each sentence in time -> word-by-word comparison inside each sentence ->
report (ok / changed / missing words, extras) -> optional generation + insertion of missing
sentences. Reuses a few helpers from engine.py; nothing here changes engine.py's behaviour.
"""
import bisect
import difflib
import html

import engine as E

OK, CHANGED, MISSING, PUNCT = "ok", "changed", "missing", "punct"
ICON = {"ok": "✅", "changed": "🟡", "partial": "🟠", "missing": "🔴"}
LABEL = {"ok": "all words heard", "changed": "some words heard differently",
         "partial": "some words missing", "missing": "sentence missing"}


# ---------------------------------------------------------------- transcription
def transcribe(audio, model_size="small", language="ar", vad=False, progress=None):
    """Plain Whisper transcription with word times -> [(word, start_ms, end_ms)].
    No initial prompt (it would bias Whisper towards the script and hide real mistakes)."""
    model = E.get_whisper(model_size, progress)
    x = E.to_np(audio.set_frame_rate(16000))
    segs, info = model.transcribe(
        x, language=language or None, word_timestamps=True, vad_filter=vad, beam_size=5,
        condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append((w.word.strip(), int(w.start * 1000), int(w.end * 1000)))
        if progress:
            progress(min(s.end / max(info.duration, 1e-6), 1.0),
                     f"Transcribed {s.end:.0f}s of {info.duration:.0f}s")
    return words


# ---------------------------------------------------------------- word-level diff
_SIM = {}


def _sim(a, b):
    r = _SIM.get((a, b))
    if r is None:
        r = _SIM[(a, b)] = difflib.SequenceMatcher(None, a, b).ratio()
    return r


def diff_words(script, heard, gap=0.7, min_sim=0.34):
    """Edit alignment of two lists of normalised words.
    Returns ops: ("ok"|"changed", i, j) / ("missing", i, None) / ("extra", None, j).
    Words that are similar enough count as 'changed'; unrelated ones as missing + extra."""
    n, m = len(script), len(heard)
    INF = 1e9

    def sub(i, j):
        a, b = script[i], heard[j]
        if a == b:
            return 0.0
        r = _sim(a, b)
        return 2 * (1 - r) if r >= min_sim else INF

    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + sub(i - 1, j - 1), D[i - 1][j] + gap, D[i][j - 1] + gap)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = sub(i - 1, j - 1)
            if s < INF and abs(D[i][j] - (D[i - 1][j - 1] + s)) < 1e-9:
                ops.append((OK if s == 0 else CHANGED, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(D[i][j] - (D[i - 1][j] + gap)) < 1e-9:
            ops.append((MISSING, i - 1, None))
            i -= 1
        else:
            ops.append(("extra", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------- verification
def verify(sentences, words, total_ms, margin=400):
    """Compare the script with the transcript. Returns a result dict:
    sentences: [{id,text,start_ms,end_ms,status,tokens:[{w,tag,heard}],extras:[str],counts}]
    totals: word / sentence counts and percentages; rows: contiguous time windows; found."""
    toks = [[(w, E._norm(w)) for w in s.split()] for s in sentences]
    heard = [(E._norm(w), a, b, w) for w, a, b in words if E._norm(w)]
    mids = [(h[1] + h[2]) / 2 for h in heard]
    found = E.align_words(sentences, words)
    rows = E._rows_from_found(found, sentences, total_ms, lambda ms: int(ms))
    used, hp, out = set(), 0, []
    for k, row in enumerate(rows):
        j0 = max(hp, bisect.bisect_left(mids, row["start_ms"] - margin))
        j1 = bisect.bisect_right(mids, row["end_ms"] + margin)
        cand = list(range(j0, max(j0, j1)))
        sc = [t for _, t in toks[k] if t]
        ops = diff_words(sc, [heard[j][0] for j in cand])
        tag_of, hit = {}, []
        for tag, i, j in ops:
            if tag in (OK, CHANGED):
                tag_of[i] = (tag, heard[cand[j]][3])
                used.add(cand[j])
                hp = max(hp, cand[j] + 1)
                hit.append(cand[j])
            elif tag == MISSING:
                tag_of[i] = (MISSING, None)
        tokens, i = [], 0
        for w, t in toks[k]:
            if not t:
                tokens.append(dict(w=w, tag=PUNCT, heard=None))
            else:
                tag, h = tag_of.get(i, (MISSING, None))
                tokens.append(dict(w=w, tag=tag, heard=h))
                i += 1
        cnt = {OK: 0, CHANGED: 0, MISSING: 0}
        for t in tokens:
            if t["tag"] in cnt:
                cnt[t["tag"]] += 1
        n = sum(cnt.values())
        present = cnt[OK] + cnt[CHANGED]
        if n == 0 or cnt[OK] == n:
            status = "ok"
        elif present / n < 0.35:
            status = "missing"
        elif cnt[MISSING]:
            status = "partial"
        else:
            status = "changed"
        out.append(dict(id=k, text=sentences[k], status=status, tokens=tokens, extras=[], counts=cnt,
                        start_ms=heard[min(hit)][1] if hit else row["start_ms"],
                        end_ms=heard[max(hit)][2] if hit else row["end_ms"]))
    starts = [r["start_ms"] for r in rows]
    n_extra = 0
    for j, h in enumerate(heard):          # heard words that belong to no script word
        if j not in used:
            n_extra += 1
            k = max(bisect.bisect_right(starts, mids[j]) - 1, 0)
            out[k]["extras"].append(h[3])
    tot = {OK: 0, CHANGED: 0, MISSING: 0}
    for s in out:
        for c in tot:
            tot[c] += s["counts"][c]
    nw = sum(tot.values()) or 1
    st = [s["status"] for s in out]
    totals = dict(words=sum(tot.values()), ok=tot[OK], changed=tot[CHANGED], missing=tot[MISSING],
                  extra=n_extra, exact_pct=100 * tot[OK] / nw,
                  present_pct=100 * (tot[OK] + tot[CHANGED]) / nw,
                  sentences=len(out), s_ok=st.count("ok"), s_changed=st.count("changed"),
                  s_partial=st.count("partial"), s_missing=st.count("missing"))
    return dict(sentences=out, totals=totals, rows=rows, found=found)


def check_clip(clip, text, model_size="small", language="ar", vad=False):
    """Verify one generated clip against its sentence. Returns the sentence dict."""
    return verify([text], transcribe(clip, model_size, language, vad), len(clip))["sentences"][0]


# ---------------------------------------------------------------- reports
def mmss(ms):
    s = int(ms / 1000)
    return f"{s // 60}:{s % 60:02d}"


def render_html(result, only_problems=False, show_extras=True):
    """Script with red (missing) and yellow (changed) highlights, as one HTML string."""
    p = ['<div style="font-size:19px;line-height:2.2">']
    for s in result["sentences"]:
        if only_problems and s["status"] == "ok":
            continue
        p.append('<div dir="auto" style="margin:5px 0;padding:4px 10px;border:1px solid #8886;border-radius:8px">'
                 f'<span style="font:12px sans-serif;opacity:.75;direction:ltr;display:inline-block;'
                 f'margin:0 8px">{ICON[s["status"]]} #{s["id"] + 1} · {mmss(s["start_ms"])}</span>')
        for t in s["tokens"]:
            w = html.escape(t["w"])
            if t["tag"] == MISSING:
                p.append(f'<span title="not heard" style="background:#ff6b6b;color:#000;padding:1px 4px;'
                         f'border-radius:4px">{w}</span> ')
            elif t["tag"] == CHANGED:
                p.append(f'<span title="heard: {html.escape(t["heard"])}" style="background:#ffd43b;color:#000;'
                         f'padding:1px 4px;border-radius:4px">{w}</span>'
                         f'<sup style="font:11px sans-serif;direction:ltr;opacity:.8"> {html.escape(t["heard"])}</sup> ')
            else:
                p.append(f"{w} ")
        if show_extras and s["extras"]:
            p.append('<span style="color:#4dabf7;font-style:italic;font-size:15px" title="heard in the audio but '
                     f'not in the script">⟨{html.escape(" ".join(s["extras"]))}⟩</span>')
        p.append("</div>")
    p.append("</div>")
    return "".join(p)


def issues_table(result):
    """List of dicts (one per problem sentence) for a table / CSV."""
    rows = []
    for s in result["sentences"]:
        if s["status"] == "ok" and not s["extras"]:
            continue
        rows.append({"Sentence": s["id"] + 1, "Time": mmss(s["start_ms"]), "Status": LABEL[s["status"]],
                     "Missing words": " ".join(t["w"] for t in s["tokens"] if t["tag"] == MISSING),
                     "Changed (script → heard)": "  |  ".join(f'{t["w"]} → {t["heard"]}' for t in s["tokens"]
                                                            if t["tag"] == CHANGED),
                     "Extra words heard": " ".join(s["extras"]), "Text": s["text"]})
    return rows


def manifest(result, total_ms, snap=None):
    """Manifest in the format of the main app (+ verification fields)."""
    rows = E._rows_from_found(result["found"], [s["text"] for s in result["sentences"]], total_ms,
                              snap or (lambda ms: int(ms)))
    out = []
    for r, s in zip(rows, result["sentences"]):
        r = dict(r)
        r["verify"] = s["status"]
        r["missing_words"] = [t["w"] for t in s["tokens"] if t["tag"] == MISSING]
        r["changed_words"] = [dict(script=t["w"], heard=t["heard"]) for t in s["tokens"] if t["tag"] == CHANGED]
        out.append(r)
    return out


# ---------------------------------------------------------------- generate + insert missing
def make_refs(audio, result, k, span=2):
    """Neighbouring good sentences (clip, text) used to match tone and loudness."""
    refs = []
    for m in range(max(0, k - span), min(len(result["rows"]), k + span + 1)):
        s = result["sentences"][m]
        if m != k and s["status"] in ("ok", "changed"):
            r = result["rows"][m]
            refs.append((audio[r["start_ms"]:r["end_ms"]], s["text"]))
    return refs


def generate_missing(c, model, voice, style, audio, result, k, boost=0.0, temperature=1.0,
                     tail="normal", tries=2, fallback_db=None):
    """TTS for sentence k, matched to its neighbours. Returns (clip, match_info)."""
    refs = make_refs(audio, result, k)
    return E.regenerate_refs(c, model, voice, result["sentences"][k]["text"], style, audio[:1000], refs,
                             tries, boost, temperature,
                             fallback_db if fallback_db is not None else E.global_speech_db(audio), tail)


def insert_sentences(audio, result, clips, snap, fade=15):
    """clips: {sentence_id: clip}. Consecutive ids go in as one block, at the gap where the
    sentence belongs (end of the previous heard sentence, snapped to a pause). Blocks are
    inserted from the end backwards so earlier positions stay valid. Returns the new audio."""
    ids = sorted(clips)
    runs = []
    for k in ids:
        if runs and runs[-1][-1] == k - 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    for run in reversed(runs):
        first = result["rows"][run[0]]["start_ms"]
        pos = 0 if run[0] == 0 else min(max(int(snap(first)), 0), len(audio))
        block = clips[run[0]].fade_in(fade).fade_out(fade)
        for k in run[1:]:
            block = block + clips[k].fade_in(fade).fade_out(fade)
        audio = audio[:pos] + block + audio[pos:]
    return audio

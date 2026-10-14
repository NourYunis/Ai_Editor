import base64, importlib.util, io, json, os, re
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
import engine as E

# ======================= CONFIG =======================
API_KEY = ""   # <- paste your Gemini API key between the quotes to hard-code it.
#                 Never upload this file to a public place with the key inside.
TEXT_MODEL = "gemini-3.1-pro-preview"
TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "Charon"
DIALECT = "Egyptian Arabic"
# ======================================================

st.set_page_config(page_title="Audio Revision Studio", layout="wide")
S = st.session_state
HERE = os.path.dirname(os.path.abspath(__file__))

def check_component(folder, marker, label):
    path = os.path.join(HERE, folder, "index.html")
    if not os.path.isfile(path):
        st.error(f"Missing file: {folder}/index.html. Create the folder '{folder}' next to "
                 f"app.py and put the {label} index.html inside it.")
        st.stop()
    if marker not in open(path, encoding="utf-8").read():
        st.error(f"{folder}/index.html is the wrong file (it looks like the other component's). "
                 f"Put the {label} index.html in '{folder}'.")
        st.stop()


check_component("player_component", 'id="follow"', "PLAYER")
check_component("editor_component", 'id="bApply"', "EDITOR")
check_component("tapsync_component", 'id="bStart"', "TAP-SYNC")
player = components.declare_component("rev_player", path=os.path.join(HERE, "player_component"))
editor = components.declare_component("rev_editor", path=os.path.join(HERE, "editor_component"))
tapper = components.declare_component("rev_tapsync", path=os.path.join(HERE, "tapsync_component"))
S.setdefault("tap_run", 0)


def secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def b64(seg, fmt, **kw):
    buf = io.BytesIO()
    seg.export(buf, format=fmt, **kw)
    mime = "audio/mpeg" if fmt == "mp3" else "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def ensure_b64():
    """Light mp3 copy of the current audio for the in-browser components."""
    if S.get("b64_ver") != S.ver:
        S.b64 = b64(S.audio.set_channels(1), "mp3", bitrate="96k")
        S.b64_ver = S.ver


# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    key = API_KEY or secret_key() or os.getenv("GEMINI_API_KEY", "")
    if key:
        st.success("API key loaded")
    else:
        key = st.text_input("Gemini API key", type="password")
    text_model = st.text_input("Text model (analysis / rewrite)", TEXT_MODEL)
    tts_model = st.text_input("TTS model", TTS_MODEL)
    v = st.selectbox("Voice", E.VOICES + ["Other…"],
                     index=E.VOICES.index(DEFAULT_VOICE) if DEFAULT_VOICE in E.VOICES else 0)
    voice = st.text_input("Custom voice name", "") if v == "Other…" else v
    style = st.text_area("Style prompt", "Speak in natural Egyptian Arabic, calm and clear, "
                         "same pace and tone as the surrounding narration.")
    dialect = st.text_input("Dialect", DIALECT)
    temperature = st.slider("Temperature (TTS)", 0.0, 2.0, 1.0, 0.05,
                            help="Same setting as Temperature in Google AI Studio, sent to the TTS model. "
                                 "Lower = steadier delivery, higher = more varied.")
    tail_ui = st.selectbox("Remove noise at the end of generated audio", ["Normal", "Aggressive", "Off"],
                           help="Cuts the short burst of noise Gemini TTS can add after the speech.")
    boost = st.slider("Extra volume for regenerated audio (dB)", -6, 12, 0,
                      help="Applied by this app after generation (not a Google AI Studio setting). The clip is "
                           "already auto-matched to the surrounding speech level; use this only to fine-tune.")
    if st.button("↩ Undo last change", disabled=not S.get("history")):
        S.audio, S.manifest = S.history.pop()
        S.cand, S.ver = None, S.ver + 1
        st.rerun()

st.title("🎧 Audio Revision Studio")

# ---------------- Upload ----------------
c1, c2 = st.columns(2)
af = c1.file_uploader("1. Audio file", type=["mp3", "wav", "m4a", "ogg"])
mf = c2.file_uploader("2a. Manifest JSON (if you have one)", type=["json"])
sf = c2.file_uploader("2b. …or the script text file (.txt)", type=["txt"])
if not af:
    st.info("Upload the audio to begin.")
    st.stop()

sig = (af.name, af.size, mf and mf.size, sf and sf.size)
if S.get("sig") != sig:
    S.sig, S.history, S.cand, S.out, S.draft, S.step = sig, [], None, None, None, "revise"
    S.jobs, S.job_seq = [], 0
    S.fmt = af.name.rsplit(".", 1)[-1].lower()
    S.audio = AudioSegment.from_file(af)
    S.ver = S.get("ver", 0) + 1
    S.manifest, S.mwarn = None, None
    if mf:
        S.manifest, probs = E.normalize(E.load_manifest(mf.getvalue()), len(S.audio))
        S.mwarn = f"Overlapping or reversed times in sentences: {probs}" if probs else None

if S.manifest is None:
    if not sf and not S.get("draft"):
        st.info("Upload a manifest JSON or the script text file.")
        st.stop()
    st.subheader("Sentence timings")
    if sf:
        sentences = E.split_script(sf.getvalue().decode("utf-8"))
        st.caption(f"{len(sentences)} sentences found (split by line breaks and . ! ? ؟)")
        has_wh = importlib.util.find_spec("faster_whisper") is not None
        method = st.radio("Alignment method", ["Whisper (local, recommended)", "Gemini"],
                          index=0 if has_wh else 1, horizontal=True)
        if method.startswith("Whisper"):
            w1, w2 = st.columns(2)
            wh_size = w1.selectbox("Whisper model", ["base", "small", "medium", "large-v3"], index=1,
                                   help="Bigger = more accurate but slower and needs more RAM. 'small' fits "
                                        "Streamlit Community Cloud; use 'medium'/'large-v3' on your own machine. "
                                        "It is downloaded only the first time, then loaded from disk.")
            wh_lang = w2.text_input("Audio language code", "ar")
            if not has_wh:
                st.warning("faster-whisper is not installed. Run `pip install faster-whisper` "
                           "(it is already listed in requirements.txt) and restart.")
            if st.button("⚙ Align with Whisper", disabled=not has_wh):
                bar = st.progress(0.0, text="Preparing the Whisper model…")
                try:
                    S.wwords = E.whisper_words(S.audio, wh_size, wh_lang.strip() or None,
                                               prompt=" ".join(sentences[:2])[:200],
                                               progress=lambda f, t: bar.progress(f, text=t))
                    S.draft, S.draft_msg = E.build_manifest_whisper(S.audio, sentences, S.wwords)
                    bar.empty()
                except Exception as ex:
                    st.error(f"Whisper alignment failed: {ex}")
        else:
            if not key:
                st.warning("No API key yet. Paste it in the sidebar (then press Enter) or put it in "
                           "API_KEY at the top of app.py. The button unlocks once a key is set.")
            if st.button("⚙ Align script to audio (Gemini)", disabled=not key):
                bar = st.progress(0.0, text="Starting…")
                try:
                    S.draft, S.draft_msg = E.build_manifest(E.client(key), text_model, S.audio, sentences,
                                                            progress=lambda f, t: bar.progress(f, text=t))
                except Exception as ex:
                    st.error(f"Alignment failed: {ex}")
    if S.get("draft"):
        if S.get("draft_msg"):
            st.warning(S.draft_msg)
        st.write("Check the times (ms). Rows marked **estimated** are guesses. You do **not** have to fix "
                 "them to continue: they stay locked for regeneration, and every other sentence works "
                 "normally. To fix them quickly, use the tap-sync tool below instead of typing milliseconds.")
        try:
            ed = st.data_editor(S.draft, num_rows="fixed", width="stretch", disabled=["id"])
        except TypeError:  # older Streamlit
            ed = st.data_editor(S.draft, num_rows="fixed", use_container_width=True, disabled=["id"])
        orig = {r["id"]: (r["start_ms"], r["end_ms"]) for r in S.draft}
        ed_rows = [dict(r) for r in ed]
        for r in ed_rows:  # a row the user changed by hand counts as checked
            if r.get("status") in ("estimated", "unverified") and (r["start_ms"], r["end_ms"]) != orig.get(r["id"]):
                r["status"] = "edited"
        fl = E.flagged(ed_rows)
        st.caption(f"{len(fl)} sentence(s) flagged" + (f": {[i + 1 for i in fl[:25]]}" if fl else ""))

        with st.expander("🖱 Tap-sync: mark sentence times by listening (no millisecond typing)", expanded=bool(fl)):
            st.caption("Play the audio. Press **Start** the moment a sentence begins and **End** when it ends "
                       "(or **B** when the next one follows with no pause). Keys: S / E / B / K skip / Z undo / "
                       "P play. Then press *Done* in the tool and *Apply taps* here.")
            mode = st.radio("Sentences to tap", [f"Flagged only ({len(fl)})", "A range of sentences"],
                            horizontal=True, key="tap_mode")
            if mode.startswith("Flagged"):
                ids = set(fl)
            else:
                r1, r2 = st.columns(2)
                a_ = int(r1.number_input("From sentence #", 1, len(ed_rows), 1, key="tap_from"))
                b_ = int(r2.number_input("To sentence #", 1, len(ed_rows), min(10, len(ed_rows)), key="tap_to"))
                ids = set(range(min(a_, b_) - 1, max(a_, b_)))
            targets = [dict(id=r["id"], text=r["text"], start_ms=int(r["start_ms"]), end_ms=int(r["end_ms"]))
                       for r in ed_rows if r["id"] in ids]
            if not targets:
                st.info("No sentences selected.")
            else:
                ensure_b64()
                val = tapper(audio=S.b64, ver=S.ver, run=S.tap_run, targets=targets, key="tapper", default=None)
                got = val["taps"] if val and val.get("run") == S.tap_run else {}
                st.caption(f"{len(got)} sentence(s) received from the tap tool.")
                if st.button("⬇ Apply taps to the table", disabled=not got):
                    S.draft = E.apply_taps(ed_rows, got, S.audio)
                    S.draft_msg = None
                    S.tap_run += 1
                    st.rerun()

        with st.expander("Re-align the rest of the audio from a sentence"):
            n_from = int(st.number_input("From sentence #", 1, len(ed), 1, key="nfrom"))
            t_from = st.number_input("That sentence really starts at (seconds)", 0.0,
                                     len(S.audio) / 1000, int(ed[n_from - 1]["start_ms"]) / 1000,
                                     key=f"tfrom{n_from}")
            can_realign = bool(S.get("wwords")) or bool(key)
            if st.button("🔄 Re-align from here", disabled=not can_realign):
                n0, t0 = n_from - 1, int(t_from * 1000)
                texts = [r["text"] for r in ed][n0:]
                bar2 = st.progress(0.0, text="Starting…")
                try:
                    if S.get("wwords"):  # reuse the transcript: instant
                        words = [(w, a - t0, b - t0) for w, a, b in S.wwords if a >= t0]
                        new, msg = E.build_manifest_whisper(S.audio[t0:], texts, words)
                    else:
                        new, msg = E.build_manifest(E.client(key), text_model, S.audio[t0:], texts,
                                                    progress=lambda f, t: bar2.progress(f, text=t))
                    for r in new:
                        r["id"] += n0
                        r["start_ms"] += t0
                        r["end_ms"] += t0
                    keep = [dict(r) for r in ed_rows][:n0]
                    if keep:
                        keep[-1]["end_ms"] = t0
                    S.draft, S.draft_msg = keep + new, msg
                    st.rerun()
                except Exception as ex:
                    st.error(f"Re-alignment failed: {ex}")

        if fl:
            st.info(f"{len(fl)} sentence(s) are still flagged. You can continue anyway: those stay locked "
                    "(you can still listen to them), and you can regenerate all the others. You can come "
                    "back later with “✏ Edit sentence times”.")
        if st.button("✅ Use this manifest"):
            rows, probs = E.normalize(
                [dict(id=int(r["id"]), text=r["text"], start_ms=r["start_ms"], end_ms=r["end_ms"],
                      status=r.get("status") or "ok") for r in ed_rows], len(S.audio), check_slots=False)
            if probs:
                st.error(f"Fix the times of these sentences first (overlap / wrong order): {probs}")
            else:
                bad = set(E.flagged(rows))
                for r in rows:
                    if r["id"] in bad:
                        r["status"] = "unverified"
                S.manifest = rows
                st.rerun()
    st.stop()

audio, manifest = S.audio, S.manifest

# =================== STEP 2: final edit ===================
if S.step == "final":
    st.subheader("Final audio — cut & delete")
    st.caption("Drag on the waveform to select · Delete key or 🗑 removes it · wheel = zoom · "
               "Space = play/pause · press **Apply edits** when done.")
    if S.get("ed_ver") != S.ver:
        S.ed64 = b64(audio.set_channels(1).set_frame_rate(16000).set_sample_width(2), "wav")
        S.ed_ver = S.ver
    val = editor(audio=S.ed64, ver=S.ver, key="editor", default=None)
    ops = val["ops"] if val and val.get("ver") == S.ver else []
    final, fman = audio, manifest
    for a, b in ops:
        final, fman = E.cut(final, fman, a, b)
    st.write(f"Length: {len(audio) / 1000:.1f}s → **{len(final) / 1000:.1f}s** "
             f"({len(ops)} edit(s) applied)")
    okey = (S.ver, json.dumps(ops))
    if st.button("📦 Prepare download", type="primary"):
        buf = io.BytesIO()
        final.export(buf, format=S.fmt)
        S.out, S.out_key = buf.getvalue(), okey
    if S.get("out") and S.get("out_key") == okey:
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Final audio", S.out, f"final.{S.fmt}")
        d2.download_button("⬇ Updated manifest", json.dumps(fman, ensure_ascii=False, indent=1),
                           "manifest_final.json")
    if st.button("← Back to revising"):
        S.step = "revise"
        st.rerun()
    st.stop()

# =================== STEP 1: revise ===================
flag = set(E.flagged(manifest))
if flag:
    st.warning(f"{len(flag)} sentence(s) have unverified timing, so their highlight may be off: "
               f"{sorted(i + 1 for i in flag)[:25]}. They are locked for regeneration; everything else works "
               "normally. To fix them: ✏ Edit sentence times → Tap-sync.")
with st.expander("Sentence times / timeline check", expanded=False):
    st.caption(f"{len(manifest)} sentences · audio {len(audio) / 1000:.1f}s · last sentence starts at "
               f"{manifest[-1]['start_ms'] / 1000:.1f}s")
    if st.button("✏ Edit sentence times"):
        S.draft = [dict(id=m["id"], text=m["text"], start_ms=m["start_ms"], end_ms=m["end_ms"],
                        status=m.get("status", "ok")) for m in manifest]
        S.draft_msg, S.manifest = None, None
        st.rerun()
if S.get("mwarn"):
    st.warning(S.mwarn + ". Replacing sentences may land in the wrong place until this is fixed.")

ensure_b64()  # light mp3 copy for the in-browser player

ACTIVE = ("queued", "rewriting", "generating")
LABEL = {"queued": "waiting", "rewriting": "rewriting the sentence", "generating": "generating audio"}


@st.cache_resource
def executor():
    return ThreadPoolExecutor(max_workers=3)


def enqueue(seg, word, wi, corr):
    """Start a regeneration in the background and return immediately."""
    like = S.audio[seg["start_ms"]:seg["end_ms"]]
    S.job_seq += 1
    job = dict(id=S.job_seq, sid=seg["id"], word=word, wi=wi, corr=corr, status="queued",
               clip=like, like=like, refs=E.neighbours(S.audio, S.manifest, seg) + [(like, seg["text"])],
               base_text=seg.get("tts_text", seg["text"]), rev=seg.get("rev", 0), key=key,
               text_model=text_model, tts_model=tts_model, voice=voice, style=style,
               dialect=dialect, temperature=temperature, boost=boost,
               fallback_db=E.global_speech_db(S.audio), tail=tail_ui.lower())
    S.jobs.append(job)
    executor().submit(E.run_job, job)


def requeue(j, corr):
    seg = next(m for m in S.manifest if m["id"] == j["sid"])
    j["status"] = "discarded"
    enqueue(seg, j["word"], j["wi"], corr)


def accept(j):
    seg = next(m for m in S.manifest if m["id"] == j["sid"])
    S.history.append((S.audio, S.manifest))
    S.audio, S.manifest = E.splice(S.audio, S.manifest, seg, j["new"], j["new_text"])
    j["status"] = "accepted"
    S.out = None
    S.ver += 1


def jobs_panel():
    jobs = S.get("jobs", [])
    running = sum(j["status"] in ACTIVE for j in jobs)
    st.subheader("Fix queue" + (f" — {running} running" if running else ""))
    shown = [j for j in jobs if j["status"] not in ("accepted", "discarded")]
    if not shown:
        st.caption("Queued fixes appear here. Keep listening while they generate, "
                   "then come back to accept each one.")
    for j in reversed(shown):
        tag = f"Sentence #{j['sid'] + 1}" + (f" · «{j['word']}»" if j["word"] else "")
        with st.container(border=True):
            if j["status"] in ACTIVE:
                st.markdown(f"⏳ **{tag}** — {LABEL[j['status']]}…")
            elif j["status"] == "failed":
                st.markdown(f"❌ **{tag}**")
                st.error(j["error"])
                f1, f2 = st.columns(2)
                if f1.button("Retry", key=f"r{j['id']}"):
                    requeue(j, j["corr"]); st.rerun()
                if f2.button("Dismiss", key=f"x{j['id']}"):
                    j["status"] = "discarded"; st.rerun(scope="fragment")
            else:  # ready
                st.markdown(f"✅ **{tag}** — ready")
                m = j["m"]
                (st.caption if m["ok"] else st.warning)(
                    f"Tone {'matches' if m['ok'] else 'is off'}: pitch Δ {m['pitch_semitones']} st, "
                    f"pace Δ {m['pace_log2']}")
                p1, p2 = st.columns(2)
                p1.caption("Original"); p1.audio(j["old_wav"], format="audio/wav")
                p2.caption("Regenerated"); p2.audio(j["new_wav"], format="audio/wav")
                cur = next((x for x in S.manifest if x["id"] == j["sid"]), None)
                if cur and cur.get("rev", 0) != j["rev"]:
                    st.warning("This sentence was changed after this job started. "
                               "Accepting replaces that change.")
                nc = st.text_input("Correct pronunciation (optional)", value=j["corr"] or "",
                                   key=f"c{j['id']}")
                k1, k2, k3 = st.columns(3)
                if k1.button("✅ Accept", key=f"a{j['id']}"):
                    accept(j); st.rerun()
                if k2.button("🔁 Again", key=f"g{j['id']}"):
                    requeue(j, nc.strip() or None); st.rerun()
                if k3.button("🗑", key=f"d{j['id']}"):
                    j["status"] = "discarded"; st.rerun(scope="fragment")


unresolved = {j["sid"] for j in S.jobs if j["status"] in ACTIVE + ("ready",)}
left, right = st.columns([3, 2], gap="large")
with left:
    sel = player(audio=S.b64, ver=S.ver, dur_ms=len(audio), key="player", default=None,
                 segments=[dict(id=m["id"], text=m["text"], start_ms=m["start_ms"],
                                end_ms=m["end_ms"], chg=bool(m.get("tts_text")),
                                pend=m["id"] in unresolved, unv=m["id"] in flag) for m in manifest])

with right:
    st.subheader("Fix a mistake")
    live = sel and sel.get("n", 0) > S.get("cleared_n", 0)
    seg = next((m for m in manifest if m["id"] == sel["sid"]), None) if live else None
    if not seg:
        st.info("Keep listening. When you hear a mispronounced word, **click it** in the "
                "transcript, then queue the fix here. You can queue several while the audio plays.")
    else:
        word = re.sub(r'''[\s.,!?؟،:;"'()«»\[\]-]+''', "", sel.get("word") or "") or None
        st.markdown(f"**Sentence #{seg['id'] + 1}**" + (f" — word: **{word}**" if word else
                    " — whole sentence (Gemini finds every mispronounced word and respells the sentence phonetically)"))
        corr = st.text_input("Correct pronunciation (optional — leave empty to let Gemini decide)")
        allow = True
        if seg["id"] in flag:
            st.warning("This sentence's timing is unverified: the audio highlighted for it may belong to "
                       "another sentence, and replacing it could damage the wrong part.")
            allow = st.checkbox("I listened and the timing is right — allow anyway", key=f"allow{seg['id']}")
        if st.button("➕ Queue regeneration", type="primary", disabled=not key or not allow):
            enqueue(seg, word, sel.get("wi"), corr.strip() or None)
            S.cleared_n = sel["n"]
            st.rerun()
    st.divider()
    any_active = any(j["status"] in ACTIVE for j in S.jobs)
    st.fragment(run_every=2 if any_active else None)(jobs_panel)()

st.divider()
left_over = sum(j["status"] in ACTIVE + ("ready",) for j in S.jobs)
if left_over:
    st.warning(f"{left_over} fix(es) are still running or waiting for your decision in the queue.")
if st.button("✅ Submit revisions → final edit", type="primary"):
    S.step = "final"
    st.rerun()

"""Verify that an audio file contains every word of its script (shown as a page of the site)."""
import io, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st
from pydub import AudioSegment
import engine as E
import verify_engine as V

st.set_page_config(page_title="Verify audio", layout="wide")
S = st.session_state
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_const(name, default):
    """Read a constant (API_KEY, TTS_MODEL, ...) from app.py so it is configured in one place."""
    try:
        m = re.search(rf'^{name}\s*=\s*"([^"]*)"', open(os.path.join(HERE, "app.py"), encoding="utf-8").read(), re.M)
        return m.group(1) if m and m.group(1) else default
    except Exception:
        return default


def secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Verification settings")
    wh_size = st.selectbox("Whisper model", ["base", "small", "medium", "large-v3"], index=1,
                           help="Used only to listen to the audio. Bigger = more accurate but slower. "
                                "The script is NOT given to it, so it cannot guess missing words.")
    lang = st.text_input("Audio language code", "ar")
    vad = st.checkbox("Skip silences (VAD)", value=False,
                      help="Off is safer for verification: VAD can drop quiet speech and cause false 'missing'.")
    st.divider()
    st.subheader("Generating missing sentences")
    key = app_const("API_KEY", "") or secret_key() or os.getenv("GEMINI_API_KEY", "")
    if key:
        st.success("API key loaded")
    else:
        key = st.text_input("Gemini API key", type="password")
    tts_model = st.text_input("TTS model", app_const("TTS_MODEL", "gemini-2.5-pro-preview-tts"))
    v = st.selectbox("Voice", E.VOICES + ["Other…"],
                     index=E.VOICES.index(app_const("DEFAULT_VOICE", "Charon"))
                     if app_const("DEFAULT_VOICE", "Charon") in E.VOICES else 0)
    voice = st.text_input("Custom voice name", "") if v == "Other…" else v
    style = st.text_area("Style prompt", "Speak in natural Egyptian Arabic, calm and clear, "
                         "same pace and tone as the surrounding narration.")
    temperature = st.slider("Temperature (TTS)", 0.0, 2.0, 1.0, 0.05)
    boost = st.slider("Extra volume (dB)", -6, 12, 0)
    tail_ui = st.selectbox("Remove noise at the end of generated audio", ["Normal", "Aggressive", "Off"])

st.title("Verify audio against the script")
st.caption("Checks that every word of the script is spoken in the audio (after cutting, generating and "
           "stitching). 🔴 red = word not heard · 🟡 yellow = heard differently (shown small next to it) · "
           "blue ⟨…⟩ = heard but not in the script.")

c1, c2 = st.columns(2)
a_up = c1.file_uploader("Audio", type=["mp3", "wav", "m4a", "ogg", "flac", "aac"])
t_up = c2.file_uploader("Script (.txt)", type=["txt"])
if not (a_up and t_up):
    st.info("Upload the audio and the script text file.")
    st.stop()

sig = (a_up.name, a_up.size, t_up.name, t_up.size)
if S.get("v_sig") != sig:
    try:
        S.v_audio = AudioSegment.from_file(io.BytesIO(a_up.getvalue()))
    except Exception as ex:
        st.error(f"Could not read the audio: {ex}")
        st.stop()
    S.v_sig, S.v_ver = sig, 0
    S.v_words = S.v_wsig = S.v_result = None
    S.v_gen, S.v_stale, S.v_dl = {}, False, {}
sentences = E.split_script(t_up.getvalue().decode("utf-8"))
audio = S.v_audio
st.caption(f"{len(sentences)} sentences in the script · audio length {V.mmss(len(audio))}")

if st.button("▶ Run verification", type="primary"):
    wsig = (S.v_sig, S.v_ver, wh_size, lang, vad)
    try:
        if S.get("v_wsig") != wsig:  # transcription is the slow part: reuse it when nothing changed
            bar = st.progress(0.0, text="Preparing the Whisper model…")
            S.v_words = V.transcribe(audio, wh_size, lang.strip() or None, vad,
                                     progress=lambda f, t: bar.progress(f, text=t))
            S.v_wsig = wsig
            bar.empty()
        S.v_result = V.verify(sentences, S.v_words, len(audio))
        S.v_gen, S.v_stale, S.v_dl = {}, False, {}
    except Exception as ex:
        st.error(f"Verification failed: {ex}")

R = S.get("v_result")
if not R:
    st.stop()
if S.get("v_stale"):
    st.warning("The audio changed after this check (sentences were inserted). Press **Run verification** "
               "again to re-check the new audio.")

# ---------------- summary ----------------
T = R["totals"]
m = st.columns(5)
m[0].metric("Words in script", T["words"])
m[1].metric("Exact match", f"{T['exact_pct']:.1f}%")
m[2].metric("Present (exact + changed)", f"{T['present_pct']:.1f}%")
m[3].metric("🔴 Missing words", T["missing"])
m[4].metric("🟡 Changed words", T["changed"])
s = st.columns(6)
s[0].metric("Sentences", T["sentences"])
s[1].metric("✅ Fully heard", T["s_ok"])
s[2].metric("🟡 With changed words", T["s_changed"])
s[3].metric("🟠 With missing words", T["s_partial"])
s[4].metric("🔴 Missing sentences", T["s_missing"])
s[5].metric("Extra words heard", T["extra"])
if not (T["missing"] or T["changed"]):
    st.success("Every word of the script was heard in the audio.")
else:
    st.warning("Some words were not heard or were heard differently. Whisper can also drop or mis-hear words "
               "(especially in dialect), so listen to the flagged places before regenerating.")

# ---------------- highlighted script ----------------
mode = st.radio("Show", ["Only problems", "All sentences"], horizontal=True)
with st.container(height=520):
    st.markdown(V.render_html(R, only_problems=mode == "Only problems"), unsafe_allow_html=True)

issues = [x for x in R["sentences"] if x["status"] != "ok"]
if issues:
    lab = [f"{V.ICON[x['status']]} #{x['id'] + 1} · {V.mmss(x['start_ms'])} · {x['text'][:50]}" for x in issues]
    i = st.selectbox("Listen to a problem sentence", range(len(issues)), format_func=lambda k: lab[k])
    x = issues[i]
    a0, b0 = max(0, x["start_ms"] - 1500), min(len(audio), x["end_ms"] + 1500)
    st.audio(E.wav_bytes(audio[a0:b0]), format="audio/wav")
    miss_w = " ".join(t["w"] for t in x["tokens"] if t["tag"] == "missing")
    chg_w = ", ".join(f"{t['w']} → {t['heard']}" for t in x["tokens"] if t["tag"] == "changed")
    st.caption(f"Playing {a0 / 1000:.1f}s – {b0 / 1000:.1f}s (1.5 s of context on each side)."
               + (f" Missing: {miss_w}." if miss_w else "") + (f" Changed: {chg_w}." if chg_w else ""))

# ---------------- generate missing sentences ----------------
miss = [x for x in R["sentences"] if x["status"] == "missing"]
with st.expander(f"🎙 Generate the missing sentences ({len(miss)})", expanded=bool(miss) and not S.get("v_stale")):
    if not miss:
        st.caption("No fully missing sentences. Sentences with only some words missing or changed are fixed in "
                   "the main editor (download the manifest below and upload it there).")
    elif not key:
        st.warning("Add a Gemini API key (sidebar) to generate audio.")
    else:
        pick = st.multiselect("Sentences to generate", [x["id"] for x in miss], default=[x["id"] for x in miss],
                              format_func=lambda k: f"#{k + 1} · {sentences[k][:60]}")
        if st.button("Generate selected", disabled=not pick or S.get("v_stale")):
            client = E.client(key)
            fb = E.global_speech_db(audio)
            for k in pick:
                with st.spinner(f"Generating sentence #{k + 1}…"):
                    try:
                        new, info = V.generate_missing(client, tts_model, voice, style, audio, R, k, boost,
                                                       temperature, tail_ui.lower(), 2, fb)
                        chk = V.check_clip(new, sentences[k], wh_size, lang.strip() or None, vad)
                        chk["id"] = k
                        S.v_gen[k] = dict(new=new, wav=E.wav_bytes(new), chk=chk)
                    except Exception as ex:
                        st.error(f"Sentence #{k + 1}: {ex}")
        for k, g in sorted(S.v_gen.items()):
            ok = g["chk"]["status"] == "ok"
            st.markdown(f"**#{k + 1}** {V.ICON[g['chk']['status']]} {V.LABEL[g['chk']['status']]} — {sentences[k]}")
            if not ok:
                st.markdown(V.render_html(dict(sentences=[g["chk"]])), unsafe_allow_html=True)
            st.audio(g["wav"], format="audio/wav")
            st.checkbox("Insert this one", value=ok, key=f"v_ins{k}")
        if S.v_gen and st.button("⬇ Insert the checked sentences into the audio", type="primary"):
            clips = {k: g["new"] for k, g in S.v_gen.items() if S.get(f"v_ins{k}")}
            if clips:
                S.v_audio = V.insert_sentences(audio, R, clips, E._snapper(audio))
                S.v_ver += 1
                S.v_gen, S.v_stale, S.v_dl = {}, True, {}
                st.rerun()

# ---------------- downloads ----------------
st.subheader("Reports and files")
rep = pd.DataFrame(V.issues_table(R))
if len(rep):
    st.dataframe(rep, hide_index=True)
if st.button("Prepare downloads"):
    snap = E._snapper(audio)
    S.v_dl = dict(wav=E.wav_bytes(audio),
                  manifest=json.dumps(V.manifest(R, len(audio), snap), ensure_ascii=False, indent=1).encode("utf-8"),
                  csv=rep.to_csv(index=False).encode("utf-8-sig"))
if S.get("v_dl"):
    d = S.v_dl
    cols = st.columns(3)
    cols[0].download_button("Audio (wav)", d["wav"], "verified_audio.wav", "audio/wav")
    cols[1].download_button("Manifest (JSON, with verification fields)", d["manifest"], "manifest.json",
                            "application/json")
    cols[2].download_button("Problem report (CSV)", d["csv"], "verification_report.csv", "text/csv")
    if S.get("v_stale"):
        st.caption("The manifest times belong to the audio before insertion. Run verification again, "
                   "then prepare the downloads again.")

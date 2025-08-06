import os, re, json, uuid, argparse, datetime, random, subprocess
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
from faster_whisper import WhisperModel

from openai import OpenAI
from langdetect import detect

import torch
from diffusers import StableDiffusionPipeline

from TTS.api import TTS
import librosa
import soundfile as sf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===== Config =====
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
HISTORY_JSON = DATA_DIR / "history.json"
HISTORY_EMB  = DATA_DIR / "history_emb.json"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")     # gpt-4o | gpt-4o-mini | gpt-4-turbo
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SD_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

VOICE_MODEL_ES = os.getenv("VOICE_MODEL_ES", "tts_models/es/css10/vits")
VOICE_MODEL_EN = os.getenv("VOICE_MODEL_EN", "tts_models/en/ljspeech/tacotron2-DDC")

PROMPT_BASE = """
Actúa como un guionista profesional. Escribe una historia inspiradora completamente ORIGINAL 
(no reutilices frases ni estructuras) de {longitud_min} a {longitud_max} palabras.

Condiciones:
- Tema del día: {tema}
- Estructura: gancho breve (<=80 palabras) → viaje del héroe → clímax emocional → moraleja clara.
- Cambia punto de vista y escenario respecto a historias previas.
- Incluye 2 giros narrativos y 1 metáfora distinta.
- Evita clichés como "nunca te rindas" o "todo es posible".
- Estilo: cinematográfico, con detalles sensoriales y emociones intensas.
- Entrega: TÍTULO, SINOPSIS (2–3 líneas) y luego la HISTORIA.

Restricciones sorpresa de hoy:
{restriccion}

Referencia de tono/temas (no copiar ni parafrasear literalmente):
{referencia}

Responde en el idioma detectado: {lang_code}.
Instrucciones del usuario: {instruccion}
"""

RESTRICCIONES = [
    "Ambientar en un puerto durante temporada de tormentas.",
    "Narrador poco confiable, revelar la verdad al final.",
    "Estructura no lineal con saltos temporales.",
    "Final abierto a interpretación, sin moraleja explícita.",
    "Contar a través de cartas encontradas.",
    "Protagonista con una cicatriz significativa.",
    "Ambientar en un pueblo de montaña aislado.",
    "Incluir un antagonista que se redime.",
    "Objeto simbólico que aparece en 3 momentos clave.",
    "Clímax con decisión moral difícil.",
    "Ambientar en una fábrica en decadencia.",
    "Mentor silencioso con un gesto característico.",
]

TEMAS = [
    "Resiliencia ante la adversidad",
    "Perdón después de la traición",
    "Superar miedo escénico",
    "Reconstruir la vida tras una pérdida",
    "Reinventarse profesionalmente",
    "Volver a empezar después de una quiebra",
    "Encontrar propósito en el voluntariado",
    "Vencer una enfermedad inesperada",
    "Lograr un sueño de la infancia",
    "Romper un ciclo familiar negativo",
]


# ===== Utilidades =====
def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]+)", url)
    if not m: raise ValueError("No se pudo extraer el ID del video.")
    return m.group(1)

def get_transcript_text(video_id: str, preferred_langs=('es','en')) -> str|None:
    try:
        trs = YouTubeTranscriptApi.get_transcript(video_id, languages=list(preferred_langs))
        return " ".join(seg['text'] for seg in trs if seg.get('text'))
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def download_audio(video_url: str, out_path: Path) -> Path:
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    tmp = stream.download(output_path=str(out_path.parent), filename=out_path.stem)
    p = Path(tmp)
    newp = out_path.with_suffix(".mp3")
    os.system(f'ffmpeg -y -i "{p}" -vn -acodec libmp3lame "{newp}"')
    if p.exists() and p != newp: p.unlink(missing_ok=True)
    return newp

def transcribe_with_whisper(audio_path: Path, lang_hint: str|None=None) -> str:
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")
    segments, info = model.transcribe(str(audio_path), language=lang_hint)
    return " ".join(seg.text.strip() for seg in segments)

def load_histories():
    if HISTORY_JSON.exists():
        return json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    return []

def save_history(meta_list):
    HISTORY_JSON.write_text(json.dumps(meta_list, ensure_ascii=False, indent=2), encoding="utf-8")

def load_embeddings():
    if HISTORY_EMB.exists():
        data = json.loads(HISTORY_EMB.read_text(encoding="utf-8"))
        return data.get("texts", [])
    return []

def save_embeddings(texts):
    HISTORY_EMB.write_text(json.dumps({"texts": texts}, ensure_ascii=False, indent=2), encoding="utf-8")

def tfidf_similarity(new_text: str, corpus_texts: list[str]) -> float:
    if not corpus_texts: return 0.0
    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(corpus_texts + [new_text])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    return float(sims.max()) if len(sims) else 0.0

def choose_voice_model(lang_code: str):
    return VOICE_MODEL_ES if str(lang_code).startswith("es") else VOICE_MODEL_EN

def text_to_speech(text: str, wav_out: Path, lang_code: str):
    tts = TTS(choose_voice_model(lang_code))
    tts.tts_to_file(text=text, file_path=str(wav_out))

def split_sentences(text: str):
    raw = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s for s in raw if s.strip()]

def generate_image(prompt: str, out_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    if device == "cuda": pipe = pipe.to("cuda")
    img = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    img.save(out_path)

def sanitize_filename(s: str):
    return re.sub(r'[^\w\-. ]', '_', s).strip()[:80]

def parse_title_and_body(text: str):
    title = "Historia Inspiradora"
    m = re.search(r"(?i)^t[íi]tulo[:\s\-]*\s*(.+)$", text, re.MULTILINE)
    if m: title = m.group(1).strip()
    body = text
    ms = re.search(r"(?i)^historia[:\s\-]*", text)
    if ms: body = text[ms.end():].strip()
    return title[:120], body


# ===== LLM (OpenAI) =====
def build_prompt(reference_text: str, instruction: str, tema: str, restriccion: str, lang_code: str, video_length_min: int = 8):
    palabras_por_minuto = 150
    longitud_min = palabras_por_minuto * video_length_min
    longitud_max = int(longitud_min * 1.2)
    return PROMPT_BASE.format(
        longitud_min=longitud_min, longitud_max=longitud_max,
        tema=tema, restriccion=restriccion, referencia=reference_text[:4000],
        lang_code=lang_code, instruccion=instruction
    )

def generate_story_with_chatgpt(prompt: str, model: str = OPENAI_MODEL,
                                temperature: float = 0.9, top_p: float = 0.9):
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un narrador experto en historias inspiradoras y creativas."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature, top_p=top_p
    )
    return resp.choices[0].message.content


# ===== Render del video con ffmpeg (quema subtítulos) =====
def make_video_ffmpeg(image_path: Path, audio_path: Path, srt_path: Path, out_path: Path, resolution=(1920,1080), fps=30):
    # Duración del audio
    audio, sr = librosa.load(str(audio_path), sr=None)
    duration = len(audio) / sr

    # -loop 1: mantener imagen fija, -t duración del audio
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
        "-vf", f"subtitles='{srt_path.as_posix()}',scale={resolution[0]}:{resolution[1]}:flags=lanczos",
        "-c:v", "libx264", "-t", f"{duration}",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-shortest",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)


# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube_url", required=True, help="Link público del video de referencia.")
    parser.add_argument("--instruction", required=True, help="Instrucciones (ej: 'Historia inspiradora de 15 minutos...').")
    parser.add_argument("--lang_hint", default="", help="Pista de idioma para Whisper (es|en).")
    parser.add_argument("--video_length", type=int, default=8, help="Duración deseada del video en minutos")
    parser.add_argument("--similarity_threshold", type=float, default=0.80)
    parser.add_argument("--out_prefix", default=None, help="Prefijo para carpeta de salida.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    out_id = args.out_prefix or datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    work_dir = OUT_DIR / out_id; work_dir.mkdir(parents=True, exist_ok=True)

    print(">> 1) Obteniendo transcripción/tono de referencia…")
    vid = extract_video_id(args.youtube_url)
    transcript = get_transcript_text(vid)
    if not transcript:
        print("No hay transcripción pública. Descargando audio y transcribiendo con Whisper…")
        audio_path_ref = work_dir / "ref_audio.mp3"
        download_audio(args.youtube_url, audio_path_ref)
        transcript = transcribe_with_whisper(audio_path_ref, lang_hint=args.lang_hint or None)

    # Detecta idioma
    try:
        lang_code = detect(transcript[:500]) if transcript else "es"
    except:
        lang_code = "es"

    tema = random.choice(TEMAS)
    restriccion = random.choice(RESTRICCIONES)
    prompt = build_prompt(transcript, args.instruction, tema, restriccion, lang_code, args.video_length)

    print(">> 2) Generando historia con ChatGPT… (modelo:", OPENAI_MODEL, ")")
    histories = load_histories()
    old_texts = load_embeddings()

    story = None
    for attempt in range(4):
        draft = generate_story_with_chatgpt(prompt)
        _, body_for_sim = parse_title_and_body(draft)
        sim = tfidf_similarity(body_for_sim, old_texts)
        print(f"   - Similitud con corpus previo: {sim:.3f}")
        if sim < args.similarity_threshold:
            story = draft
            old_texts.append(body_for_sim)
            save_embeddings(old_texts)
            break
        else:
            # Reforzar variedad
            tema = random.choice(TEMAS)
            restriccion = random.choice(RESTRICCIONES)
            prompt = build_prompt(transcript, args.instruction, tema, restriccion, lang_code)

    if not story:
        raise RuntimeError("No se pudo generar una historia suficientemente distinta. Intenta nuevamente.")

    title, body = parse_title_and_body(story)
    txt_out = work_dir / "story.txt"
    txt_out.write_text(story, encoding="utf-8")

    meta = {
        "id": out_id, "title": title, "tema": tema, "restriccion": restriccion,
        "lang": lang_code, "created_at": datetime.datetime.now().isoformat()
    }
    histories.append(meta); save_history(histories)

    print(">> 3) TTS (Coqui) → narración WAV…")
    wav_out = work_dir / "narration.wav"
    tts_model = TTS(choose_voice_model(lang_code))
    tts_model.tts_to_file(text=body, file_path=str(wav_out))

    print(">> 4) Subtítulos .srt… (timing proporcional)")
    srt_out = work_dir / "subtitles.srt"
    # Segmentación sencilla por oraciones y reparto proporcional al texto
    audio, sr = librosa.load(str(wav_out), sr=None)
    total_dur = len(audio) / sr
    sentences = [s for s in re.split(r'(?<=[\.\?\!])\s+', body.strip()) if s.strip()]
    weights = [max(1, len(s)) for s in sentences]
    total_w = sum(weights)
    t = 0.0
    def fmt_ts(sec: float):
        h = int(sec // 3600); sec -= 3600*h
        m = int(sec // 60); s = sec - 60*m
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    with srt_out.open("w", encoding="utf-8") as f:
        for i, s in enumerate(sentences, start=1):
            dur = total_dur * (weights[i-1] / total_w)
            a, b = t, t + dur
            t = b
            f.write(f"{i}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{s.strip()}\n\n")

    print(">> 5) Imagen (Stable Diffusion)… 1 sola imagen para todo el video")
    img_out = work_dir / "cover.png"
    img_prompt = f"Cinematic, emotional, highly-detailed illustration representing: {tema}. Inspirational, warm light, volumetric lighting, 4k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    if device == "cuda": pipe = pipe.to("cuda")
    img = pipe(img_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    img.save(img_out)

    print(">> 6) Render final (ffmpeg con subtítulos quemados)…")
    mp4_out = work_dir / f"{re.sub(r'[^\w\-. ]', '_', title).strip()[:80]}.mp4"
    make_video_ffmpeg(img_out, wav_out, srt_out, mp4_out, resolution=(args.width, args.height))

    print("\n✅ Listo")
    print(f"- Carpeta: {work_dir}")
    print(f"- Historia: {txt_out}")
    print(f"- Audio: {wav_out}")
    print(f"- Subtítulos: {srt_out}")
    print(f"- Imagen: {img_out}")
    print(f"- Video: {mp4_out}")


if __name__ == "__main__":
    main()


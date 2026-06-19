# main.py
import shutil, uuid, tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
encoder.load_model(Path("saved_models/default/encoder.pt"))
synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
vocoder.load_model(Path("saved_models/default/vocoder.pt"))


@app.post("/api/clone")
def clone_voice(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    text: str = Form(...),
):
    # FIXED: tempfile.gettempdir() Windows + Linux dono pe sahi path deta hai
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.wav"
    out_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}_out.wav"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(audio.file, f)

        wav = encoder.preprocess_wav(tmp_path)
        embed = encoder.embed_utterance(wav)
        spec = synthesizer.synthesize_spectrograms([text], [embed])[0]
        generated_wav = vocoder.infer_waveform(spec)

        import soundfile as sf
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        sf.write(str(out_path), generated_wav.astype(np.float32), synthesizer.sample_rate)
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    background_tasks.add_task(out_path.unlink, missing_ok=True)

    response = FileResponse(str(out_path), media_type="audio/wav", filename="cloned.wav")
    response.background = background_tasks
    return response
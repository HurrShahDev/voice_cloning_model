# api.py
import shutil, uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import numpy as np
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

app = FastAPI()

# Load models once at startup
encoder.load_model(Path("saved_models/default/encoder.pt"))
synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
vocoder.load_model(Path("saved_models/default/vocoder.pt"))

@app.post("/clone-voice")
async def clone_voice(audio: UploadFile = File(...), text: str = Form(...)):
    # Save uploaded audio temporarily
    tmp_path = Path(f"/tmp/{uuid.uuid4()}.wav")
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Run the pipeline
    wav = encoder.preprocess_wav(tmp_path)
    embed = encoder.embed_utterance(wav)
    spec = synthesizer.synthesize_spectrograms([text], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)

    # Save output and return
    out_path = Path(f"/tmp/{uuid.uuid4()}_out.wav")
    import soundfile as sf, torch
    import numpy as np
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    sf.write(str(out_path), generated_wav.astype(np.float32), synthesizer.sample_rate)

    tmp_path.unlink(missing_ok=True)  # cleanup input
    return FileResponse(str(out_path), media_type="audio/wav", filename="cloned.wav")
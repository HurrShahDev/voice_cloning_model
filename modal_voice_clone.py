# modal_voice_clone.py
# Place this file in the ROOT of your Real-Time-Voice-Cloning folder
# Deploy with: modal deploy modal_voice_clone.py

import modal
from pathlib import Path

# ─────────────────────────────────────────────
# 1.  IMAGE  — mirrors your local requirements
#     (PyQt5, visdom, sounddevice removed — not
#      needed on a headless GPU server)
# ─────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        # audio / codec deps your libs need at the C level
        "libsndfile1",
        "ffmpeg",
        "libportaudio2",
        "gcc",
        "g++",
    )
    .pip_install(
        # Core ML
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "torchvision==0.16.0",
        "numpy==1.26.4",
        # Audio
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "webrtcvad-wheels==2.0.14",
        "scipy==1.13.0",
        "numba==0.59.1",
        "audioread==3.0.1",
        "soxr==0.3.7",
        # TTS / synthesis deps
        "Unidecode==1.3.8",
        "inflect==7.0.0",
        # API / serving
        "fastapi==0.111.0",
        "python-multipart==0.0.9",
        "uvicorn==0.30.1",
        "pydantic==2.7.1",
        # Misc
        "requests==2.32.3",
        "tqdm==4.66.4",
        "matplotlib==3.9.0",
        "scikit-learn==1.5.0",
    )
)

# ─────────────────────────────────────────────
# 2.  APP
# ─────────────────────────────────────────────
app = modal.App("rtvc-voice-cloning", image=image)

# ─────────────────────────────────────────────
# 3.  VOLUME — stores your .pt model files
#     so they are NOT re-uploaded on every call
#
#     HOW TO UPLOAD YOUR MODELS (one-time):
#       modal volume create rtvc-models
#       modal volume put rtvc-models \
#         saved_models/default/encoder.pt   /default/encoder.pt
#       modal volume put rtvc-models \
#         saved_models/default/synthesizer.pt /default/synthesizer.pt
#       modal volume put rtvc-models \
#         saved_models/default/vocoder.pt   /default/vocoder.pt
# ─────────────────────────────────────────────
volume = modal.Volume.from_name("rtvc-models", create_if_missing=True)
MODEL_DIR = Path("/models")   # where the volume is mounted inside the container

# ─────────────────────────────────────────────
# 4.  CLONE CLASS  — loaded once per container,
#     reused across every request (cold-start
#     only happens on the first call or scale-up)
# ─────────────────────────────────────────────
@app.cls(
    gpu="T4",                          # change to "A10G" or "A100" if you need more VRAM
    volumes={str(MODEL_DIR): volume},
    # Mount the entire RTVC source so encoder / synthesizer / vocoder packages
    # are importable inside the container
    mounts=[
        modal.Mount.from_local_dir(
            ".",                       # your local RTVC root
            remote_path="/rtvc",
            condition=lambda p: not any(
                skip in p for skip in [
                    "__pycache__", ".git", "saved_models",
                    "demo_output", ".wav", "venv_phase1",
                ]
            ),
        )
    ],
)
class VoiceCloner:

    @modal.enter()
    def load_models(self):
        """Runs once when the container starts — keeps models hot in memory."""
        import sys
        sys.path.insert(0, "/rtvc")   # make encoder / synthesizer / vocoder importable

        from encoder import inference as encoder
        from synthesizer.inference import Synthesizer
        from vocoder import inference as vocoder

        encoder.load_model(MODEL_DIR / "default" / "encoder.pt")
        self.synthesizer = Synthesizer(MODEL_DIR / "default" / "synthesizer.pt")
        vocoder.load_model(MODEL_DIR / "default" / "vocoder.pt")

        # keep refs so @web_endpoint methods can use them
        self.encoder = encoder
        self.vocoder = vocoder

    # ──────────────────────────────────────────
    # 5.  WEB ENDPOINT
    #     POST  /clone
    #     multipart/form-data:
    #       audio : WAV file  (the reference speaker)
    #       text  : string    (what to say)
    #     Returns: audio/wav  binary
    # ──────────────────────────────────────────
    @modal.web_endpoint(method="POST")
    async def clone(self, request):
        """
        Accepts multipart/form-data with:
          - audio: reference .wav file
          - text:  text to synthesize

        Returns raw WAV bytes with Content-Type: audio/wav
        """
        import shutil, uuid, tempfile
        import numpy as np
        import soundfile as sf
        from fastapi import Request
        from fastapi.responses import Response

        # --- parse multipart form ---
        form = await request.form()

        text_field = form.get("text")
        audio_field = form.get("audio")

        if not text_field:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Missing field: text")
        if not audio_field:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Missing field: audio")

        text = str(text_field)

        # save uploaded audio to a temp file
        tmp_dir  = Path(tempfile.gettempdir())
        tmp_in   = tmp_dir / f"{uuid.uuid4()}.wav"
        tmp_out  = tmp_dir / f"{uuid.uuid4()}_out.wav"

        try:
            with tmp_in.open("wb") as f:
                # audio_field is a Starlette UploadFile
                shutil.copyfileobj(audio_field.file, f)

            # ── core RTVC pipeline (identical to your local main.py) ──
            wav   = self.encoder.preprocess_wav(tmp_in)
            embed = self.encoder.embed_utterance(wav)
            spec  = self.synthesizer.synthesize_spectrograms([text], [embed])[0]
            generated_wav = self.vocoder.infer_waveform(spec)

            generated_wav = np.pad(
                generated_wav,
                (0, self.synthesizer.sample_rate),
                mode="constant",
            )
            sf.write(
                str(tmp_out),
                generated_wav.astype(np.float32),
                self.synthesizer.sample_rate,
            )

            # read back as bytes and return directly
            audio_bytes = tmp_out.read_bytes()

        finally:
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="cloned.wav"'},
        )
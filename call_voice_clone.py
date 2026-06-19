# test_voice_clone_endpoint.py
# Run this from your main backend to test the Modal endpoint
# pip install requests

import requests

# ── Replace with your actual Modal URL after deploying ──
# It looks like: https://YOUR-USERNAME--rtvc-voice-cloning-voicecloner-clone.modal.run
MODAL_URL = "https://YOUR-USERNAME--rtvc-voice-cloning-voicecloner-clone.modal.run"

def test_clone(audio_path: str, text: str, output_path: str = "output_cloned.wav"):
    """
    Calls the Modal /clone endpoint.

    Args:
        audio_path:  path to a reference .wav file (the speaker's voice)
        text:        what you want the cloned voice to say
        output_path: where to save the returned .wav
    """
    with open(audio_path, "rb") as f:
        response = requests.post(
            MODAL_URL,
            files={"audio": ("reference.wav", f, "audio/wav")},
            data={"text": text},
            timeout=120,   # first call has cold-start, give it time
        )

    if response.status_code == 200:
        with open(output_path, "wb") as out:
            out.write(response.content)
        print(f"✅ Success! Saved to {output_path}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    test_clone(
        audio_path="samples/1320_00000.mp3",  # any reference wav/mp3 in your RTVC folder
        text="Hello, this is my cloned voice speaking.",
    )
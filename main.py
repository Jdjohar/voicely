import os
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from starlette.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# ✅ Initialize FastAPI app
app = FastAPI()
print(f"✅ ENV: PORT={os.environ.get('PORT')}")
print(f"✅ ENV: COQUI_TOS_AGREED={os.environ.get('COQUI_TOS_AGREED')}")

# ✅ Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Compatibility for PyTorch 2.6+
torch.serialization.add_safe_globals([XttsConfig])

# ✅ Automatically agree to Coqui license
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

# ✅ Initialize XTTS model (with GPU if available)
print("⏳ Loading TTS model...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print(f"✅ Model loaded successfully on {device}!")

# ✅ Ensure output directory exists
output_dir = "outputs"
Path(output_dir).mkdir(exist_ok=True)

# ✅ Check for speaker WAV file
custom_speaker_wav = os.path.join(os.getcwd(), "sample.wav")
if not os.path.isfile(custom_speaker_wav):
    raise FileNotFoundError(f"❌ Custom speaker WAV file not found: {custom_speaker_wav}")

# ✅ Health check (for Render port detection)
@app.get("/health")
async def health():
    return {"status": "ok"}

# ✅ Homepage - Render HTML Form
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <h1>Text-to-Speech Generator 🎤</h1>
    <form method="post" action="/generate">
        <label>Enter Text:</label>
        <textarea name="text" rows="4" cols="50">खुशहाल गाँव की कहानी...</textarea>
        <br>
        <label>Select Language:</label>
        <select name="language">
            <option value="hi" selected>Hindi</option>
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
        </select>
        <br>
        <button type="submit">Generate Speech</button>
    </form>
    """

# ✅ Handle form submission and speech generation
@app.post("/generate")
async def generate_speech(text: str = Form(...), language: str = Form(...)):
    try:
        # Supported languages
        supported_languages = ["hi", "en", "es", "fr"]
        if language not in supported_languages:
            return HTMLResponse(f"<h3>Error: Unsupported language '{language}'</h3>")

        # Output file path
        output_path = f"{output_dir}/output.wav"

        print(f"🎙️ Generating speech in '{language}'...")

        # Generate speech
        tts.tts_to_file(text=text, speaker_wav=custom_speaker_wav, language=language, file_path=output_path)

        print("✅ Speech generation completed!")
        return RedirectResponse(url=f"/download/output.wav", status_code=303)

    except Exception as e:
        print(f"❌ Error: {e}")
        return HTMLResponse(f"<h3>Error: {e}</h3>")

# ✅ Serve output files for download
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    return HTMLResponse("<h3>File not found!</h3>")

# ✅ Ensure Dynamic Port Binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))  # Dynamic port from Render
    print(f"🚀 Starting FastAPI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

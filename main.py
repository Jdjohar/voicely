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

# ✅ HTML Template for Home Page
html_template = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Text-to-Speech (TTS)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="bg-light">

    <div class="container mt-5">
        <h1 class="mb-4">Text-to-Speech (TTS) Generator 🎤</h1>

        <form method="post" action="/generate" enctype="application/x-www-form-urlencoded">
            <div class="mb-3">
                <label for="text" class="form-label">Enter your story:</label>
                <textarea class="form-control" id="text" name="text" rows="5" required>खुशहाल गाँव की कहानी...</textarea>
            </div>

            <div class="mb-3">
                <label for="language" class="form-label">Select Language:</label>
                <select class="form-select" id="language" name="language" required>
                    <option value="hi" selected>Hindi</option>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                </select>
            </div>

            <button type="submit" class="btn btn-primary">Generate Speech</button>
        </form>

    </div>

</body>

</html>
"""

# ✅ HTML Template for Download Page
download_template = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Download Speech</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="bg-light">

    <div class="container mt-5">
        <h1 class="mb-4">🎧 Your Speech is Ready!</h1>
        <a href="/download/{filename}" class="btn btn-success">Download Speech</a>
        <a href="/" class="btn btn-secondary">Generate Again</a>
    </div>

</body>

</html>
"""

# ✅ Homepage - Render HTML Form
@app.get("/", response_class=HTMLResponse)
async def home():
    return html_template

# ✅ Handle form submission and speech generation
@app.post("/generate")
async def generate_speech(text: str = Form(...), language: str = Form(...)):
    try:
        # Supported languages (extend if needed)
        supported_languages = ["hi", "en", "es", "fr"]
        if language not in supported_languages:
            return HTMLResponse(f"<h3>Error: Unsupported language '{language}'</h3>")

        # Output file path
        output_path = f"{output_dir}/output.wav"

        print(f"🎙️ Generating speech in '{language}'...")

        # Generate speech (without the unsupported 'stream' argument)
        tts.tts_to_file(text=text, speaker_wav=custom_speaker_wav, language=language, file_path=output_path)

        print("✅ Speech generation completed!")

        # Redirect to download page with the filename
        return RedirectResponse(url=f"/success?filename=output.wav", status_code=303)

    except Exception as e:
        print(f"❌ Error: {e}")
        return HTMLResponse(f"<h3>Error: {e}</h3>")

# ✅ Success page with download link
@app.get("/success", response_class=HTMLResponse)
async def success_page(filename: str):
    return download_template.replace("{filename}", filename)

# ✅ Serve output files for download
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    return HTMLResponse("<h3>File not found!</h3>")
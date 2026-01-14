from fastapi import FastAPI, UploadFile
import whisper

app = FastAPI()
model = whisper.load_model("small")  # modelo mais leve para rodar rápido

@app.post("/transcribe/")
async def transcribe(file: UploadFile):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    result = model.transcribe(audio_path)
    return {"text": result["text"]}

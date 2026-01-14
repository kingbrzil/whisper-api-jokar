from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile, os

app = Flask(__name__)

model = WhisperModel("base", device="cpu", compute_type="int8")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files["audio"]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tmp_path = tmp.name
    tmp.close()

    f.save(tmp_path)

    segments, info = model.transcribe(tmp_path, language="pt")
    text = " ".join(seg.text.strip() for seg in segments)

    try:
        os.remove(tmp_path)
    except:
        pass

    return jsonify({"text": text})

if __name__ == "__main__":
    import os
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8765)))


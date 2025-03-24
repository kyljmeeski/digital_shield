import json
from typing import Dict, List

from flask import Flask, request, jsonify
from pydub import AudioSegment
import speech_recognition as sr
import os

from sentence_transformers import SentenceTransformer

from Pair import Pair
from Phrase import Phrase
from Transcription import Transcription


SIMILARITY_THRESHOLD = 0.6  # настолько предложение должна быть похожа на фразу
SIGNATURE_THRESHOLD = 2  # вот на столько фраз должно быть похоже предложение
MARKERS_THRESHOLD = 3  # вот столько подозрительных предложений должно быть

SEMANTIC_ANALYZER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# SEMANTIC_ANALYZER.save('paraphrase-multilingual-MiniLM-L12-v2')  # for the first time only -- to save locally
app = Flask(__name__)

# Папка для хранения временных файлов
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Преобразование MP3 в WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    print(f"Файл {mp3_file} преобразован в {wav_file}")


# Распознавание речи из WAV
def recognize_speech_from_wav(wav_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language='ru-RU')  # Для русского языка
        return text
    except sr.UnknownValueError:
        return "Речь не распознана"
    except sr.RequestError as e:
        return f"Ошибка при запросе к сервису Google Speech Recognition: {e}"


def load_signatures() -> Dict[str, List[str]]:
    signatures = dict()
    with open("signatures.json", encoding="utf-8") as file:
        data = json.load(file)
    for signature, sentences in data.items():
        signatures[signature] = sentences
    return signatures


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file and not file.filename.endswith(".mp3"):
        return jsonify({"error": "Invalid file format. Only MP3 files are allowed."}), 400



    # Сохранение временного файла
    mp3_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(mp3_path)

    # Конвертация MP3 в WAV
    wav_path = os.path.splitext(mp3_path)[0] + '.wav'
    convert_mp3_to_wav(mp3_path, wav_path)

    # Распознавание речи из WAV
    text = recognize_speech_from_wav(wav_path)
    markers = dict()
    signatures = load_signatures()
    chat_sentences = Transcription(text).sentences()
    for chat_sentence in chat_sentences:
        for signature, signature_sentences, in signatures.items():
            signature_markers = dict()
            for signature_sentence in signature_sentences:
                similarity = Pair(Phrase(chat_sentence, SEMANTIC_ANALYZER),
                                  Phrase(signature_sentence, SEMANTIC_ANALYZER),
                                  SEMANTIC_ANALYZER
                                  ).similarity()
                print(chat_sentence)
                print("\t", signature_sentence, similarity)
                if similarity >= SIMILARITY_THRESHOLD:
                    signature_markers[chat_sentence] = signature
            if len(signature_markers) > SIGNATURE_THRESHOLD:
                markers.update(signature_markers)
    if len(markers) >= MARKERS_THRESHOLD:
        markers["suspicious"] = "true"
    else:
        markers["suspicious"] = "false"

    # Удаляем временные файлы
    os.remove(mp3_path)
    os.remove(wav_path)
    return jsonify(markers), 200


NUMBERS = ["996708584859"]


@app.route("/scam", methods=["GET"])
def check_number() -> bool:
    name = request.args.get('name', 'Guest')
    return name in NUMBERS


@app.route("/scam", methods=["POST"])
def add_number():
    data = request.get_json()  # Parse JSON request body
    number = data.get('number', 'No number provided')  # Extract 'number' field
    NUMBERS.append(number)


if __name__ == '__main__':
    app.run(debug=True)

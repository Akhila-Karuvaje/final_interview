from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import speech_recognition as sr
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import json

# ----------------- FLASK CONFIG -----------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ----------------- CRITICAL FIX: Download NLTK Data FIRST -----------------
print("🔧 Downloading NLTK data BEFORE importing nltk...")

import subprocess
import sys

# Download using Python subprocess to avoid importing nltk
nltk_data_dir = '/root/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)

resources = ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab', 'stopwords']
for resource in resources:
    try:
        # Use subprocess to download without importing nltk modules
        result = subprocess.run(
            [sys.executable, '-m', 'nltk.downloader', '-d', nltk_data_dir, resource],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"  ✅ {resource} downloaded")
        else:
            print(f"  ℹ️ {resource} may already exist or download failed")
    except Exception as e:
        print(f"  ⚠️ Error downloading {resource}: {e}")

print("📦 NLTK data download complete, now importing nltk...")

# ----------------- NOW SAFE TO IMPORT NLTK -----------------
import nltk
import ssl

# Disable SSL verification for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data paths
nltk.data.path.insert(0, nltk_data_dir)
nltk.data.path.insert(0, os.path.join(os.getcwd(), 'nltk_data'))

print(f"📂 NLTK data paths: {nltk.data.path[:2]}")

# Import NLTK utilities
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    print("✅ NLTK ready!")
except Exception as e:
    print(f"⚠️ NLTK import warning: {e}")
    # Fallback functions
    def word_tokenize(text):
        return text.split()
    class DummyStopwords:
        def words(self, lang):
            return set()
    stopwords = DummyStopwords()

# ----------------- GOOGLE AI CONFIG -----------------
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not set!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    print("✅ Gemini API configured")

# ----------------- VIDEO FEATURES (OPTIONAL) -----------------
USE_VIDEO_FEATURES = False
model_whisper = None
model_bert = None

if os.environ.get('USE_VIDEO_FEATURES', 'false').lower() == 'true':
    try:
        import cv2
        import librosa
        import numpy as np
        from sentence_transformers import SentenceTransformer, util
        import whisper
        import mediapipe as mp
        
        print("⏳ Loading video models...")
        model_whisper = whisper.load_model("small")
        model_bert = SentenceTransformer('all-MiniLM-L6-v2')
        USE_VIDEO_FEATURES = True
        print("✅ Video features enabled")
    except Exception as e:
        print(f"⚠️ Video features disabled: {e}")
else:
    print("ℹ️ Video features disabled")

# ============================================================
# ===== Utility Functions ====================================
# ============================================================

def SpeechToText():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source)
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-IN')
        return query
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clean_answer(answer):
    try:
        words = word_tokenize(answer)
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in words if word.lower() not in stop_words])
    except Exception as e:
        print(f"Clean answer error: {e}")
        return answer

# ============================================================
# ===== Routes ===============================================
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    job = request.form['job']
    level = request.form['level']
    session['job_title'] = job
    session['difficulty'] = level
    return redirect(url_for('regenerate_questions'))

@app.route('/regenerate_questions')
def regenerate_questions():
    job = session.get('job_title')
    level = session.get('difficulty')

    prompt = f"""
    Generate exactly 10 interview questions for the job role: {job} 
    with difficulty level: {level}. 
    Only return the 10 questions in plain text, numbered 1 to 10. 
    Do not include any introduction or extra comments.
    """
    
    try:
        response = model.generate_content(prompt)
        raw_questions = response.text.strip().split("\n")
        questions = []
        for q in raw_questions:
            match = re.match(r'^\d+[\).\s-]+(.*)', q.strip())
            if match:
                questions.append(match.group(1).strip())

        questions = questions[:10]
        session['questions'] = questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        session['questions'] = ["Error generating questions. Please try again."]

    return redirect(url_for('questions'))

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    job = session.get('job_title')
    difficulty = session.get('difficulty')
    question_list = list(enumerate(questions, start=1))
    return render_template('questions.html', questions=question_list, job_title=job, difficulty=difficulty)

@app.route('/interview/<int:qid>')
def interview(qid):
    questions = session.get('questions', [])
    if 1 <= qid <= len(questions):
        question = questions[qid - 1]
    else:
        question = 'No question found'
    return render_template('interview.html', question=question, qid=qid)

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    audio_path = "user_audio.wav"
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
        duration = 10
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return jsonify({
        "transcription": transcribed_text,
        "duration": duration
    })

@app.route('/submit_answer/<qid>', methods=['POST'])
def submit_answer(qid):
    user_answer = request.form.get('answer', '').strip()

    prompt = f"""
    You are an expert interviewer. Analyze the user's answer concisely in JSON:

    {{
        "correct_answer": "Ideal answer",
        "validation": "Valid/Invalid/Partial",
        "feedback": "Brief feedback"
    }}

    Question ID: {qid}
    User Answer: "{user_answer}"
    """
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON object found")
    except Exception as e:
        print(f"Error analyzing answer: {e}")
        result = {
            "correct_answer": "Unable to parse response.",
            "validation": "Unknown",
            "feedback": "N/A"
        }

    return jsonify({
        'user_answer': user_answer,
        'validation_result': {
            'correct_answer': result.get('correct_answer', ''),
            'validation': result.get('validation', ''),
            'feedback': result.get('feedback', '')
        },
    })

# ============================================================
# ===== Video Interview Routes ===============================
# ============================================================

@app.route('/video_interview')
def video_interview():
    if not USE_VIDEO_FEATURES:
        return "Video features are disabled. Set USE_VIDEO_FEATURES=true environment variable.", 503
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    if not USE_VIDEO_FEATURES:
        return jsonify({"error": "Video features disabled"}), 503
        
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", f"answer_{qid}.webm")
    file.save(filepath)

    try:
        result = model_whisper.transcribe(filepath)
        transcript = result['text']

        prompt = f"""
        You are an expert interview evaluator.
        Analyze this interview answer for question ID {qid} and return JSON in this exact format:
        {{
            "Confidence Score": <float between 0 and 1>,
            "Content Relevance": <float between 0 and 1>,
            "Fluency Score": <float between 0 and 1>,
            "Feedback": "3–5 line constructive feedback on how the user can improve"
        }}

        User's answer transcript: "{transcript}"
        """

        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)

        if json_match:
            json_str = json_match.group()
            scores = json.loads(json_str)
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        print(f"Video analysis error: {e}")
        scores = {
            "Confidence Score": 0.0,
            "Content Relevance": 0.0,
            "Fluency Score": 0.0,
            "Feedback": "Unable to analyze video properly."
        }
        transcript = "Transcription failed"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    final_eval = round(
        (scores["Confidence Score"] +
         scores["Content Relevance"] +
         scores["Fluency Score"]) / 3 * 100, 2
    )

    session['video_feedback'] = scores["Feedback"]

    return jsonify({
        "Confidence Score": scores["Confidence Score"],
        "Content Relevance": scores["Content Relevance"],
        "Fluency Score": scores["Fluency Score"],
        "Final Evaluation": final_eval,
        "Transcript": transcript,
        "Feedback": scores["Feedback"]
    })

# ============================================================
# ===== Result Page ==========================================
# ============================================================

@app.route('/result')
def result():
    feedback = session.get('video_feedback', "No feedback available yet.")
    return render_template('result.html', feedback=feedback)

# ============================================================
# ===== Health Check (for Render) ===========================
# ============================================================

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "nltk_ready": True}), 200

# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

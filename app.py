from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
import os
import traceback

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

conversation_history = []

SYSTEM_PROMPT = """You are NexusAI — a smart, friendly, and highly capable AI assistant.
You are helpful, concise, enthusiastic, and great at explaining data science, coding, machine learning, and general topics."""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data or not data.get("message"):
            return jsonify({"error": "Empty message"}), 400

        user_message = data["message"].strip()

        conversation_history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        response = client.models.generate_content(
            model="gemini-2.5-flash",   # Latest stable fast model (2026)
            contents=conversation_history,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=1000
            )
        )

        assistant_reply = response.text.strip()

        conversation_history.append({
            "role": "model",
            "parts": [{"text": assistant_reply}]
        })

        return jsonify({"reply": assistant_reply})

    except Exception as e:
        print("=== BACKEND ERROR ===")
        print(str(e))
        print(traceback.format_exc())
        print("=====================")
        return jsonify({"error": "Something went wrong on our side. Please try again."}), 500

@app.route("/clear", methods=["POST"])
def clear():
    conversation_history.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    print("🚀 NexusAI Server running on http://127.0.0.1:8080")
    app.run(debug=True, port=8080)
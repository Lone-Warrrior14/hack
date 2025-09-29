import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# --- Gemini AI Configuration ---
def initialize_gemini():
    """Initializes the Gemini model."""
    # 🔑 SECURITY NOTE: For production, it's highly recommended to load the 
    # API key from a secure location like an environment variable.
    # e.g., api_key = os.environ.get("GEMINI_API_KEY")
    
    # Replace the placeholder with your actual Gemini API key.
    api_key = "AIzaSyB-xO57JCrkj4_wFk-ZMtZGU7D1TfHbuwo" 
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Gemini API key is not set. Please replace 'YOUR_GEMINI_API_KEY_HERE' in main.py with your actual key.")

    genai.configure(api_key=api_key)
    
    # We'll use a powerful and up-to-date model.
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    return model

# Initialize the Gemini model when the application starts
try:
    gemini_model = initialize_gemini()
except ValueError as e:
    # Handle the case where the API key is not set and print an error.
    print(f"Error initializing Gemini: {e}")
    gemini_model = None

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates')

# --- DEBUGGING: Print the application's root path ---
# This will show us the exact directory Flask is using as its starting point.
print(f"Flask App Root Path: {app.root_path}")
# This shows the current working directory from the OS's perspective.
print(f"Current Working Directory: {os.path.abspath(os.getcwd())}")


# --- API and Web Page Routes ---

@app.route('/')
def index():
    """Serves the main HTML chat page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint to interact with the chatbot."""
    if not gemini_model:
        return jsonify({"error": "The Gemini model is not initialized. Please check your API key in main.py."}), 500

    try:
        data = request.get_json()
        user_prompt = data.get('prompt')
        history = data.get('history', []) # Get history from frontend

        if not user_prompt:
            return jsonify({"error": "A 'prompt' is required in the request."}), 400

        # Start a chat session with the provided history
        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(user_prompt)
        
        # The frontend will manage the history; we just need to send back the new text.
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"An error occurred in /api/chat: {e}")
        return jsonify({"error": "An internal server error occurred on the server."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # For development, debug=True is useful. For production, use a proper WSGI server.
    app.run(host='0.0.0.0', port=5000, debug=True)


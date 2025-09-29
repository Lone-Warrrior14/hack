import google.generativeai as genai

def initialize_gemini():
    # 🔑 Directly use your Gemini API key here
    api_key = "AIzaSyB-xO57JCrkj4_wFk-ZMtZGU7D1TfHbuwo"

    # Configure the SDK
    genai.configure(api_key=api_key)
    
    # Choose a Gemini model (flash = faster, pro = more powerful)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model

def chat_with_gemini(model, user_input, history=None):
    """
    model: the Gemini model object
    user_input: the user’s message/text
    history: optional list of prior messages (for context)
    """
    if history is None:
        history = []
    
    # Start a chat session (stateful)
    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    
    # response.text will have the model’s reply
    return response.text, chat.history  # return reply + updated history

def main():
    model = initialize_gemini()
    print("Welcome to Gemini chatbot! (type 'exit' to quit)")
    history = []
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        reply, history = chat_with_gemini(model, user_input, history)
        print("Bot:", reply)

if __name__ == "__main__":
    main()

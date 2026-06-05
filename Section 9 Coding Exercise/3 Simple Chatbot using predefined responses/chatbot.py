# Import regex
import re

# Mapped reposnses to keywords
responses = {
    "Hello": "Hi there! How can I help you today?",
    "hi": "Hello! How can I help you?",
    "how are you": "I am just a bot, but I am doing great! How about you?",
    "what is your name": "I'm a chatbot created to assist you",
    "help": "Sure, I am here to help. What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're wellcome! I'm happy to help.",
    "default": "I'm not sure I uderstand. Could you please rephrase?"
}

# Funtion to find the appropriate response based on the user input
def chatbot_response(user_input):
    # Convert input to lowercase
    user_input = user_input.lower()

    for keyword in responses:
        if re.search(keyword, user_input):
            return responses[keyword]

    return responses["default"]

# Main function to run the chatbot
def chatbot():
    print(f"Chatbot: Hello! I'm here to assist you. (type 'bye' to exit)")

    while True:
        # Get user input
        user_input = input("You: ")

        # If user types 'bye', exit the loop
        if user_input.lower() == 'bye':
            print(f"Chatbot: Goodbye! Have a great day!")
            break

        # Get chatbot's response based on user inpput
        response = chatbot_response(user_input)

        # Print Chatbot's response
        print(f"Chatbot:{response}")

# Run Chatbot
chatbot()
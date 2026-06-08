# Install libraries
## pip install SpeechRecognition pyttsx3 pyaudio

# import libraries
## speech revognition
import speech_recognition as sr

## speech a text to speech allowing the assistant to respond vocally
import pyttsx3

## Provides date and time information
import datetime

## Using web browser
import webbrowser

## Allows the assistant to run system commands like opening notepad or calculator or any other tools that you have
import os

# Initialize the speech engine for text to speech
engine = pyttsx3.init()

# Function to make the assistant speak
def speak(text):
    print(f"Assistant: {text}")  # add this to debug
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

# Function to take a voice command from the user
## This command this function kind of listens to the user's voice using the microphone, converts the audio to text using Google speech recognition, and returns the text command in lowercase back to us
def take_command():
    # This will recognize the voice
    recognizer = sr.Recognizer()        
    with sr.Microphone() as source:
        print("Listening...")
        # Adjust for background noise to improve recognition
        recognizer.adjust_for_ambient_noise(source)
        # Activelly listening
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            # Call that function Google API with the audio and see if we can understand what it's saying
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}")
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Network error.")
            return None

    # Command it converted from speech to text will be returned back
    return command.lower()

# Function to respond to different commands
## This function handles the assistance responses based on the user's command
def respond(command):
    if 'hello' in command or 'hi' in command:
        speak("Hello! How can I help you today?")

    elif 'time' in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")

    elif 'weather' in command:
        speak("Searching for current weather")
        webbrowser.open("https://www.google.com/search?q=current+weather")

    elif 'search' in command:
        search_query = command.replace('search for', '').replace('search', '').strip()
        if search_query:
            speak(f"Searching for {search_query}")
            webbrowser.open(f"https://www.google.com/search?q={search_query}")

    elif 'firefox' in command:
        speak("Opening Firefox")
        os.system("start firefox")

    elif 'calculator' in command:
        speak("Opening Calculator")
        os.system("start calc")

    elif 'bye' in command or 'exit' in command or 'quit' in command:
        speak("Goodbye! Have a great day.")
        exit()

    else:
        speak("I'm sorry, I don't know that command.")

# Main function to run the assistant
def run_assistant():
    speak("Hello I am your assistent. How can I help you?")
    while True:
        command = take_command()
        if command:
            respond(command)

# Start the assistant
run_assistant()
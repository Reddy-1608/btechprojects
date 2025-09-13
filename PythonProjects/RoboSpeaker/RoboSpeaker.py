#Program for converting the input text into speech or convert it into speaking
#importing the library for converting text to speech
import pyttsx3
from pyttsx3 import speak #impoting speak as it is important to define the word 'speak'
print("welcome to ROBO SPEAKER")  #just for an intro
speak("welcome to ROBO SPEAKER please enter your input")  #just for adding on to the progran

#checking whether the user want to exit or convert his text to speech
while True:
    # asking the input from the user
    x = input("enter your word which has to be spoken('for exiting the program type exit'): ")
     #the input is inside the loop  because it will take the input as many time the loop gets an 'exit' from the output screen
    if(x.lower() == "exit"): #converting upper to lower so that it wont be a problem
        print("exiting the program")
        print("goodbye")
        speak("goodbye")
        break
    else:
        speak(x)

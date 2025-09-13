#program for weather details
import json  # the data from weather will be in json format and this is used because it is convinient

import requests  # library used for accessing the data
from pyttsx3 import speak  # importing speak from the library

#asking the use the input
city = input("enter the city name: ")

#url for the weather details with the api key created by us
url = f"https://api.weatherapi.com/v1/current.json?key=cbb6a462a86e4fa399c54800240310&q={city}"

#this line sends a get request
x = requests.get(url)
#print(x.text) #contains all the details such as humidity,wind speed etc..
#for coverting json to dictionary
wdic = json.loads(x.text)
y = wdic["current"]["temp_c"]#assigning so that it will be easy for the speak function
print(wdic["current"]["temp_c"])
speak(f"the temperture in {city} is {y} degrees") #speaking the temperature
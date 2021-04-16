import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer
import numpy
import tflearn
import tensorflow
import json
import random
import pickle

#nltk.download('punkt')

with open("contents.json") as archivo:
	datos = json.load(archivo)

palabras = []
tags = []
auxX = []
auxY = []

for contents in datos["contents"]:
	for patrones in contents["patrones"]:
		auxPalabra = nltk.word_tokenize(patrones)
		palabras.extend(auxPalabra)
		auxX.append(auxPalabra)
		auxY.append(contents["tag"])

		if contents["tag"] not in tags:
			tags.append(contents["tag"])

print(palabras)
print(auxX)
print(auxY)
print(tags)
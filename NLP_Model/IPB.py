import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())  # Opening the json file

#Opening words and classes data from pkl files
words = pickle.load(open('words.pkl', 'rb')) 
classes = pickle.load(open('classes.pkl', 'rb'))

#loading the model
model = load_model('IPB_model.h5')

# Creating a function for cleaning up and tokenizing sentences
def cleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return sentenceWords

# Creating a function for enumerating cleaned up sentences
def bagOfWords(sentence):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0] * len(words)

    for w in sentenceWords:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Creating a function for predicting the class of the function
def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]
    errorThreshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > errorThreshold]
    results.sort(key = lambda x: x[1], reverse = True)
    returnList = []

    for r in results:
        returnList.append({'intent': classes[r[0]], 'probability:': str(r[1])})
    
    return returnList

# Creating a fuction for response
def getResponse(intentsList, intentsJson):
    tag = intentsList[0]['intent']
    listOfIntents = intentsJson['intents']
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Creating a loop for bot to run
while True:
    message = input('You: ')
    ints = predictClass(message)
    res = getResponse(ints, intents)
    print('IPB: ', res)
    if message.lower() == 'exit':
        break

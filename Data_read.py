import pandas as pd
df = pd.read_csv("./Data/data.csv", sep=",")
print(df)
#print(df.iloc[:,1])
#print(len(list(set(list(df["Technologies"])))))
tech=list(set(list(df["Technologies"])))
#print(tech)
#
# print(len(list(set(list(df["Metier"])))))


import nltk
nltk.download('punkt')
import string 
#from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
#print("remove_punctuation_map: ",remove_punctuation_map)
from stop_words import get_stop_words
#stop_words = get_stop_words('fr')
stop_words=[" ", "/"]
def stem_tokens(text):
    """
    Stemming each words using NLTK
    """
    return [stemmer.stem(token) for token in text]

def normalize(text):
    """
    remove punctuation, lowercase, team
    """
    #print(stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map))))

    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

def cosine_sim(text1, text2):
    """
    calcul de similarité entre deux text
    """
    vectorizer = TfidfVectorizer(
        tokenizer = normalize,
        stop_words = stop_words,
        ngram_range=(1,1)
    )
    tfidf = vectorizer.fit_transform([text1, text2]) 
    return ((tfidf*tfidf.T).A)[0,1]

text1="C/C++/Java/Python"
text2="Python/Pyspark/machine learning/Microsoft Azure"
import pandas as pd

df = pd.DataFrame()

for t in tech:
    col=[]
    for t1 in tech:
        #print("similarité est: ", cosine_sim(t.replace(' ','_').replace('/',' '), t1.replace(' ','_').replace('/',' ')))
        col.append(cosine_sim(t.replace(' ','_').replace('/',' '), t1.replace(' ','_').replace('/',' ')))
    df[t]=col
#print(df)        

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#print(df.corr())
sns.heatmap(df.corr())

#print('end')
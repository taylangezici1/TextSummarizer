import nltk
import glob
import os
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# The corpus files should be in here...
path = ''
# The corpus files should be in here...

text = ""
for infile in glob.glob(os.path.join(path, '*.txt')):
    
    print("The file being processed is: " + infile)
    with open(infile,'r') as func:
        text += func.read()

text = re.sub(r'\[[0-9]*,\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'[@#\$£&%\*\(\)\<\>\?\'\",:;\]\[-]',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

dataset = nltk.sent_tokenize(text)

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
# Word2Vec modelini eğitme
model = Word2Vec(sentences, min_count=1)

words = model.wv.vocab

# Finding word vectors
vector = model.wv["thank"]

# Word similarities
similar = model.wv.most_similar('thank')

for word in words:
    matrix = model.wv[words.keys()]

longmatrix = np.fromstring(matrix.tostring(),dtype='float32')

words_str = str(words.keys())
words_str = re.sub(r'[@#\$£&%\*\(\)\<\>\?\'\",:;\]\[-]',' ',words_str)
words_str = re.sub(r'dict_keys','',words_str)

main = np.array_split(longmatrix, len(words_str.split()))
dataset= ""
for i in range(0,len(words_str.split())):
    dataset += words_str.split()[i]
    dataset += str(main[i])
dataset = re.sub(r'[@#\$£&%\*\(\)\<\>\?\'\",:;\]\[]',' ',dataset)

newstring = ""
for i in dataset.split():
    try:
        newstring += "{:.6g}".format(float(i)) + " "
    except:
        newstring += "\n" + i + " "
        continue
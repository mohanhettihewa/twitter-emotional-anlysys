import nltk
import tweepy
import numpy as np
import matplotlib.pyplot as plt
try:
    import json
except ImportError:
    import simplejson as json
import re
from collections import Counter

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

auth = tweepy.OAuthHandler('dHNE1hqozwJImPvyhXCxM6jve', 'naKLKndOoeVHSXFVPZdXcBZyPqKuwmDJN2gt2SddtRV495K0kN')
auth.set_access_token('3272116838-lVsSd2Cm7lkhi4cXTIduAV1StAQ6t7O9yGO1sJB', 'zRLD4KIAgiRNNIWeuwdQ0fPfSnj86NPBlUAjdRaNzQuIc')

api = tweepy.API(auth)


data={}
data['tt']=[]
for status in tweepy.Cursor(api.user_timeline, screen_name='@MahelaJay').items(100):
      data['tt'].append(status._json)

titles = [] 
for word in data['tt']:
     titles.append(word['text'])

# copy tokenizer from sentiment example
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
# add more stopwords specific to this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })
def my_tokenizer(s):
  
    url_reg  = r'[a-z]*[:.]+\S+'
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    txt = emoji_pattern.sub(r'', s)
    txt   = re.sub(url_reg, '', txt)    #remove urls
    txt=txt.lower()
    import string
    table = str.maketrans('', '', string.punctuation)
    tokens = nltk.tokenize.word_tokenize(txt) # split string into words (tokens)
    import string
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    #tokens=[stemmer.stem(w)  for w in tokens]
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"

    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
x = 0
all_tokens = []
all_titles = []
test_data=[]
index_word_map = []
for title in titles:
    
        # title = title.encode('ascii', 'ignore') # this will throw exception if bad characters
        
        #tokens = my_tokenizer(title)
        tokens = my_tokenizer(title)

        all_tokens.extend(tokens)

        all_titles.append(tokens)
        thefile = open('Texxt%s.txt'%x, 'w')
       
        for token in tokens:
           
            thefile.write("%s\n"% token)
           
        x=x+1
    
        
       
thefile1 = open('Trump.txt', 'w')
for item in all_tokens:
           
    thefile1.write("%s\n"% item)
   
 

print(all_tokens)
# now let's create our input matrices - just indicator variabes for this example - works better than proportions



  

from __future__ import division
import datetime, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import nltk
from nltk.corpus import reuters
from nltk.stem.snowball import SnowballStemmer
import os

fname = "/Users/adityarai/Desktop/INPUT_ARTICLE.txt"
with open(fname) as f:
    article0 = f.readlines()

article=" "
for sent in article0:
    article = article+ str(sent)
article0 = nltk.sent_tokenize(article)

# article is the string representation of the entire input



def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# parsing data from the synonym documant to list of words
#fname1 = "/Users/adityarai/Desktop/annual reports/keywords.txt"

#with open(fname1) as f1:
#   listSyn = f1.readlines()




inputdir = "/Users/adityarai/Desktop/Axis Bank/Management Discussion"

token_dict={}
file_article=[]
i=0
for file in os.listdir(inputdir):
    fpath = inputdir+"/"+file ;
    if(fpath != inputdir+"/"+".DS_Store"):
        with open(fpath,'rb') as f1:
            
            file_article = f1.readlines()
       
        article_str=" "

        for sent in file_article:
            article_str = article_str+ str(sent)
        
        token_dict[i]= article_str
       
        i=i+1






        
tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english', decode_error='ignore')
print('building term-document matrix... [process started: ' + str(datetime.datetime.now()) + ']')
sys.stdout.flush()

tdm = tfidf.fit_transform(token_dict.values()) # this can take some time (about 60 seconds on my machine)
print('done! [process finished: ' + str(datetime.datetime.now()) + ']')


from random import randint

feature_names = tfidf.get_feature_names()
print('TDM contains ' + str(len(feature_names)) + ' terms and ' + str(tdm.shape[0]) + ' documents')

print('first term: ' + feature_names[0])
print('last term: ' + feature_names[len(feature_names) - 1])

for i in range(0, 4):
    print('random term: ' + feature_names[randint(1,len(feature_names) - 2)])




index = 0
lookup = {}
for sentence in article0:
    lookup[sentence] = index;
    index = index + 1

index1 = 0
reverse_lookup = {}
for sentence in article0:
    reverse_lookup[index1] = sentence;
    index1 = index1 + 1


sent_count=0
article_id = 0
sent_scores = []
for sentence in article0:
    
    score = 0
    sent_tokens = tokenize_and_stem(sentence)

    for token in (t for t in sent_tokens if t in feature_names):
        
        score += tdm[article_id, feature_names.index(token)]

    sent_scores.append((score / len(sent_tokens), sentence))

summary_length = int(math.ceil(len(sent_scores) / 5))
sent_scores.sort(key=lambda sent: sent[0], reverse=True)

final_sent = []


for summary_sentence in sent_scores[:summary_length]:
    final_sent.append(summary_sentence[1])

sent_indexes = []
for sent in final_sent:
    sent_indexes.append(lookup[sent])

sent_indexes.sort();



print('\n*** ORIGINAL ***')
print("TOTAL SENTENCES IN DOCUMENT :" , len(article0)," sentences\n")
print(article0)


print('\n*** SUMMARY ***')
print("SUMMARIZED IN :" , len(sent_indexes)," sentences\n")
for val in sent_indexes:
    print(reverse_lookup[val])
##################################################################
print("\n\n\n\nHIGHLIGHTED SENTENCES OUT OF THE ORIGNAL")
for sent in article0:
    if sent in final_sent:
        print("\033[44;33m"+sent+"\033[m")
    else:
        print(sent)

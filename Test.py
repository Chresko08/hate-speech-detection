from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
import sys

twt=""
if(len(sys.argv)==2):
	twt=sys.argv[1]
tweets=[twt]
with open('Resources/TweetBrain','rb') as f:
    hate_speech_detect=pickle.load(f)
f=open('Resources/featurefile.txt','r')
dim=f.read()
f.close()
cv=CountVectorizer(max_features=34000)

wordnet = WordNetLemmatizer()
try :
    df =pd.DataFrame(data = tweets,columns = ["tweet"])
    lis = []
    for i in range(0,len(tweets)):
        review = re.sub('[^a-zA-Z]', ' ',df['tweet'][i])
        review = review.lower()
        review = review.split()
        review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        lis.append(review)
except Exception as e:
    pass
    
inp=[]
inp.append(dim)
l=lis[0]
inp.append(l)
predictionset=cv.fit_transform(inp).toarray()
    #print(predictionset)
y_predion=hate_speech_detect.predict(predictionset)
if y_predion[1] == 1:
    print("Your Text is Hateful!")
else:
    print("Your Text is neutral!")

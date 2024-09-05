#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
tweets = pd.read_csv('tweets.csv')
import re
import nltk


# In[2]:


#Data precrocessing and cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()
corpus = []
for i in range(0,len(tweets)):
    review = re.sub('[^a-zA-Z]', ' ',tweets['tweet'][i])
    review = review.lower()
    review = review.split()
    
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[3]:


# Creating the bagsof model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 34000)
X = cv.fit_transform(corpus).toarray()
y = tweets['label']


# In[ ]:





# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[5]:


from sklearn.naive_bayes import MultinomialNB
hate_speech_detect = MultinomialNB().fit(X_train, y_train)


# In[6]:


import pickle
with open('Resources/TweetBrain','wb') as f:
    pickle.dump(hate_speech_detect,f)


# In[7]:


y_pred=hate_speech_detect.predict(X_test)


# In[8]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_m = confusion_matrix(y_test,y_pred)
print(confusion_m)
print(classification_report(y_test,y_pred))


# In[9]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


# In[14]:


#tweets = ["This shit geat is nothing but a camouflage #Vanity event for the OrangeBigot & his Nazis ! #HATE disgraceful placed in front of the #LincolnMemorial how MORE of a POScan he show us he is ?! This shit has NOTHING TO DO WITH #PATROITIC ways as Rump is TREASON period ! "]
#tweets = ["I had a GREAT week,thanks to YOU! If you need anything, please reach out."]
try :
    tweets =["This shit geat is nothing but a camouflage #Vanity event for the OrangeBigot & his Nazis ! #HATE disgraceful placed in front of the #LincolnMemorial how MORE of a POScan he show us he is ?! This shit has NOTHING TO DO WITH #PATROITIC ways as Rump is TREASON period ! "]
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
    print(e)


# In[15]:


try:
    inp=[]
    feat=cv.get_feature_names()
    dim=""
    for item in feat:
        dim+=item+" "
    inp.append(dim)
#Saving the content of unique words
    featurefile=open('Resources/featurefile.txt','w')
    featurefile.write(dim)
    featurefile.close()
    
    inp.append(*lis)
    predictionset=cv.fit_transform(inp).toarray()
    #print(predictionset)
    y_predion=hate_speech_detect.predict(predictionset)
    print(y_predion)
    if y_predion[1] == 1:
        print("Your Text is Hateful!")
    else:
        print("Your Text is neutral!")
except Exception as e:
    print(e)


# In[ ]:





# In[ ]:






# coding: utf-8

# In[1]:

import json
from pprint import pprint

data0 = json.load(open('reuter0.json'))
data1 = json.load(open('reuter1.json'))
data2 = json.load(open('reuter2.json'))
data3 = json.load(open('reuter3.json'))
data4 = json.load(open('reuter4.json'))
data5 = json.load(open('reuter5.json'))
data6 = json.load(open('reuter6.json'))
data7 = json.load(open('reuter7.json'))
data8 = json.load(open('reuter8.json'))
data9 = json.load(open('reuter9.json'))
data10 = json.load(open('reuter10.json'))
data11 = json.load(open('reuter11.json'))
data12 = json.load(open('reuter12.json'))
data13 = json.load(open('reuter13.json'))
data14 = json.load(open('reuter14.json'))
data15 = json.load(open('reuter15.json'))
data16 = json.load(open('reuter16.json'))
data17 = json.load(open('reuter17.json'))
data18 = json.load(open('reuter18.json'))
data19 = json.load(open('reuter19.json'))
data20 = json.load(open('reuter20.json'))
data21 = json.load(open('reuter21.json'))


# In[2]:

data=data0+data1+data2+data3+data4+data5+data6+data7+data8+data9+data10+data11+data12+data13+data14+data15+data16+data17+data18+data19+data20+data21


# In[3]:

len(data)


# In[4]:

count=0
data_red=[]
for i in range(0,len(data)):
    if 'topics' in data[i] and 'body' in data[i]:
        count=count+1
        data_red.append(data[i])
        


# In[5]:

body=[]
topic=[]
for i in range(0,len(data_red)):
    body.append(data_red[i]["body"])
    topic.append(data_red[i]["topics"])


# In[6]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()


# In[7]:

X = count_vect.fit_transform(body) 
print(X.toarray())
print(count_vect.get_feature_names())


# In[26]:

X


# In[32]:

len(count_vect.inverse_transform(X))


# In[9]:

y=[]
for i in range(0,len(topic)):
    if 'earn' in topic[i]:
        y.append(1)
    else:
        y.append(0)


# In[10]:

[len(y),len(topic)]


# In[11]:

from sklearn.ensemble import RandomForestClassifier


# In[12]:

clf = RandomForestClassifier(n_estimators=50)


# In[13]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[44]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[45]:

clf.fit(X_train, y_train)


# In[46]:

y_pred=clf.predict(X_test)


# In[47]:

from __future__ import division
sum(y_pred==y_test)/len(y_test)


# # Dictionnary bag of words

# In[35]:

import collections, re
bagsofwords = [ collections.Counter(re.findall(r'\w+', txt)) for txt in body]


# In[ ]:




# # Feature Hashing

# In[25]:

from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=1000)


# In[39]:

X_hash=h.transform(bagsofwords)


# In[48]:

X_train, X_test, y_train, y_test = train_test_split(X_hash, y, test_size=0.2, random_state=0)


# In[49]:

clf.fit(X_train, y_train)


# In[50]:

y_pred=clf.predict(X_test)


# In[51]:

from __future__ import division
sum(y_pred==y_test)/len(y_test)


# # Hash Image

# In[186]:

from PIL import Image, ImageFile
import imagehash
hasher=[]
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[187]:

hash1=imagehash.average_hash(Image.open('mickey1.jpg')
hash2=imagehash.average_hash(Image.open('mickey2.jpg')


# In[179]:

import hashlib
im1 = Image.open('mickey1.jpg')
im1.resize((10, 10), Image.ANTIALIAS) 
im1.convert("L")  
hash1=hashlib.md5(im1.tobytes()).hexdigest()
im2 = Image.open('mickey2.jpg')
im2 = im2.resize((30, 30), Image.ANTIALIAS) 
im2 = im2.convert("L")  
hash2=hashlib.md5(im2.tobytes()).hexdigest()
im3 = Image.open('pingouin1.png')
im3 = im3.resize((30, 30), Image.ANTIALIAS) 
im3 = im3.convert("L")  
hash3=hashlib.md5(im3.tobytes()).hexdigest()
im4 = Image.open('pingouin2.png')
im4 = im4.resize((30, 30), Image.ANTIALIAS) 
im4 = im4.convert("L")  
hash4=hashlib.md5(im4.tobytes()).hexdigest()
im5 = Image.open('mickey1.jpg')
im5 = im5.resize((30, 30), Image.ANTIALIAS) 
im5 = im5.convert("L")  
hash5=hashlib.md5(im5.tobytes()).hexdigest()


# In[154]:

def hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs


# In[175]:

hamdist(hash3,hash4)


# In[183]:

img = Image.open('mickey1.jpg').convert('L')
new_img = img.resize((10,10))


# In[184]:

new_img


import pandas as pd
import nltk

df=pd.read_csv("spam.csv",encoding="latin-1")

df.head()

df.shape


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)

df.rename(columns={'v1':'class', 'v2':'sms'},inplace=True)
df.head()

df.groupby('class').describe()

df=df.drop_duplicates(keep='first')

df.groupby('class').describe()

df["Length"]=df["sms"].apply(len)

df.head(2)

df.hist(column='Length',by='class',bins=50)

from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('punkit')
ps=PorterStemmer()

df.head()

#Preprocessing task
  # Lower case
  #Tokenization
  #Removing special characters
  #Removing stop words and punctuation
  #stemmin



from pandas.io.parsers.readers import TextFileReader
import string

def clean_text(text):
   text= text.lower()
   text=nltk.wordpunct_tokenize(text)

   y=[]
   for i in text:
     if i.isalnum():
       y.append(i)

   text= y[:]
   y.clear()

   for i in text:
    if i not in stopwords.words('english') and string.punctuation:
      y.append(i)

   text=y[:]
   y.clear()

   for i in text:
    y.append(ps.stem(i))

   return " ".join(y)

df['sms_cleaned']= df['sms'].apply(clean_text)

df.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tf_vec=TfidfVectorizer(max_features=3000)
x=tf_vec.fit_transform(df['sms_cleaned']).toarray()

x.shape

y=df['class'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=2)

from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB()
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))


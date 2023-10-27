import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string

# Loading the required things
tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing function
# Converting to lowercase

def text_transfer(x):
  text = x.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

# Title
st.title("Email/SMS Spam Classifier")

# Text Box
sms = st.text_area("Enter the message")

if st.button("Predict"):
  # Preprocess
  transformed_sms = text_transfer(sms)
  # Vectorize
  vector_input = tf.transform([transformed_sms])
  # Result
  result = model.predict(vector_input)[0]
  # Display
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")
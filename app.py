import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
import tensorflow as tf
print(tf.__version__) 
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

nltk.download('stopwords')
nltk.download('punkt_tab')


new_model = tf.keras.models.load_model('model.keras')


from nltk.tokenize import word_tokenize
def clean_sentence(sentence: str) -> list:
  # Remove the review tag
  tags = re.compile("(|<\/review_text>)")
  sentence = re.sub(tags, '', sentence)

  # lower case
  sentence = sentence.lower()

  # Remove emails and urls
  email_urls = re.compile("(\bhttp.+? | \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)")
  sentence = re.sub(email_urls, '', sentence)

  # Some used '@' to hide offensive words (bla -> bl@)
  ats = re.compile('@')
  sentence = re.sub(ats, 'a', sentence)

  # Remove Punctuation
  # punc = re.compile("[!\"\#$\%\&\'\*\+,\-\.\/\:;<=>\?
  punc = re.compile("[^\w\s(\w+\-\w+)]")
  sentence = re.sub(punc, '', sentence)

  # Remove stopwords and tokenize
  # sentence = sentence.split(sep=' ')
  sentence = word_tokenize(sentence)
  sentence = [word for word in sentence if not word in stopwords.words()]

  return sentence



MAX_SEQ_LEN = 125
import pickle
with open("x_vocab", "rb") as fp:   # Unpickling
  x_train = pickle.load(fp)

vocab = set()
for sentence in x_train:
  for word in sentence:
    vocab.add(word)

vocab.add('') # for dummy words, to avoid adding a word that has a meaning
print("Vocab size:", len(vocab))

# Make a mapping betwween words and their IDs
word2id = {word:id for  id, word in enumerate(vocab)}
id2word = {id:word for  id, word in enumerate(vocab)}
dummy = word2id['']

def encode_sentence(old_sentence):
  encoded_sentence = []
  dummy = word2id['']
  for word in old_sentence:
    try:
      encoded_sentence.append(word2id[word])
    except KeyError:
      encoded_sentence.append(dummy) # the none char

  return encoded_sentence


def lstm_predict(sentence:str):
  prediction = 0
  sentence = clean_sentence(sentence)
  # Encode sentence
  ready_sentence = encode_sentence(sentence)
  # Padding sentence
  ready_sentence = pad_sequences(sequences = [ready_sentence],
                                 maxlen=MAX_SEQ_LEN,
                                 dtype='int32',
                                 padding='post',
                                 truncating='post',
                                 value = dummy)

  # Predict
  prediction = round(new_model.predict(ready_sentence)[0][0])
  if prediction==0:
    print("Negative Review")
    return "Negative Review"
  elif prediction==1:
    print("Positive Review")
    return "Positive Review"
  else:
    print('Error')



@app.route("/")
def Home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def predict():
    # float_features = [float(x) for x in request.form.values()]
    # features = [np.array(float_features)]
    # prediction = model.predict(features)'
    val = str(request.form.values())
    return render_template("home.html", prediction_text = lstm_predict(val))

if __name__ == "__main__":
    app.run(debug=True)
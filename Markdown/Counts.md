

```
import nltk 
nltk.download('punkt')

import nltk
nltk.download('averaged_perceptron_tagger')

import nltk
nltk.download('wordnet')
```


```
text = open("Airgas Inc2009.txt").read()
```


```
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

sentences = sent_tokenize(text)
words = word_tokenize(text)

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

input_file = "Airgas Inc2009.txt"
words_file = "words.txt"
output_file = "output.txt"
curriculum_words = []
pos_tagged_array = []
base_words = []
wordnet_lemmatizer = WordNetLemmatizer()

def process_words_file():
    """
    Read the words file and store the words in a list
    """
    global curriculum_words
    curriculum_words = []
    with open(words_file) as curriculum_file:
        try:
            for line in curriculum_file:
                curriculum_words.append(line.strip())
        except Exception as e:
            print (e)


def process_input():
    """
    Read the input file and tokenize and POS_tag the words
    """
    global pos_tagged_array
    with open(input_file) as input_text:
        try:
            for line in input_text:
                if line.strip():
                    words = word_tokenize(line)
                    for tag in nltk.pos_tag(words):
                        # eliminating unnecessary POS tags
                        match = re.search('\w.*', tag[1])
                        if match:
                            pos_tagged_array.append(tag)
        except Exception as e:
            print (e)


def lemmatize_words():
    """
    Convert each word in the input to its base form
    and save it in a list
    """
    global base_words
    for tag in pos_tagged_array:
        base_word = tag[0].lower()
        base_words.append(base_word)


def analyze_input():
    """
    Find count of words from the curriculum_words list
    in the base_words list
    """
    output = open(output_file, 'w')
    for curriculum_word in curriculum_words:
        count = base_words.count(curriculum_word)
        output.write("%-15s | %10s\n" % (curriculum_word, str(count)))
    output.close()


process_words_file()
process_input()
lemmatize_words()
analyze_input()
```


```
from textblob import TextBlob
words = "cat dog child goose pants"
blob = TextBlob(ra)
plurals = [word.pluralize() for word in blob.words]
```

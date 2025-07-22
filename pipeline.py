import pickle
import numpy as np
import pandas as pd

# Load preprocessed data dan vektor dokumen
df = pd.read_csv('Result/All_Process/Test_Data.csv')  # termasuk kolom 'Berita' dan 'word2vec_vector'
print("DataFrame Loaded")


# Load Neural Network Model
with open('Result/without_val/NeuralNetwork_Model_Z2.pkl', 'rb') as f:
    nn_model_Z2 = pickle.load(f)
    print("Neural Network Model Loaded")

# Preprocessing manual (sesuai pipeline)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
exception_words = {"pendidik", "pengasuh", "pembantu"}

def stem_words(words):
    stemmed_words = []
    for word in words:
        if word in exception_words:  # Check if the word is in the exception list
            stemmed_words.append(word)  # Keep the word as is
        else:
            stemmed_words.append(stemmer.stem(word))  # Stem other words
    return stemmed_words


def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text) # remove punctuation
    lower_text = cleaned_text.lower()  # case folding
    tokens = word_tokenize(lower_text) # tokenization
    tokens_no_stopword = [word for word in tokens if word not in stop_words] # stopword removal
    stemmed_tokens = stem_words(tokens_no_stopword) # stemming
    return stemmed_tokens

def parse_vector(vector_str):
    return np.array(list(map(float, vector_str.strip("[]").split(','))))

# Fungsi pipeline lengkap
def pipeline_from_row(index):
    sentence = df['Berita'][index]  
    preprocessed_tokens = clean_text(sentence)  # hasil tokenisasi + stemming
    preprocessed_text = ' '.join(preprocessed_tokens)
    vector_str = df['Word2Vec Vector'][index]

    word2vec_vector = parse_vector(vector_str).reshape(-1, 1)  # untuk ngubah ke bentuk (n,1) (2 dimensi)

    _, P, p_activation, Q, q_activation = nn_model_Z2.forward(word2vec_vector)
    prediction = np.argmax(q_activation).item()

    result = {
        "success": True,
        "model": "Neural Network - Z2",
        "data": {
            'sentence': sentence,
            'preprocessed' : preprocessed_text,
            'vector': word2vec_vector.flatten(), # untuk ngubah ke bentuk (n,)  [0.25, 0.14, 0.09]
            'nn_output': {
                'X': _.tolist(),
                'P': P.tolist(),
                'p_activation': p_activation.tolist(),
                'Q': Q.tolist(),
                'q_activation': q_activation.tolist()
            },
            'prediction': prediction
        }
    }
    return result

# Contoh pemanggilan
if __name__ == "__main__":
    index = 103  # Misal ambil baris ke-0
    result = pipeline_from_row(index)
    # print("Prediction:", result["data"]["prediction"])
    # print("Kalimat preprocessed:", result["data"]["sentence"])
    print("Result =", result)

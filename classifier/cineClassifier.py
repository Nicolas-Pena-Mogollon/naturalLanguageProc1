import warnings
import pickle
import pandas as pd
import nltk
import unicodedata

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# Read data from Google Sheet: experimentos.analitica.datos - EncuestaCineColombiano_Respuestas
df = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQC3CXrmRk6mpK9-DrpO--faGVT_KsR8rj-AToUfFlsbKNnUB2wVslmNYiFT1pv80Z5gp76tgSqp1aN/pub?gid=1802142849&single=true&output=tsv",
    sep="\t")

df.columns = ['A', 'B', 'C', 'D', 'E']

# Transform dataset: Opinion-Type
good_df = df[['C']]
good_df['Opinion'] = "POSITIVE"
bad_df = df[['D']]
bad_df.columns = ['C']
bad_df['Opinion'] = "NEGATIVE"
df_op = pd.concat([good_df, bad_df])
df_op.columns = ['Opinion', 'Type']

df_op.groupby(['Type']).count()

nltk.download('stopwords')
stemmer = SnowballStemmer('spanish')
nltk.download('punkt')

stop_words = set(stopwords.words('spanish'))
stop_words = stop_words.union(
    set(['pelicul', 'colombian', 'cin', 'me', 'le', 'da', 'mi', 'su', 'ha', 'he', 'ya', 'un', 'una', 'es', 'del', 'las',
         'los', 'en', 'que', 'y', 'la', 'de']))


def remove_accents(input_str):
    """
    Remueve los acentos de un texto
    :param input_str: texto con acentos
    :return: texto sin acentos
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def fast_preproc(text):
    """
    Preprocesa el texto antes de entrenar el modelo
    :param text: texto a preprocesar
    :return: texto preprocesado
    """
    text = text.lower()
    text = ''.join(c for c in text if not c.isdigit())
    text = ''.join(c for c in text if c not in punctuation)
    text = remove_accents(text)
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if not word in stop_words]
    try:
        text = " ".join(str(word) for word in words)
    except Exception as e:
        print(e)
        pass
    return text


df_op['Opinion'] = df_op['Opinion'].astype(str)

df_op = df_op.assign(
    TextPreproc=lambda df: df_op.Opinion.apply(fast_preproc)
)

df_op.head()

# Split dataset
X = df_op['TextPreproc']
Y = df_op['Type']

vec = TfidfVectorizer(max_df=0.5)

# Tokenize and build vocabulary
vec.fit(X)

# Encode documents
trans_text_train = vec.transform(X)

# Print Document-Term Matrix
df = pd.DataFrame(trans_text_train.toarray(), columns=vec.get_feature_names_out())

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.1)

# Machine learning algorithms used in Text Classification
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)


def serialize_objects():
    """
    Serializa los objetos
    :return: None
    """
    with open("../files/vec.pkl", "wb") as archivo1:
        pickle.dump(vec, archivo1)

    with open("../files/classifier.pkl", "wb") as archivo2:
        pickle.dump(classifier, archivo2)


# Para Testear el modelo

def test_model():
    """
    Testea el modelo entrenado
    :return:
    """
    # Predice para el conjunto de testeo.
    y_pred = classifier.predict(X_test)

    print("\nMatriz de confusión:\n")
    print(confusion_matrix(Y_test, y_pred))

    print("\nEstadisticas del clasificador:\n")
    print(classification_report(Y_test, y_pred))

    new_opinion = ["Me parecen malas por la razon de que no llevan un hilo conductor claro"]
    # new_opinion = ["Me gustan porque son chistosas"]

    Xt_new = [fast_preproc(str(new_opinion))]

    trans_new_doc = vec.transform(Xt_new)  # Use same TfIdfVectorizer

    print("\nPredicted result: " + str(classifier.predict(trans_new_doc)))

# serialize_objects()
# test_model()

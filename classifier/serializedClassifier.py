import pickle

from cineClassifier import fast_preproc

with open("../files/vec.pkl", "rb") as archivo1:
    vec = pickle.load(archivo1)

with open("../files/classifier.pkl", "rb") as archivo2:
    classifier = pickle.load(archivo2)

new_opinion = ["rfgdyihsajcoasbncabncbnaos"]
Xt_new = [fast_preproc(str(new_opinion))]
trans_new_doc = vec.transform(Xt_new)  # Use same TfIdfVectorizer
prediction = classifier.predict(trans_new_doc)
probabilidad = classifier.predict_proba(trans_new_doc)

print(probabilidad.max())

if prediction[0] == "POSITIVE" and probabilidad.max() > 0.54:
    print("Muchas gracias por sus buenas opiniones!")
elif probabilidad.max() > 0.54:
    print("Muchas gracias por su opinión, intentaremos mejorar")
else:
    print("No ha sido posible procesar su opinión, intente con una mejor descripción")
print("\nPredicted result: " + str(prediction))

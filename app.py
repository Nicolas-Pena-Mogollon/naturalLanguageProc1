import pickle

from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from classifier.cineClassifier import fast_preproc

# Configura tus credenciales de Twilio
account_sid = 'AC6de246b948746b8beb54c27115eb4ad0'
auth_token = '2868c62cde71d7a0ea5866a683587936'
client = Client(account_sid, auth_token)

app = Flask(__name__)
with open("./files/vec.pkl", "rb") as archivo1:
    vec = pickle.load(archivo1)

with open("./files/classifier.pkl", "rb") as archivo2:
    classifier = pickle.load(archivo2)


@app.route('/webhook', methods=['POST'])
def webhook():
    processed_request = process_request(request.values.get('Body'))

    response = MessagingResponse()
    response.message(processed_request)
    return str(response)


def process_request(message):
    """
    Procesa el mensaje y retorna una respuesta del modelo entrenado
    :param message: mensaje a procesar
    :return: respuesta del modelo
    """
    if len(message) > 20:
        Xt_new = [fast_preproc(str(message))]
        trans_new_doc = vec.transform(Xt_new)  # Utiliza el mismo TfIdfVectorizer
        accuracy = classifier.predict_proba(trans_new_doc)
        # accuracy.max() obtiene el valor de la presición (probabilidad de pertenencia a una clase)
        # Al ser de testeo el valor es bajo
        if classifier.predict(trans_new_doc)[0] == "POSITIVE" and accuracy.max() > 0.54:
            return "Muchas gracias por sus buenas opiniones!"
        elif accuracy.max() > 0.54:
            return "Muchas gracias por su opinión, intentaremos mejorar"
        else:
            return "No ha sido posible procesar su opinión, intente con una mejor descripción"

    return "Tu opinión debe tener más de 20 caracteres."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

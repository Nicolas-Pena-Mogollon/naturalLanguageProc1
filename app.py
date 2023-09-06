from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

# Configura tus credenciales de Twilio
account_sid = 'AC6de246b948746b8beb54c27115eb4ad0'
auth_token = '2868c62cde71d7a0ea5866a683587936'
client = Client(account_sid, auth_token)

app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    # Procesa el mensaje entrante de WhatsApp
    mensaje_entrante = request.values.get('Body')

    # Aquí puedes procesar el mensaje entrante y generar una respuesta
    respuesta = 'Has dicho: ' + mensaje_entrante

    # Crea una respuesta de TwiML
    response = MessagingResponse()
    response.message(respuesta)

    return str(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

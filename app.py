import os
import time
from flask import Flask, request, render_template, send_file
import torch

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, Xtts, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


print("Cargando modelo de TTS (Coqui)...")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("Modelo de TTS cargado.")

app = Flask(__name__)

respuestas_definidas = {
    "hola": "Hola, qué bueno que estás aquí. Soy un agente conversacional listo para responder tus preguntas. ¿En qué te puedo ayudar hoy?",
    "que tal": "¡Todo de maravilla! Mis sistemas están funcionando perfectamente y estoy listo para procesar cualquier consulta que tengas para mí. ¡Adelante, pregunta lo que quieras!",
    "como te llamas": "No tengo un nombre propio como tal. Puedes pensar en mí como una interfaz conversacional. Fui diseñado para ayudarte, utilizando un modelo de lenguaje y un sintetizador de voz para interactuar contigo.",
    "que puedes hacer": "Actualmente, mi principal función es responder a una serie de preguntas específicas que tengo programadas, utilizando esta voz que estás escuchando. Aunque mis capacidades son limitadas por ahora, estoy construido sobre una arquitectura que podría expandirse para buscar información en tiempo real.",
    "gracias": "¡De nada! Ha sido un placer ayudarte. Si tienes alguna otra pregunta o necesitas algo más, no dudes en consultarme. Estoy aquí para servirte.",
    "cual es el clima en el alto": "Esa es una excelente pregunta. Sin embargo, en esta versión simplificada, mi conexión para buscar datos en tiempo real como el clima ha sido desactivada. Por lo tanto, no puedo darte la temperatura actual en El Alto."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/preguntar', methods=['POST'])
def preguntar_agente():
    pregunta_usuario = request.form['pregunta'].lower().strip()
    print(f"Pregunta recibida: '{pregunta_usuario}'")
    
    respuesta_texto = respuestas_definidas.get(pregunta_usuario, "Lo siento, no entiendo esa pregunta. Mis capacidades son limitadas por ahora.")
    print(f"Respuesta generada: '{respuesta_texto}'")

 
    archivo_salida = f"respuesta_{int(time.time())}.wav"
    
    tts_model.tts_to_file(
        text=respuesta_texto,
        speaker_wav="voz_referencia.wav",
        language="es",
        file_path=archivo_salida
    )
    print(f"Audio generado y guardado como '{archivo_salida}'")

    return send_file(archivo_salida, mimetype='audio/wav')

if __name__ == '__main__':
    
    print("Modelo cargado. Iniciando servidor Flask...")
    app.run(debug=True, port=5000)
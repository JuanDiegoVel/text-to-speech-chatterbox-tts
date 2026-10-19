
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto = (
    "Hola, bienvenidos a Inteligencia Artificial Generativa. "
    "Vamos a crear audios, imágenes, texto y, si el PC aguanta, videos."
)

print("Generando audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7,
    cfg_weight=0.4
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado en audio.wav")

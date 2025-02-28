# stt.py
import os
import time
import queue
import threading
import tempfile
import requests
import wave
import logging
import openai
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY")

class WhisperStreamingSTT:
    def __init__(self, on_text, session_id):
        self.on_text = on_text
        self.session_id = session_id
        self.audio_queue = queue.Queue()
        self.stop_flag = False

        # Accumule en RAM
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def start(self):
        # Lance un thread pour périodiquement vider le buffer
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_flag = True
        if self._thread.is_alive():
            self._thread.join()

    def push_audio(self, chunk_pcm):
        with self._lock:
            self._buffer.extend(chunk_pcm)

    def _run(self):
        # On envoie un chunk toutes les 5s
        CHUNK_INTERVAL = 5.0
        last_time = time.time()

        while not self.stop_flag:
            time.sleep(0.1)
            now = time.time()
            if (now - last_time) >= CHUNK_INTERVAL:
                # récup le buffer, le clear
                with self._lock:
                    data = bytes(self._buffer)
                    self._buffer.clear()
                if len(data) > 1024:
                    # On fait la transcription
                    text = self._transcribe_chunk(data)
                    if text.strip():
                        self.on_text(self.session_id, text)
                last_time = now

    def _transcribe_chunk(self, pcm_data: bytes) -> str:
        """
        Convertit un buffer PCM 16 bits en .wav, 
        l'envoie à openai.Audio.transcriptions.create(model="whisper-1")
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
                # On va générer un WAV standard 16 bits / 16kHz
                # pydub le fera. Sinon wave direct, but we must set params.
                audio_seg = AudioSegment(
                    data=pcm_data,
                    sample_width=2,
                    frame_rate=8000,  # Twilio 8k
                    channels=1
                )
                # convert to 16k for better whisper accuracy
                audio_seg_16k = audio_seg.set_frame_rate(16000)
                audio_seg_16k.export(tmp_name, format="wav")

            with open(tmp_name, "rb") as f:
                resp = openai.Audio.transcribe(
                    file=f,
                    model="whisper-1",
                    response_format="text",
                    language="fr",  # ex: for French if you want
                )
            text = resp
            return text
        except Exception as e:
            logging.error(f"STT chunk error: {e}")
            return ""
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)


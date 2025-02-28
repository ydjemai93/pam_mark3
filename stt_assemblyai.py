# stt_assemblyai.py

import os
import logging
import assemblyai as aai
import threading

class AssemblyAIStreamingSTT:
    """
    Gère la connexion streaming à AssemblyAI:
      - start() => ouvre la connexion
      - send_audio(chunk) => envoie l'audio
      - callbacks on_partial / on_final
    """

    def __init__(self, on_partial, on_final):
        self.on_partial = on_partial
        self.on_final = on_final
        self.stop_flag = False
        self.transcriber = None
        self._thread = None

        # Config AssemblyAI
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

        self.transcriber = aai.RealtimeTranscriber(
            # Choix du sample rate, ex 8000 => Twilio
            sample_rate=8000,
            encoding=aai.AudioEncoding.pcm_s16le,  # ou aai.AudioEncoding.pcm_mulaw si chunk_ulaw
            # Callbacks
            on_error=self._on_error,
            on_close=self._on_close,
            on_message=self._on_message,
            end_utterance_silence_threshold=700,  # ms
            word_boost=[],  # si besoin
            # ...
        )

    def start(self):
        # On lance le transcriber dans un thread
        def run_transcriber():
            try:
                self.transcriber.start()
            except Exception as e:
                logging.error(f"AssemblyAI transcriber error: {e}")

        self._thread = threading.Thread(target=run_transcriber, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_flag = True
        if self.transcriber:
            self.transcriber.close()
        if self._thread:
            self._thread.join()

    def send_audio(self, pcm_data: bytes):
        """
        Envoie un chunk PCM16 (8kHz) à la transcriber
        """
        if not self.stop_flag and self.transcriber:
            self.transcriber.send(pcm_data)

    def _on_error(self, error_msg: str):
        logging.error(f"[AssemblyAI] error: {error_msg}")

    def _on_close(self):
        logging.info("[AssemblyAI] streaming closed")

    def _on_message(self, msg):
        """
        Reçoit un objet RealtimeTranscript,
        contient is_final et text
        """
        if msg.message_type == aai.RealtimeMessageType.Transcript:
            if msg.is_final:
                # final
                text = msg.text
                if self.on_final:
                    self.on_final(text)
            else:
                # partial
                if self.on_partial:
                    self.on_partial(msg.text)

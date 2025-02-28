import os
import logging
import assemblyai as aai
import threading

class AssemblyAIStreamingSTT:
    """
    Gère la connexion streaming à AssemblyAI:
      - start() => ouvre la connexion
      - send_audio(chunk) => envoie l'audio
      - on_partial(text) et on_final(text) pour gérer la transcription
    """

    def __init__(self, on_partial, on_final):
        self.on_partial = on_partial
        self.on_final = on_final
        self.stop_flag = False
        self.transcriber = None
        self._thread = None

        # Configure la clé
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

        # Définir la config du flux
        config = aai.TranscriptionConfig(
            sample_rate=8000,                     # Twilio = 8k
            encoding=aai.AudioEncoding.pcm_s16le, # PCM 16 bits LE
            # word_boost=[],                      # si tu veux un word boost
            # end_utterance_silence_threshold=700 # ms de silence
            # ... plus d'options si besoin
        )

        # Crée le RealtimeTranscriber
        self.transcriber = aai.RealtimeTranscriber(config=config)

        # Associer les callbacks
        self.transcriber.on_result = self._on_result
        self.transcriber.on_error = self._on_error
        self.transcriber.on_close = self._on_close


    def start(self):
        """
        Lance le transcriber dans un thread
        """
        def run():
            try:
                self.transcriber.start()
            except Exception as e:
                logging.error(f"AssemblyAI Transcriber error: {e}")

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Ferme la connexion streaming et stop le thread
        """
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

    # --------------------- NOS PROPRES CALLBACKS -----------------------

    def _on_result(self, response: aai.RealtimeResponse):
        """
        Appelé quand AssemblyAI renvoie un message transcription
        => response.message_type PARTIAL ou FINAL
        => response.transcript : le texte
        """
        if response.message_type == aai.RealtimeMessageType.FINAL:
            # Transcription finale
            text = response.transcript
            if self.on_final:
                self.on_final(text)
        else:
            # PARTIAL
            partial_text = response.transcript
            if self.on_partial:
                self.on_partial(partial_text)

    def _on_error(self, error_msg: str):
        logging.error(f"[AssemblyAI] Error: {error_msg}")

    def _on_close(self):
        logging.info("[AssemblyAI] streaming closed")

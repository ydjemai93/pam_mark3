# stt_assemblyai.py
import os
import logging
import assemblyai as aai
import threading

class AssemblyAIStreamingSTT:
    """
    Gère la connexion streaming à AssemblyAI :
      - start() ouvre la connexion
      - send_audio(chunk) envoie l'audio
      - Les callbacks on_partial(text) et on_final(text) gèrent la transcription.
    """

    def __init__(self, on_partial, on_final):
        self.on_partial = on_partial
        self.on_final = on_final
        self.stop_flag = False
        self.transcriber = None
        self._thread = None

        # Configure la clé API AssemblyAI
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

        # Créer la configuration du flux sans sample_rate
        config = aai.TranscriptionConfig(
            encoding=aai.AudioEncoding.pcm_s16le  # PCM 16 bits little-endian
            # Vous pouvez ajouter d'autres paramètres ici si nécessaire
        )

        # Instancier le transcriber avec la configuration
        self.transcriber = aai.RealtimeTranscriber(config=config)

        # Assigner les callbacks après l'instanciation (nouvelle API)
        self.transcriber.on_result = self._on_result
        self.transcriber.on_error = self._on_error
        self.transcriber.on_close = self._on_close

    def start(self):
        """Lance le transcriber dans un thread dédié."""
        def run_transcriber():
            try:
                self.transcriber.start()
            except Exception as e:
                logging.error(f"AssemblyAI transcriber error: {e}")

        self._thread = threading.Thread(target=run_transcriber, daemon=True)
        self._thread.start()

    def stop(self):
        """Arrête la session de transcription."""
        self.stop_flag = True
        if self.transcriber:
            self.transcriber.close()
        if self._thread:
            self._thread.join()

    def send_audio(self, pcm_data: bytes):
        """Envoie un chunk PCM16 à AssemblyAI."""
        if not self.stop_flag and self.transcriber:
            self.transcriber.send(pcm_data)

    def _on_result(self, response):
        """
        Callback déclenché pour chaque résultat de transcription.
        L'objet response possède des attributs tels que message_type et transcript.
        """
        if response.message_type == aai.RealtimeMessageType.FINAL:
            text = response.transcript
            if self.on_final:
                self.on_final(text)
        else:
            partial_text = response.transcript
            if self.on_partial:
                self.on_partial(partial_text)

    def _on_error(self, error_msg: str):
        logging.error(f"[AssemblyAI] Error: {error_msg}")

    def _on_close(self):
        logging.info("[AssemblyAI] Streaming closed")

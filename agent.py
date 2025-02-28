# agent.py
import uuid
import threading
import time
import queue
import logging

from stt import WhisperStreamingSTT
from llm import gpt4_stream
from tts import ElevenLabsStreamer

class StreamingAgent:
    """
    Gère plusieurs sessions (plusieurs appels).
    Sur Twilio inbound audio -> STT -> GPT -> TTS -> Twilio outbound
    """
    def __init__(self):
        self.sessions = {}  # session_id -> {...}

    def start_session(self, ws):
        session_id = str(uuid.uuid4())
        sess = {
            "ws": ws,
            "stop_flag": False,
            "speaking": False,
            "interrupt": False,
        }

        # On crée un STT worker
        stt = WhisperStreamingSTT(on_text=self.on_user_text, session_id=session_id)
        sess["stt"] = stt

        # On crée un TTS (ElevenLabs) streamer
        tts = ElevenLabsStreamer(
            on_audio_chunk=lambda pcm: self.send_audio_chunk(session_id, pcm)
        )
        sess["tts"] = tts

        # Historique conversation
        sess["conversation"] = [
            {"role":"system", "content": "You are a helpful assistant. (fr)"},
        ]

        # Lancement du thread STT
        stt.start()

        self.sessions[session_id] = sess
        return session_id

    def end_session(self, session_id):
        sess = self.sessions.get(session_id)
        if not sess:
            return
        sess["stop_flag"] = True
        sess["stt"].stop()
        # on pourrait stopper TTS
        del self.sessions[session_id]

    def on_user_audio_chunk(self, session_id, chunk_pcm):
        sess = self.sessions.get(session_id)
        if not sess:
            return

        # Si IA parle => barge-in
        if sess["speaking"]:
            logging.info(f"[Session {session_id}] Barge-in triggered!")
            sess["interrupt"] = True

        # On envoie ce chunk au STT
        sess["stt"].push_audio(chunk_pcm)

    def on_user_text(self, session_id, text):
        """
        Callback STT: on a un nouveau bloc de transcription
        => on appelle GPT-4 en streaming => TTS => renvoie Twilio
        """
        sess = self.sessions.get(session_id)
        if not sess or sess["stop_flag"]:
            return

        # Ajoute au conversation history
        conversation = sess["conversation"]
        conversation.append({"role":"user","content":text})

        # Lance GPT-4 en streaming dans un thread
        def do_gpt():
            sess["speaking"] = True
            sess["interrupt"] = False
            partial_response = ""

            for token in gpt4_stream(conversation):
                if sess["interrupt"]:
                    logging.info(f"[Session {session_id}] GPT stream interrupted.")
                    break
                partial_response += token
                # On envoie token par token ou batch
                sess["tts"].stream_text(token)  # envoi streaming TTS

            if not sess["interrupt"]:
                # conversation "officielle"
                conversation.append({"role":"assistant","content":partial_response})

            sess["speaking"] = False

        t = threading.Thread(target=do_gpt, daemon=True)
        t.start()

    def send_audio_chunk(self, session_id, pcm_data):
        """
        Recoit un chunk PCM 16bits du TTS, 
        Convertit en ulaw, envoie JSON media sur la WS Twilio
        """
        import audioop
        import base64
        import json

        sess = self.sessions.get(session_id)
        if not sess or sess["stop_flag"]:
            return
        if sess["interrupt"]:
            # Barge-in => on n'envoie plus l'audio
            return

        ws = sess["ws"]
        ulaw = audioop.lin2ulaw(pcm_data, 2)
        b64 = base64.b64encode(ulaw).decode("utf-8")

        ws.send(json.dumps({
            "event":"media",
            "media":{"payload": b64}
        }))


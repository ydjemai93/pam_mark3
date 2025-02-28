# agent.py
import uuid
import logging
import threading

from stt_assemblyai import AssemblyAIStreamingSTT
from llm import gpt4_stream
from tts import ElevenLabsStreamer

class StreamingAgent:
    def __init__(self):
        self.sessions = {}  # session_id -> dict

    def start_session(self, ws):
        session_id = str(uuid.uuid4())
        logging.info(f"[Session {session_id}] start")

        # On crée la structure
        sess = {
            "ws": ws,
            "speaking": False,
            "interrupt": False,
            "conversation": [
                {"role":"system","content":"You are a helpful FR assistant."},
            ]
        }
        # Crée un TTS
        tts = ElevenLabsStreamer(
            on_audio_chunk=lambda pcm: self.send_audio_chunk(session_id, pcm)
        )
        sess["tts"] = tts

        # Crée un STT streaming AssemblyAI
        # On fournit des callbacks partial/final.
        stt = AssemblyAIStreamingSTT(
            on_partial=lambda text: self.on_stt_partial(session_id, text),
            on_final=lambda text: self.on_stt_final(session_id, text)
        )
        sess["stt"] = stt

        # Démarre la session stt
        stt.start()

        self.sessions[session_id] = sess
        return session_id

    def end_session(self, session_id):
        sess = self.sessions.pop(session_id, None)
        if not sess:
            return
        # arrête stt
        sess["stt"].stop()

    def on_user_audio_chunk(self, session_id, chunk_pcm):
        sess = self.sessions.get(session_id)
        if not sess:
            return
        # barge in
        if sess["speaking"]:
            sess["interrupt"] = True
            logging.info(f"[Session {session_id}] barge-in triggered.")

        # envoie le chunk au STT
        stt = sess["stt"]
        stt.send_audio(chunk_pcm)

    def on_stt_partial(self, session_id, text):
        """
        Callback quand AssemblyAI renvoie un partial transcript.
        Optionnel : on peut l'ignorer ou l'afficher
        """
        logging.debug(f"[Session {session_id}] partial: {text}")

    def on_stt_final(self, session_id, text):
        """
        Callback quand AssemblyAI renvoie un final transcript.
        => On appelle GPT-4
        """
        sess = self.sessions.get(session_id)
        if not sess:
            return

        if not text.strip():
            return
        # Ajoute un message "user"
        sess["conversation"].append({"role":"user","content":text})

        # Lance un thread GPT -> TTS
        def run_gpt():
            sess["speaking"] = True
            sess["interrupt"] = False
            partial_resp = ""
            for token in gpt4_stream(sess["conversation"]):
                if sess["interrupt"]:
                    logging.info(f"[Session {session_id}] GPT interrupted.")
                    break
                partial_resp += token
                # on envoie ce token au TTS
                sess["tts"].stream_text(token)
            if not sess["interrupt"]:
                # On finalize la réponse
                sess["conversation"].append({"role":"assistant","content":partial_resp})
            sess["speaking"] = False

        t = threading.Thread(target=run_gpt, daemon=True)
        t.start()

    def send_audio_chunk(self, session_id, pcm_data):
        """
        Reçoit un chunk PCM16 bits du TTS, convertit en ulaw,
        l'envoie sur la websocket Twilio.
        """
        import audioop, base64, json
        sess = self.sessions.get(session_id)
        if not sess:
            return
        if sess["interrupt"]:
            # barge in => stop l'envoi
            return
        ws = sess["ws"]
        ulaw = audioop.lin2ulaw(pcm_data, 2)
        b64 = base64.b64encode(ulaw).decode("utf-8")

        ws.send(json.dumps({
            "event":"media",
            "media":{"payload": b64}
        }))

# twilio_server.py
import os
import json
import base64
import logging
import audioop
from flask import Flask, request
from flask_sock import Sock
import simple_websocket
from gevent.pywsgi import WSGIServer

from agent import StreamingAgent

logging.basicConfig(level=logging.INFO)

XML_MEDIA_STREAM = """
<Response>
  <Start>
    <!-- tracks="both" permet la capture microphone ET le playback, utile pour le barge-in -->
    <Stream url="wss://{host}/audiostream" tracks="both" />
  </Start>
  <Pause length="3600"/>
</Response>
"""

def create_app():
    app = Flask(__name__)
    sock = Sock(app)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "secret")

    # L'agent orchestrateur
    AGENT = StreamingAgent()

    @app.route("/call", methods=["POST"])
    def call():
        # Twilio voice webhook
        host = request.host  # Ex: "xxxx.railway.app"
        return XML_MEDIA_STREAM.format(host=host)

    @sock.route("/audiostream", websocket=True)
    def audiostream(ws):
        """
        Twilio Media Stream endpoint
        """
        session_id = AGENT.start_session(ws)
        logging.info(f"[Session {session_id}] Twilio WebSocket connected")

        try:
            while True:
                message = ws.receive()
                if not message:
                    break
                data = json.loads(message)

                if data["event"] == "start":
                    logging.info(f"[Session {session_id}] Call started")
                elif data["event"] == "media":
                    chunk_ulaw = base64.b64decode(data["media"]["payload"])
                    # Convert ulaw -> 16-bit PCM
                    chunk_pcm = audioop.ulaw2lin(chunk_ulaw, 2)
                    # Envoyer chunk à l'agent
                    AGENT.on_user_audio_chunk(session_id, chunk_pcm)
                elif data["event"] == "stop":
                    logging.info(f"[Session {session_id}] Call ended by Twilio")
                    break
        except simple_websocket.ConnectionClosed:
            logging.info(f"[Session {session_id}] WebSocket closed")
        finally:
            AGENT.end_session(session_id)

        return ""

    return app


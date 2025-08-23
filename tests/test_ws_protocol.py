import json
import pytest
from fastapi.testclient import TestClient
from server.app import app


@pytest.mark.asyncio
async def test_websocket_control_and_flush(monkeypatch):
    client = TestClient(app)

    # Patch handle_utterance to avoid heavy processing
    async def fake_handle_utterance(ws, buf, mode):
        await ws.send_text(json.dumps({"src": "en", "tgt": "ru", "text": "hello"}))
        await ws.send_bytes(b"FAKEWAV")

    monkeypatch.setattr("server.app.handle_utterance", fake_handle_utterance)

    with client.websocket_connect("/ws/translate") as websocket:
        # Send control message
        websocket.send_text(json.dumps({"mode": "en->ru"}))
        # Send some bytes
        websocket.send_bytes(b"1234")
        # Flush
        websocket.send_text("__flush__")

        msg1 = websocket.receive_text()
        assert "hello" in msg1
        msg2 = websocket.receive_bytes()
        assert msg2 == b"FAKEWAV"

# WebSocket speech API

`WS /v1/audio/speech/ws` streams live text input and raw PCM audio. Clients
must offer the `nari.speech.v1` WebSocket subprotocol. Binary server messages
contain mono 24 kHz signed 16-bit little-endian PCM; text messages are JSON
control events.

```javascript
const socket = new WebSocket(
  "ws://127.0.0.1:8000/v1/audio/speech/ws",
  "nari.speech.v1",
);
socket.binaryType = "arraybuffer";

socket.addEventListener("message", ({ data }) => {
  if (typeof data !== "string") {
    console.log(`received ${data.byteLength} bytes of PCM`);
    return;
  }

  const event = JSON.parse(data);
  console.log(event);

  if (event.type === "session.created") {
    socket.send(JSON.stringify({
      type: "request.start",
      model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      voice: "ryan",
      language: "english",
    }));
  } else if (event.type === "request.configured") {
    socket.send(JSON.stringify({
      type: "input_text.append",
      sequence: 0,
      text: "Hello from Nari Labs.",
    }));
    socket.send(JSON.stringify({ type: "input_text.end", sequence: 1 }));
  }
});
```

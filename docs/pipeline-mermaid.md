# Speech-to-Speech Translation Pipeline (Mermaid Diagram)

```mermaid
sequenceDiagram
    participant User as User (Browser)
    participant WS as WebSocket /ws/audio
    participant AST as Canary AST (ASR+Translation)
    participant TTS as Silero TTS
    participant Util as Audio Utils
    participant Out as Output (Browser)

    User->>WS: Send PCM audio chunks (Int16LE)
    WS->>WS: Buffer audio until flush or max size
    WS->>AST: Convert PCM → float32 → Translate (srcLang→dstLang)
    AST-->>WS: Transcript text
    WS->>User: Send transcript JSON {text, filtered}
    alt Suppressed (short/duplicate/stoplist)
        WS-->>User: No TTS audio
    else Valid text
        WS->>TTS: Synthesize speech (dstLang)
        TTS-->>WS: Audio (24kHz float32)
        WS->>Util: Resample to 16kHz mono
        Util-->>WS: Int16LE PCM
        WS->>Out: Send TTS audio bytes
    end
    User-->>User: Play translated audio
```

```mermaid
flowchart TD
    A[User Microphone] -->|PCM chunks| B[WebSocket /ws/audio]
    B -->|Buffer| C[Process & Respond]
    C --> D[Canary AST<br/>ASR + Translation]
    D -->|Transcript| E[Send JSON transcript]
    E -->|Filtered? | F{Suppressed?}
    F -->|Yes| G[Skip TTS]
    F -->|No| H[Silero TTS]
    H --> I[Resample 24kHz → 16kHz]
    I --> J[Convert float32 → Int16LE PCM]
    J --> K[Send audio bytes to client]
    K --> L[User hears translated speech]

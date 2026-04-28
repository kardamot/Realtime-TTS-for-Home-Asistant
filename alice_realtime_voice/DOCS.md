# Alice Realtime Voice

Bu add-on, Alice icin yeni nesil konusma hattinin ilk iskeletidir.

Hedef mimari:

- `ESP -> Realtime Voice Add-on`
- add-on icinde `VAD + STT + LLM + TTS`
- ev cihazlari icin `HA` backend olarak kalir

Bu ilk surumde odak:

- yerel `faster-whisper` tabanli STT giris noktasi
- WebSocket session protokolu
- sonraki fazlar icin LLM / TTS / HA bridge iskeleti

## Port

- Varsayilan port: `8766`
- WebSocket endpoint: `/ws`
- Health endpoint: `/health`

## Simdilik ne yapar?

- ESP benzeri bir istemciden `start` komutu alir
- PCM16 mono ses chunk'larini binary olarak toplar
- `eos` komutunda yerel `faster-whisper` ile metne cevirir
- sonucu tekrar WebSocket uzerinden JSON event olarak doner

## Simdilik ne yapmaz?

- Gercek streaming partial STT
- Tool calling
- Entity cache / HA bridge
- Streaming LLM
- Streaming TTS orkestrasyonu

Bunlar sonraki fazlarda eklenecek.

## Neden dis servis bulmaya gerek yok?

Varsayilan plan yerel STT:

- `faster-whisper`
- CPU ustunde `int8`
- Turkce dil sabitlenebilir

Yani ilk gelisim asamasinda ekstra bir cloud STT secmene gerek yok.

## Ayarlar

### STT

- `model`: `tiny`, `base`, `small`, `medium` vb.
- `language`: Varsayilan `tr`
- `compute_type`: CPU icin genelde `int8`
- `beam_size`: Dusuk tutuldu, gecikme odakli
- `vad_filter`: faster-whisper tarafindaki dahili filtre

### LLM

Bu turda sadece sonraki faz icin yer tutucu bulunur.

### TTS

Bu turda sadece sonraki faz icin yer tutucu bulunur.

## Sonraki faz

1. ESP'yi yeni `/ws` protokolune baglamak
2. Partial STT / endpointing eklemek
3. HA state/tool bridge yazmak
4. Streaming LLM ve streaming TTS zincirini tamamlamak

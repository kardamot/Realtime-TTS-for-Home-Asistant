# Alice Realtime Voice

Bu add-on, Alice icin yeni nesil konusma hattinin ilk iskeletidir.

Surum: `0.4.0`

Hedef mimari:

- `ESP -> Realtime Voice Add-on`
- add-on icinde `VAD + STT + LLM + TTS`
- ev cihazlari icin `HA` backend olarak kalir

Bu fazdaki odak:

- yerel `faster-whisper` tabanli STT giris noktasi
- WebSocket session protokolu
- yerel endpointing / VAD-benzeri eventler
- temel HA bridge komutlari
- sonraki fazlar icin LLM / TTS iskeleti

## Port

- Varsayilan port: `8766`
- WebSocket endpoint: `/ws`
- Health endpoint: `/health`

## Simdilik ne yapar?

- ESP benzeri bir istemciden `start` komutu alir
- PCM16 mono ses chunk'larini binary olarak toplar
- `eos` komutunda yerel `faster-whisper` ile metne cevirir
- sonucu tekrar WebSocket uzerinden JSON event olarak doner
- chunk seviyesinde `vad_start`, `vad_end`, `no_speech_timeout`, `max_utterance_reached` eventleri uretebilir
- `/health` icinde HA bridge durumunu raporlar
- WebSocket uzerinden temel HA komutlari kabul eder:
  - `ha_get_state`
  - `ha_list_states`
  - `ha_call_service`

## Simdilik ne yapmaz?

- Gercek streaming partial STT
- Tool calling
- Gercek entity cache / akilli HA bridge islemleri
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

### Endpointing

Bu fazda basit bir yerel endpointing katmani vardir:

- `start_avg_abs_threshold`
- `end_avg_abs_threshold`
- `speech_start_min_ms`
- `speech_end_silence_ms`
- `no_speech_timeout_ms`
- `max_utterance_ms`

Bu katman su an tam uretim kalitesinde VAD degil; amaci yeni dis voice mimarisi icin konusma baslangic/bitis
eventlerini ve timeout davranisini iskelet olarak kurmaktir.

### HA Bridge

Bu fazda HA bridge artik temel komutlara sahiptir:

- `SUPERVISOR_TOKEN` varsa `http://supervisor/core/api` uzerinden probe yapar
- `/health` cevabinda bridge gorunur
- `ha_get_state` ile tek entity okunabilir
- `ha_list_states` ile domain bazli sade liste alinabilir
- `ha_call_service` ile temel servis cagrisi yapilabilir

Ornek mesajlar:

```json
{"type":"ha_get_state","entity_id":"light.salon"}
```

```json
{"type":"ha_list_states","domain":"light","limit":25}
```

```json
{"type":"ha_call_service","domain":"light","service":"turn_on","data":{"entity_id":"light.salon"}}
```

## Sonraki faz

1. ESP'yi yeni `/ws` protokolune baglamak
2. Partial STT / daha iyi endpointing
3. HA entity cache ve akilli tool bridge yazmak
4. Streaming LLM ve streaming TTS zincirini tamamlamak

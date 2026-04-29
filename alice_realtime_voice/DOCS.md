# Alice Realtime Voice

Bu add-on, Alice icin yeni nesil konusma hattinin ilk iskeletidir.

Surum: `0.9.16`

Hedef mimari:

- `ESP -> Realtime Voice Add-on`
- add-on icinde `OpenAI Realtime VAD + audio understanding + text output`
- TTS icin mevcut dis relay hattindan `Cartesia / ElevenLabs` gibi servisler
- ev cihazlari icin `HA` backend olarak kalir

Bu fazdaki odak:

- yerel `faster-whisper` tabanli STT giris noktasi
- WebSocket session protokolu
- yerel endpointing / VAD-benzeri eventler
- temel HA bridge komutlari
- OpenAI-uyumlu streaming LLM
- ev kontrolu / hava benzeri cumleler icin HA `conversation.process` yonlendirmesi
- mevcut `Alice Realtime TTS` relay uzerinden TTS orkestrasyonu

## Ilk Canli Test Modu

Bu surum, ilk manuel canli entegrasyon testi icin hazirlanan gecis modudur.

- ESP tarafinda yeni `voice` websocket yolu aktif hale getirilecek
- STT/konusma anlama + LLM text output yeni `Alice Realtime Voice` add-on'undan gelecek
- TTS ise simdilik eski `Alice Realtime TTS` add-on'u uzerinden oynatilacak

Bu nedenle ilk testte `Alice Realtime Voice` ayarinda su sekilde kalman onerilir:

```yaml
tts:
  enabled: false
  relay_url: "ws://127.0.0.1:8765/ws"
```

Yani yeni voice add-on kendi websocket'inden binary TTS basmasin; ESP halen mevcut TTS relay
kuyrugunu kullanmaya devam etsin.

## Port

- Varsayilan port: `8766`
- WebSocket endpoint: `/ws`
- Health endpoint: `/health`

## Simdilik ne yapar?

- ESP benzeri bir istemciden `start` komutu alir
- PCM16 mono ses chunk'larini konusma devam ederken OpenAI Realtime'a akar
- ESP'nin 16 kHz PCM'ini OpenAI Realtime icin 24 kHz PCM'e cevirir
- OpenAI Realtime `server_vad` / `semantic_vad` eventlerini `vad_start` / `vad_end` olarak ESP'ye geri yollar
- Realtime text deltalarini `llm_delta` ve TTS'e uygun `llm_chunk` eventleri olarak yollar
- Realtime transkript hazirsa `stt_result` icinde gosterir; gec gelirse `stt_transcript` debug eventi yollar
- Realtime kapaliysa eski yerel `faster-whisper + streaming LLM` fallback hatti calisir
- belirgin ev kontrolu / durum sorgulari HA conversation backend'ine yonlenebilir
- TTS aciksa LLM parcalarini mevcut TTS relay'e gonderip sesi ayni websocket uzerinden geri aktarabilir
- sonucu tekrar WebSocket uzerinden JSON event olarak doner
- chunk seviyesinde `vad_start`, `vad_end`, `no_speech_timeout`, `max_utterance_reached` eventleri uretebilir
- `/health` icinde HA bridge durumunu raporlar
- WebSocket uzerinden temel HA komutlari kabul eder:
  - `ha_get_state`
  - `ha_list_states`
  - `ha_call_service`

## Simdilik ne yapmaz?

- Tool calling
- Gercek entity cache / akilli HA bridge islemleri
- Tam ESP entegrasyonu

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

Bu fazda OpenAI-uyumlu streaming LLM eklendi:

- `provider`: `openai`, `openrouter`, `none`
- `model`
- `api_key`
- `base_url`
- `system_prompt`

Varsayilan `base_url`: `https://api.openai.com/v1`

### OpenAI Realtime

`0.9.15` ile varsayilan ana hat OpenAI Realtime text-output modudur. `0.9.16` ile realtime provider
secimi ve oturum durum resetleri ayrildi; ileride OpenAI yerine farkli bir realtime provider takmak icin
ESP protokolu sabit kalacak, sadece add-on icindeki provider adapter'i degisecek.

- `realtime.enabled: true`
- `provider`: simdilik `openai`
- `model`: varsayilan hiz/maliyet odakli `gpt-realtime-mini`; kalite icin `gpt-realtime` secilebilir
- `input_sample_rate`: varsayilan `24000`
- `turn_detection`: `server_vad` veya `semantic_vad`
- `transcription_model`: varsayilan `gpt-4o-mini-transcribe`
- `transcript_wait_ms`: cevap baslatmadan once transkript icin beklenecek kisa pencere

Bu modda add-on OpenAI'den ses uretmez. OpenAI sadece gelen kullanici sesini ve cevabi text olarak isler.
Donen text parcalari ESP'ye `llm_chunk` olarak akar; ESP tarafindaki mevcut TTS relay kuyrugu Cartesia,
ElevenLabs veya baska bir TTS saglayicisini kullanmaya devam eder.

### TTS

Bu fazda TTS yerel sunucunun icinde sentezlenmez; mevcut `Alice Realtime TTS` add-on'una delege edilir.

- `tts.enabled: true` ise relay kullanilir
- `relay_url` varsayilan: `ws://127.0.0.1:8765/ws`
- LLM chunk'lari `start / append / final` mantigiyla relay'e aktarilir
- relay'den gelen `start`, binary PCM ve `done` istemciye aynen ileri tasinir

### Endpointing

Bu fazda basit bir yerel endpointing katmani vardir:

- `start_avg_abs_threshold`
- `end_avg_abs_threshold`
- `provider`
- `silero_start_threshold`
- `silero_end_threshold`
- `speech_start_min_ms`
- `speech_end_silence_ms`
- `no_speech_timeout_ms`
- `max_utterance_ms`

Bu katman su an tam uretim kalitesinde VAD degil; amaci yeni dis voice mimarisi icin konusma baslangic/bitis
eventlerini ve timeout davranisini iskelet olarak kurmaktir. ESP'den gelen eski HA tarzi 1 byte binary handler
basligi varsa sunucu bunu PCM disinda tutar; VAD kararinda ham mutlak seviyeye ek olarak merkezlenmis ses seviyesi
debug loglanir. Konusma baslangicinda kisa bir kalibrasyon penceresi kullanilir; konusma aktifken `active`
endpointing loglari `vad_end` esiginin neden dolup dolmadigini gosterir. Kucuk arka plan spike'lari artik
sessizlik sayacini sifirlamaz; sadece belirgin konusma seviyesi `resume_threshold` ustune cikarsa sayac sifirlanir.

Varsayilan endpointing provider `silero`dur. Add-on `faster-whisper` icindeki Silero ONNX modelini `onnxruntime`
ile streaming state koruyarak calistirir ve 16 kHz akista 32 ms'lik pencerelerle konusma olasiligi uretir.
Silero yuklenemezse add-on otomatik olarak eski enerji tabanli fallback endpointing'e duser.

Bos `llm.system_prompt` kullaniliyorsa add-on kisa konusan Turkce sesli asistan varsayilan prompt'u ekler.

### HA Bridge

Bu fazda HA bridge artik temel komutlara sahiptir:

- `SUPERVISOR_TOKEN` varsa `http://supervisor/core/api` uzerinden probe yapar
- `/health` cevabinda bridge gorunur
- `ha_get_state` ile tek entity okunabilir
- `ha_list_states` ile domain bazli sade liste alinabilir
- `ha_call_service` ile temel servis cagrisi yapilabilir
- `conversation.process` ile HA conversation agent'ina yonlenebilir

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

### WebSocket eventleri

Yeni eventlerden bazilari:

- `stt_result`
- `llm_started`
- `llm_delta`
- `llm_chunk`
- `llm_result`
- `ha_route_selected`
- `ha_conversation_result`
- `tts_result`
- `emotion`
- `vad_start`
- `vad_end`
- `no_speech_timeout`

## Sonraki faz

1. ESP'yi yeni `/ws` protokolune baglamak
2. Partial STT / daha iyi endpointing
3. HA entity cache ve akilli tool bridge yazmak
4. ESP tarafinda eski HA Assist yolunu devreden cikarmak
5. Gercek tool-calling / entity cache eklemek

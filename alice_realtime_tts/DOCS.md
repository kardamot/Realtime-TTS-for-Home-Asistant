# Alice Realtime TTS

Bu add-on, Home Assistant'in `intent` sonunda uretilen yanit metnini alip secilen TTS
saglayicisindan dusuk gecikmeli `PCM` akisi ceker ve Alice cihazi buna `WebSocket`
ile baglanip sesi blok blok oynatir.

Desteklenen saglayicilar:

- `openai`
- `cartesia`
- `elevenlabs`
- `google_ai`
- `google_cloud`

## Temel mantik

1. ESP32 yanit metnini bu relay'e yollar.
2. Relay secilen TTS saglayicisini cagirir.
3. Relay saglayicidan gelen sesi `pcm_s16le` olarak Alice'e aktarir.
4. Firmware tarafinda tek bir ic protokol kullanildigi icin saglayici degistiginde
   firmware degistirmen gerekmez.

## Gerekli ayarlar

Her saglayici icin anahtarlar Home Assistant add-on ayarinda tutulur. Bunlar
`/data/options.json` icinde saklanir; repo icine yazman gerekmez.

### Ortak alanlar

- `provider`: `openai`, `cartesia`, `elevenlabs`, `google_ai` veya `google_cloud`
- `port`: Varsayilan `8765`

### OpenAI

- `openai_api_key`
- `openai_model`: Varsayilan `gpt-4o-mini-tts`
- `openai_voice`: Varsayilan `coral`
- `openai_instructions`: Stil yonlendirmesi

### Cartesia

- `cartesia_api_key`
- `cartesia_model_id`: Varsayilan `sonic-3`
- `cartesia_voice_id`: Gerekli
- `cartesia_language`: Varsayilan `tr`
- `cartesia_version`: Varsayilan `2026-03-01`

### ElevenLabs

- `elevenlabs_api_key`
- `elevenlabs_model_id`: Varsayilan `eleven_flash_v2_5`
- `elevenlabs_voice_id`: Gerekli
- `elevenlabs_output_format`: Varsayilan `pcm_24000`
- `elevenlabs_latency_mode`: `0` - `4`

### Google AI

- `google_ai_api_key`
- `google_ai_model`: Varsayilan `gemini-3.1-flash-tts-preview`
- `google_ai_voice_name`: Varsayilan `Kore`
- `google_ai_prompt_prefix`: Istege bagli stil yonlendirmesi

### Google Cloud

- `google_cloud_credentials_json`: Service account JSON icerigi
- `google_cloud_voice_name`: Gerekli
- `google_cloud_language_code`: Varsayilan `tr-TR`
- `google_cloud_ssml_gender`: Varsayilan `FEMALE`

## Gecis yaparken

- Saglayici degistirince `Save` ve `Restart` yap.
- OpenAI'dan Cartesia'ya gecmek icin sadece `provider` ve ilgili Cartesia alanlarini
  degistirmen yeterli.
- Google AI icin API key yeterlidir.
- Google Cloud icin bir service account JSON gerekir; bunu tek satir veya cok satirli metin olarak
  add-on ayarina yapistirabilirsin.
- Eski anahtarlar silinmez; ayarlarda kalir. Istersen sonra tekrar secip geri donebilirsin.

## Notlar

- Firmware tarafinda relay adresi `ws://HA_IP:8765/ws` olarak derlenmistir.
- Relay, Alice'e her zaman `pcm_s16le` gonderir.
- Varsayilan ornekleme hizi `24 kHz`, kanal sayisi `mono`dur.
- Guvenlik icin API anahtarlarini repo dosyalarina yazma.

# Alice Home Assistant Add-ons

Bu klasor, HAOS tarafina tasinabilecek yerel add-on dosyalarini icerir.

## Kullanim

1. `home_assistant_addons` klasorunu Home Assistant'in gorebilecegi bir yere kopyala.
2. Home Assistant'ta yerel add-on repository olarak bu klasoru ekle.
3. `Alice Realtime TTS` veya `Alice Realtime Voice` add-on'unu kur.
4. Secili add-on'un ilgili ayarlarini doldur.
5. Mevcut firmware tarafinda TTS relay adresi `ws://192.168.1.168:8765/ws` olarak beklenir.

## Not

Bu surum tek ic protokolle birden fazla TTS saglayicisini destekler:

- OpenAI
- Cartesia
- ElevenLabs
- Google AI
- Google Cloud

Saglayici degistiginde firmware degistirmen gerekmez; yalnizca add-on ayarini guncelleyip
add-on'u yeniden baslatman yeterlidir.

`Alice Realtime Voice` ise yeni nesil dis voice pipeline icin hazirlanan ayri add-on'dur.
Bu turda `0.9.0` ile yerel `faster-whisper` STT, endpointing eventleri, OpenAI-uyumlu streaming LLM,
HA conversation yonlendirmesi, mevcut TTS relay orkestrasyonu ve temel HA bridge komutlari icerir.

Not: Home Assistant add-on ayarlari provider'a gore kosullu alan gizleme yapmaz; bunun yerine
ayarlar saglayici bazli gruplar halinde toplanmistir.

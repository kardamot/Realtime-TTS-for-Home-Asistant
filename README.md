# Alice Home Assistant Add-ons

Bu klasor, HAOS tarafina tasinabilecek yerel add-on dosyalarini icerir.

## Kullanim

1. `home_assistant_addons` klasorunu Home Assistant'in gorebilecegi bir yere kopyala.
2. Home Assistant'ta yerel add-on repository olarak bu klasoru ekle.
3. Yeni entegre panel icin `Alice Control Panel` add-on'unu kur veya mevcut referanslar icin
   `Alice Realtime TTS` / `Alice Realtime Voice` add-on'larini kullan.
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

`Alice Control Panel`, yeni tek parca panel/server hedefidir. Varsayilan portu `8099`'dur ve
Home Assistant ingress kullanmadan `http://HOME_ASSISTANT_IP:8099` adresinden acilir.

`Alice Realtime Voice` ise yeni nesil dis voice pipeline icin hazirlanan ayri add-on'dur.
Bu turda `0.9.9` ile yerel `faster-whisper` STT, endpointing eventleri, OpenAI-uyumlu streaming LLM,
HA conversation yonlendirmesi, mevcut TTS relay orkestrasyonu ve temel HA bridge komutlari icerir.

Not: Home Assistant add-on ayarlari provider'a gore kosullu alan gizleme yapmaz; bunun yerine
ayarlar saglayici bazli gruplar halinde toplanmistir.

# Alice Home Assistant Add-ons

Bu klasor, HAOS tarafina tasinabilecek yerel add-on dosyalarini icerir.

## Kullanim

1. `home_assistant_addons` klasorunu Home Assistant'in gorebilecegi bir yere kopyala.
2. Home Assistant'ta yerel add-on repository olarak bu klasoru ekle.
3. `Alice Control Panel` add-on'unu kur. Yeni birincil sistem budur.
4. Panel ayarlarini doldur.
5. Paneli `http://HOME_ASSISTANT_IP:8099` adresinden ac.

## Not

Bu surum tek ic protokolle birden fazla TTS saglayicisini destekler:

- OpenAI
- Cartesia
- ElevenLabs
- Google AI
- Google Cloud

Saglayici degistiginde firmware degistirmen gerekmez; yalnizca add-on ayarini guncelleyip
add-on'u yeniden baslatman yeterlidir.

`Alice Control Panel`, tek parca panel/server add-on'udur. Varsayilan portu `8099`'dur ve
Home Assistant ingress kullanmadan `http://HOME_ASSISTANT_IP:8099` adresinden acilir.

`Alice Realtime TTS` ve `Alice Realtime Voice` klasorleri artik aktif hedef degil; referans/arsiv
olarak tutulur. Yeni kurulum, panel, TTS, STT, LLM, ESP ve Home Assistant kontrol akisi
`alice_control_panel` icinden ilerler.

Not: Home Assistant add-on ayarlari provider'a gore kosullu alan gizleme yapmaz; bunun yerine
ayarlar saglayici bazli gruplar halinde toplanmistir.

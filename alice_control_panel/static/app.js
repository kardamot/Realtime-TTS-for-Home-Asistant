const espCommands = [
  "test_speaker", "test_mic", "capture_mic", "wake_on", "wake_off",
  "soft_sleep_on", "night_sleep_on", "sleep_mode_off",
  "motors_on", "motors_off", "amp_mute_on", "amp_mute_off", "radar_calibrate_empty", "radar_clear_empty", "reconnect", "reboot"
];
const UI_VERSION = "0.1.177";
const serverCommands = [
  "restart_stt", "restart_tts", "reload_prompt",
  "start_voice_session", "stop_voice_session", "cancel_response",
  "safe_mode_on", "safe_mode_off"
];
const commandLabels = {
  radar_calibrate_empty: "radar empty calib",
  radar_clear_empty: "radar clear calib",
};
const RADAR_CALIBRATION_KEY = "alice_radar_tech_calibration_v3";
let token = localStorage.getItem("alice_panel_token") || "";
let currentConfig = {};
let currentPrompt = {};
let latestStatus = {};
let latestRadar = {};
let radarUiTrack = { valid: false, direction: "BELIRSIZ" };
let latestRadarDraw = null;
let radarView = localStorage.getItem("alice_radar_view") || "tech";
let commandTab = localStorage.getItem("alice_command_tab") || "daily";
let pipelineView = localStorage.getItem("alice_pipeline_view") || "trace";
let radarCalibration = readRadarCalibration();
let logs = [];
let paused = false;
let logPreset = localStorage.getItem("alice_log_preset") || "all";
let logFocusMode = localStorage.getItem("alice_log_focus") === "1";
let expandedLogKey = "";
let configDirty = false;
let logSocket = null;
let logSocketSeq = 0;
let eventSocket = null;
let eventSocketSeq = 0;
let statusTimer = null;
let statusRefreshTimer = null;
let micDebugRefreshTimers = [];
let micDebug = {};
let speakerVolumeEditing = false;
const SPEAKER_VOLUME_STORAGE_KEY = "alice_speaker_volume_percent";
const SPEAKER_VOLUME_BEFORE_MUTE_KEY = "alice_speaker_volume_before_mute";
const DRIVE_SPEED_KEY = "alice_drive_speed_index";
const DRIVE_STEP_KEY = "alice_drive_step_index";
const MOTION_LOCK_KEY = "alice_motion_lock";
const DRIVE_SPEED_LABELS = ["Slow", "Normal", "Fast"];
const DRIVE_STEP_LABELS = ["Short", "Medium", "Long"];
const DAILY_TOGGLE_LABELS = {
  listen: { on: "Stop listening", off: "Start listening" },
  follow_up: { on: "Follow-up on", off: "Follow-up off" },
  touch_reactions: { on: "Touch on", off: "Touch off" },
  lift_reactions: { on: "Lift on", off: "Lift off" },
  motors: { on: "Motors on", off: "Motors off" },
  wake: { on: "Wake on", off: "Wake off" },
  sleep_mode: { on: "Wake", off: "Sleep" },
};
let rememberedSpeakerVolume = Number(localStorage.getItem(SPEAKER_VOLUME_STORAGE_KEY));
if (!Number.isFinite(rememberedSpeakerVolume) || rememberedSpeakerVolume < 0 || rememberedSpeakerVolume > 100) {
  rememberedSpeakerVolume = null;
} else {
  rememberedSpeakerVolume = Math.round(rememberedSpeakerVolume);
}
let driveSpeedIndex = readStoredIndex(DRIVE_SPEED_KEY, 1);
let driveStepIndex = readStoredIndex(DRIVE_STEP_KEY, 1);
let motionLocked = localStorage.getItem(MOTION_LOCK_KEY) === "1";
const autoScrollState = new WeakMap();
let helpPopover = null;
const RADAR_DIRECTION_DEADZONE_MM = 280;
const RADAR_DIRECTION_ENTER_MM = 340;
const RADAR_DIRECTION_EXIT_MM = 190;
const RADAR_DEFAULT_MAX_Y_MM = 6000;
const RADAR_UI_FILTER_ALPHA = 0.56;
const RADAR_UI_RESET_JUMP_MM = 1400;
const RADAR_UI_RESET_MS = 1800;
const LOG_PRESETS = {
  all: {},
  errors: { level: "ERROR" },
  warns: { level: "WARN" },
  voice: { categories: ["STT", "LLM", "TTS", "PIPELINE", "RT"] },
  esp: { categories: ["ESP"] },
  ha: { categories: ["HA"] },
};
const LOG_FOCUS_NOISE = [
  "Alice Control Panel backend starting",
  "Alice Control Panel backend stopped",
  "ESP manager started",
  "ESP WebSocket connected",
  "ESP WebSocket disconnected",
  "ESP command sent",
  "OpenAI Realtime connected",
  "OpenAI Realtime client disconnected",
  "OpenAI Realtime mic packet header stripped",
  "TTS trace ",
  "TTS relay websocket started",
  "TTS relay websocket disconnected",
  "Configuration updated",
];
const EMOTION_TAG_DISPLAY_RE = /<emotion:\s*[^>]+>/gi;

function stripEmotionTags(value) {
  return String(value || "").replace(EMOTION_TAG_DISPLAY_RE, "").replace(/\s{2,}/g, " ").trim();
}

const HELP_TEXTS = {
  connections: {
    title: "Connections",
    body: [
      "Bu panel robot ve servis bağlantılarının kısa özetidir. ESP satırı HTTP status poll tarafını, ESP WS satırı canlı WebSocket bağlantısını gösterir.",
      "STT, LLM ve TTS satırları o anda aktif seçili sağlayıcıları gösterir. HA Bridge satırı Home Assistant entegrasyonunun hazır olup olmadığını ve izin verilen entity listesinin kullanılıp kullanılmadığını anlatır.",
      "Reconnects değeri ESP bağlantısı koptuğunda yapılan otomatik deneme sayısıdır. Limit dolarsa sistem boşa uğraşmayı bırakır; yeniden denemek için reconnect komutu kullanılır."
    ]
  },
  logs: {
    title: "Logs",
    body: [
      "Burada ESP, STT, LLM, TTS, Pipeline, Home Assistant ve sistem olayları tek canlı akışta görünür. Yeni log geldikçe pencere terminal gibi aşağı kayar.",
      "Search, level ve category filtreleri sadece görüntüyü süzer; logları silmez. Pause akışı dondurur, Download mevcut logları dosya olarak indirir, Clear ise paneldeki log bufferını temizler.",
      "Hata ayıklarken en değerli yer burasıdır: bağlantı kopmaları, TTS sağlayıcı hataları, VAD/STT kararları ve HA allowlist okumaları burada görünür."
    ]
  },
  espHealth: {
    title: "ESP Health",
    body: [
      "ESP tarafindan bildirilen hafif sistem sagligi ozetidir. Sicaklik, CPU, internal RAM, PSRAM ve son reset sebebini tek bakista gosterir.",
      "Bu panel terminaldeki SYS_MON bilgisinin kisa web karsiligidir. Resetler, isiya bagli riskler veya bellek daralmasi gibi ipuclari icin kullanilir.",
      "OK normal, WARM/HOT sicaklik uyarisi, CHECK ise watchdog/brownout/panic gibi dikkat isteyen son reset sebebi anlamina gelir."
    ]
  },
  hardware: {
    title: "Hardware",
    body: [
      "Bu panel ESP tarafından bildirilen donanım durumlarını gösterir. Mic, Speaker, Servo, Amp ve Wake alanları robotun kendi status cevabından veya eventlerinden beslenir.",
      "State alanı robotun o anki çalışma durumudur: IDLE, LISTENING, THINKING, SPEAKING veya ERROR gibi. Bir değer unknown görünüyorsa panel değil, ESP tarafı henüz o bilgiyi göndermiyor demektir."
    ]
  },
  pipeline: {
    title: "Voice Pipeline",
    body: [
      "Bu panel ses ve metin hattını elle test etmek içindir. Text test kutusuna yazıp LLM + TTS dersen metin LLM'e gider, gelen cevap seçili TTS ile ESP'ye okutulur.",
      "TTS only sadece yazdığın metni seçili TTS sağlayıcısıyla okutur; LLM'e soru sormaz. Bu, ses sağlayıcısını ve ESP audio stream hattını hızlı test etmek için kullanışlıdır.",
      "Start session, Stop session ve Cancel response canlı oturum/barge-in altyapısını denemek içindir. User/STT ve LLM alanları son algılanan konuşmayı ve üretilen cevabı gösterir."
    ]
  },
  commands: {
    title: "Command Panel",
    body: [
      "Üst bölüm ESP komutlarıdır: hoparlör testi, mikrofon testi, wake aç/kapat, kısa N20 motor testi, amfi mute, reconnect ve reboot gibi doğrudan robota giden işler burada durur.",
      "Mic Debug satırı sol ve sağ I2S mikrofon kanalını ayrı ayrı kısa WAV kaydı olarak yakalamak içindir. Yeni mikrofon bağlantısında SEL/kanal tersliği veya sessiz kanal sorununu hızlıca ayırt eder.",
      "Alt bölüm server komutlarıdır. STT/TTS yeniden başlatma, prompt reload, log temizleme, safe mode aç/kapat gibi add-on tarafındaki işlemleri tetikler.",
      "Bazı butonlar ESP firmware tarafında henüz desteklenmiyorsa komut loga düşer ve 'not implemented' benzeri cevap döner. Bu normaldir; panel komut yolunu kaybetmez."
    ]
  },
  prompts: {
    title: "Prompt Editor",
    body: [
      "Prompt profilleri Alice'in genel karakterini ve davranış talimatını yönetir. Alice, Debug veya Minimal gibi profiller dosya olarak /data/prompts altında saklanır.",
      "Aktif profil, LLM system prompt boşsa classic LLM hattında kullanılır. Live Voice tarafında da Live instructions ve LLM system prompt boşsa yine aktif prompt profiline düşülür.",
      "New yeni profil oluşturur, Copy mevcut profili kopyalar, Activate seçili profili aktif yapar, Save ise metin değişikliklerini kaydeder. Prompt değişikliği server restart gerektirmez."
    ]
  },
  config: {
    title: "Config",
    body: [
      "Config paneli add-on'un kalıcı ayar merkezidir. Sağlayıcı API keyleri, ESP adresleri, Home Assistant allowlist'i, prompt fallback davranışı ve audio buffer ayarları buradan yönetilir.",
      "Kaydedilen değerler /data/alice_config.json altında kalır; add-on güncellesen de normalde korunur. API key ve tokenlar repo içine yazılmaz.",
      "Export varsayılan olarak secretları maskeleyerek dışa aktarır. Secrets kutusunu açarsan gerçek keyleri de dahil eder; bunu sadece gerçekten yedek almak istediğinde kullan."
    ]
  },
  panelEsp: {
    title: "Panel & ESP",
    body: [
      "Panel port, token ve password web panel/API erişimini yönetir. Token veya password boşsa ev içi lokal kullanım için auth kapalı kalabilir; doluysa REST, WebSocket ve UI erişimi korunur.",
      "ESP base URL robotun HTTP API adresidir. ESP WebSocket URL canlı event, log, mikrofon ve audio stream yolu için kullanılır. Genelde aynı IP'nin /ws endpointidir.",
      "Max auto reconnects bağlantı kopunca kaç kez otomatik deneneceğini belirler. Debug logs daha ayrıntılı kayıt üretir; Safe mode riskli/aktif işleri azaltmak için acil durum anahtarıdır."
    ]
  },
  liveVoice: {
    title: "Live Voice",
    body: [
      "Bu bölüm düşük gecikmeli canlı konuşma hattını yönetir. OpenAI Live seçilirse ESP'nin /voice/ws bağlantısı OpenAI Realtime hattına yönlenir; None seçilirse live hat devre dışı kalır.",
      "Turn detection, VAD threshold, silence ve prefix ayarları konuşmanın nerede başlayıp bittiğini belirler. Semantic eagerness sadece semantic_vad seçildiğinde modelin konuşma sonunu ne kadar istekli kapatacağını etkiler.",
      "Live instructions canlı oturuma özel kişilik talimatıdır. Boş bırakırsan sistem önce LLM system prompt'a, o da boşsa aktif Prompt Editor profiline düşer. Realtime STT prompt ise transkripsiyona ipucu verir; Türkçe ve özel isimlerde işe yarayabilir.",
      "Gemini Live kartı şimdilik sağlayıcı bilgilerini hazır tutmak içindir. Tam canlı Gemini WebSocket hattı henüz OpenAI Live kadar bağlı değildir."
    ]
  },
  sttVad: {
    title: "Classic STT & VAD",
    body: [
      "Classic STT tarafı tek seferlik mikrofon yakalama veya live olmayan pipeline için faster-whisper ayarlarını tutar. Model, language, compute type ve beam size transkripsiyon kalitesini/gecikmesini etkiler.",
      "Live VAD provider konuşma başlangıç/bitiş algısını yönetir. Silero daha gerçek VAD yaklaşımıdır; energy daha basit RMS tabanlı yedek yoldur.",
      "Start/end olasılıkları, RMS eşikleri, silence ve max utterance değerleri mikrofon ortamına göre ince ayar ister. Dip gürültüsü varsa enerji tabanlı ayarlar kolayca yanlış tetiklenebilir."
    ]
  },
  homeAssistant: {
    title: "Home Assistant",
    body: [
      "HA Bridge, Alice'in Home Assistant state ve servislerine kontrollü erişimini sağlar. Bu sistem bilinçli olarak allowlist mantığıyla çalışır; tüm entityler robota açılmaz.",
      "Allowed entity IDs kutusuna sadece izin vermek istediğin entityleri satır satır yazarsın. Örneğin weather.erzurum_hava_durumu burada varsa Alice onu okuyabilir; listede olmayan entitylere erişmez.",
      "Route home control açıkken LLM cevabından önce bazı ev kontrolü ve hava durumu istekleri doğrudan HA bridge tarafından karşılanır. HA API base Home Assistant Supervisor içinden varsayılan olarak doğru gelir."
    ]
  },
  llm: {
    title: "LLM",
    body: [
      "LLM bölümü metni anlayıp cevap üreten sağlayıcıyı seçer. OpenAI, OpenRouter, Groq, Gemini ve generic OpenAI-compatible profilleri ayrı ayrı saklanır; sağlayıcı değiştirince eski key/model bilgileri silinmez.",
      "Active LLM hangi profilin kullanılacağını belirler. Temperature cevapların ne kadar serbest olacağını etkiler; düşük değer daha tutarlı, yüksek değer daha yaratıcı cevap verir.",
      "LLM system prompt doluysa aktif Prompt Editor profilinin üstüne geçer. Boş bırakılırsa seçili prompt profili Alice'in genel kişiliği olarak kullanılır."
    ]
  },
  tts: {
    title: "TTS",
    body: [
      "TTS bölümü yazıyı sese çeviren sağlayıcıyı seçer. OpenAI, Cartesia, ElevenLabs, Google AI ve Google Cloud bilgileri ayrı kartlarda saklanır; geçiş yaptığında önceki sağlayıcının ayarları kaybolmaz.",
      "PCM rate genel ESP audio hedefidir; bazı sağlayıcılar kendi sabit sample rate'iyle gelebilir ve backend bunu uygun metadata ile iletir. ESP start buffer ve silence prefix ilk ses takılmalarını azaltmak için kullanılır.",
      "Mic response, mikrofon testlerinden sonra ne yapılacağını seçer: sadece asistan cevabı, duyulan metni tekrar etme veya önce tekrar edip sonra cevaplama. Barge-in cancel açıksa konuşma sırasında yeni giriş eski cevabı kesebilir."
    ]
  }
};

const HELP_DETAIL_TEXTS = {
  espHealthFields: {
    title: "ESP Health alanlari",
    body: [
      "Bu alanlar ESP'nin kendi status cevabindan gelir. Terminalde SYS_MON ile gordugun bilgilerin kisa ve panelde kalici gorunen halidir."
    ],
    items: [
      ["Temp", "ESP ic sicakligi. 70 C ustu WARM, 78 C ustu HOT olarak isaretlenir; surekli yuksekse reset veya kararsizlik nedeni olabilir."],
      ["CPU", "ESP tarafindaki ortalama CPU yuku ve varsa cekirdek dagilimi. Kisa anlik snapshot oldugu icin trend icin arka arkaya bakmak gerekir."],
      ["RAM", "Internal RAM free/total ozetidir. Free veya largest block cok duserse audio, JSON veya task allocation sorunlari gorulebilir."],
      ["PSRAM", "PSRAM free/total ozetidir. Ses bufferlari ve buyuk veri yapilari icin genel rahatlik payini gosterir."],
      ["Reset", "Son acilisin reset sebebidir. brownout, watchdog, panic, power_glitch veya cpu_lockup CHECK/WARN olarak degerlendirilir."],
      ["Freq", "ESP CPU frekansidir. Alice runtime genelde performans icin 240 MHz'e cikar."]
    ]
  },
  panelEspFields: {
    title: "Panel & ESP alanlari",
    body: [
      "Bu detaylar panelin nasil korunacagini ve ESP ile hangi adreslerden konusacagini belirler. Yanlis adres girilirse panel acilir, ama robot mock/offline gorunur."
    ],
    items: [
      ["Panel port", "Add-on web panelinin dinledigi porttur. Varsayilan 8099; Home Assistant disindan http://HA_IP:8099 ile acilir."],
      ["Panel token", "API, WebSocket ve UI icin basit bearer/token korumasi saglar. Bos kalirsa lokal kullanim icin auth kapali olabilir."],
      ["Panel password", "Token yerine veya tokenla birlikte kullanilabilen basit panel sifresidir. Ev ici kullanimda bos birakmak mumkun, ama dis aglara acma."],
      ["ESP base URL", "Robotun HTTP API adresidir. Ornek: http://192.168.1.49. Status poll ve POST /api/command buradan gider."],
      ["ESP WebSocket URL", "Robotun canli event/log/mikrofon/audio WebSocket yoludur. Genelde ws://192.168.1.49/ws seklindedir."],
      ["Reconnect interval", "ESP koptugunda otomatik denemeler arasindaki bekleme suresidir. Cok kisa olursa gereksiz log ve ag trafigi uretir."],
      ["Max auto reconnects", "Otomatik deneme limitidir. Limit dolunca sistem durur ve manuel reconnect bekler. 0 verirsen sinirsiz dener."],
      ["Debug logs", "Daha ayrintili log uretir. Testte faydali, stabil kullanimda log kalabaligini azaltmak icin kapatilabilir."],
      ["Safe mode", "Riskli otomasyonlari azaltmak veya sorunlu bir pipeline'i yavaslatmak icin guvenli moda alir. Acil durum freni gibi dusun."]
    ]
  },
  liveVoiceFields: {
    title: "Live Voice alanlari",
    body: [
      "Live Voice, wake word sonrasindaki dusuk gecikmeli konusma hattidir. Bu ayarlar konusmanin ne zaman baslayip bitecegini ve OpenAI/Gemini live profilinin nasil calisacagini belirler."
    ],
    items: [
      ["Active live", "none secilirse live hat kapali kalir. openai secilirse /voice/ws OpenAI Realtime hattina gider. gemini karti simdilik hazir profil olarak tutulur."],
      ["Input rate", "ESP'den gelen mikrofon PCM sample rate degeridir. ESP tarafindaki gercek rate ile uyusmali; yanlis olursa STT/VAD zamanlamasi sapar."],
      ["Output voice", "Live modelin dogrudan ses uretmesi kullanildiginda secilecek sestir. Classic TTS kullaniminda asil ses TTS bolumunden gelir."],
      ["Output format", "Live hattin urettigi audio formatidir. ESP tarafinin bekledigi PCM formatiyla uyumlu olmali."],
      ["Turn detection", "Konusma bitisini kimin karar verecegini secer. server_vad klasik esik/sessizlik, semantic_vad modelin anlam temelli bitis kararidir."],
      ["VAD threshold", "server_vad icin konusma algilama hassasiyetidir. Dusurmek daha kolay tetikler; yukseltmek dip gurultusune karsi daha sert davranir."],
      ["Prefix padding ms", "Konusma baslamadan hemen onceki kisa sesi de yakalamak icin basa eklenen tampon suresidir. Ilk heceleri kesmeyi azaltir."],
      ["Silence duration ms", "Konusma bittikten sonra kac ms sessizlik gorulurse turn kapanir. Kisa deger hizli cevap, uzun deger daha az erken kesme demektir."],
      ["Semantic eagerness", "semantic_vad seciliyken modelin konusma bitti demeye ne kadar istekli olacagidir. High hizli, low daha sabirli davranir."],
      ["Idle timeout ms", "Live oturum bos kalirsa ne kadar sure sonra toparlanacagini belirler. Takili kalan oturumlari temizlemeye yarar."],
      ["Live instructions", "Canli oturuma ozel kisilik ve davranis talimatidir. Bos kalirsa LLM system prompt'a, o da bossa aktif Prompt Editor profiline dusulur."],
      ["Realtime STT prompt", "Transkripsiyon icin ipucu metnidir. Turkce, Alice, yerel isimler veya sik yanlis duyulan kelimeleri buraya yazmak tanimayi iyilestirebilir; OpenAI siniri nedeniyle 1024 karakterle sinirlidir."],
      ["OpenAI Live key/model/base URL", "OpenAI Realtime icin kimlik ve model ayarlaridir. Genelde base URL varsayilan kalir; model ve key doldurulur."],
      ["Gemini Live key/model/voice", "Gelecekteki Gemini live hatti icin saklanan profil bilgileridir. Su an OpenAI Live kadar tamamlanmis bir canli yol degildir."]
    ]
  },
  sttVadFields: {
    title: "Classic STT & VAD alanlari",
    body: [
      "Bu bolum live olmayan mikrofon yakalama ve yerel VAD kararlari icindir. Mikrofon dip gurultusu varsa ozellikle VAD ayarlari hassas davranir."
    ],
    items: [
      ["STT provider", "Su an faster_whisper hedeflenir. Mikrofon kaydi metne cevrilirken bu motor kullanilir."],
      ["STT model", "Whisper model boyutudur. Kucuk modeller hizli, buyuk modeller daha dogru ama daha agir calisir."],
      ["Language", "Transkripsiyon dili. Turkce icin tr kullanmak hallucination ve dil kaymasini azaltabilir."],
      ["Compute type", "Modelin hesaplama hassasiyetidir. int8 daha hafif, float16/float32 daha agir ama bazi sistemlerde daha kaliteli olabilir."],
      ["CPU threads", "Whisper isleminde kac CPU thread kullanilacagidir. Mini PC'de fazla vermek sistemi gereksiz yorabilir."],
      ["Workers", "Ayni anda kac is parcacigi calisacagini belirler. Genelde dusuk tutmak daha stabil olur."],
      ["Beam size", "STT'nin alternatif metin arama genisligidir. Yuksek deger kaliteyi artirabilir ama gecikmeyi de artirir."],
      ["Live VAD provider", "silero gercek VAD modelidir; energy ise ses enerjisine bakar. Dip gurultulu ortamda silero daha mantikli baslangic noktasi."],
      ["Start probability", "Silero konusma basladi demek icin gereken olasilik esigidir. Dusuk deger hassas, yuksek deger secici davranir."],
      ["End probability", "Silero konusma bitti demek icin gereken esiktir. Yanlis erken bitislerde ayar gerekebilir."],
      ["RMS threshold", "Energy VAD icin ses siddeti esigidir. Dip gurultusu yuksekse bu degeri artirmak gerekebilir."],
      ["Min speech ms", "Konusma kabul edilmeden once gereken minimum ses suresidir. Cok kisa tikirti ve patlamalari elemek icin kullanilir."],
      ["Min silence ms", "Konusma bitisi icin gereken sessizlik suresidir. Kisa olursa erken keser, uzun olursa cevap gecikir."],
      ["Max utterance ms", "Tek konusma parcasi icin ust sinirdir. VAD takilsa bile oturumu sonsuza kadar acik birakmaz."]
    ]
  },
  homeAssistantFields: {
    title: "Home Assistant alanlari",
    body: [
      "Home Assistant bolumu bilincli olarak beyaz liste mantigiyla calisir. Alice sadece senin yazdigin entityleri okuyup yonetebilmeli."
    ],
    items: [
      ["HA API base", "Add-on icinden Home Assistant API adresidir. Supervisor ortaminda varsayilan deger genelde dogrudur."],
      ["HA Bridge enabled", "Alice'in Home Assistant state ve servis yolunu kullanip kullanmayacagini acar/kapatir."],
      ["Route home control", "Hava durumu veya basit ev kontrolu gibi istekleri LLM'e birakmadan once HA bridge tarafinda yakalamaya calisir."],
      ["Allowed entity IDs", "Erisime izin verdigin entityleri satir satir yazarsin. Liste disindaki entityler okunmaz ve kontrol edilmez."],
      ["Entity aliases", "Allowlist'teki entityleri dogal Turkce adlarla eslestirir. Alias yeni erisim izni vermez; sadece mevcut izinli entitynin daha dogru bulunmasini saglar."],
      ["Weather entity", "Hava durumu sorularinda oncelikli kullanilacak weather entitysidir. Allowed list icinde olmasi gerekir."],
      ["Service calls", "Kontrol komutlari ileride HA servislerine donusebilir. Allowlist bu yuzden guvenlik siniri olarak onemli kalir."]
    ]
  },
  llmFields: {
    title: "LLM alanlari",
    body: [
      "LLM metni anlayip cevap ureten kisimdir. Her saglayicinin karti ayri saklanir; saglayici degistirmek diger key ve model bilgilerini silmez."
    ],
    items: [
      ["Active LLM", "Klasik pipeline'da hangi metin modeli profilinin kullanilacagini secer. Live Voice aciksa cevap uretimi live hatta kayabilir."],
      ["Temperature", "Cevabin yaraticiligini belirler. 0.2-0.4 daha tutarli, 0.7 ve ustu daha serbest cevaplar uretir."],
      ["Streaming", "Model cevabini parca parca almak icindir. Erken TTS ve daha dusuk algilanan gecikme icin faydali olabilir."],
      ["LLM system prompt", "Bu alan doluysa Prompt Editor profilinin onune gecer. Bos birakilirsa aktif prompt profili kullanilir."],
      ["OpenAI", "OpenAI API key, model ve base URL ayarlari. Normal OpenAI kullaniminda bu kart doldurulur."],
      ["OpenRouter", "OpenRouter uzerinden farkli modelleri tek API ile denemek icindir. Base URL genelde OpenRouter varsayilanidir."],
      ["Groq", "Groq'un OpenAI uyumlu sohbet endpoint mantigiyla calisir. Dusuk gecikmeli metin cevaplari icin kullanilabilir."],
      ["Gemini", "Google Gemini classic text modeli icindir. Gemini Live ile ayni sey degildir; bu kart metin cevabi uretir."],
      ["OpenAI Compatible", "LM Studio, Ollama proxy, vLLM veya baska OpenAI uyumlu endpointler icin genel profil."]
    ]
  },
  ttsFields: {
    title: "TTS alanlari",
    body: [
      "TTS yaziyi sese cevirir ve sonuc ESP'ye stream edilir. Provider kartlari ayri saklandigi icin Cartesia'dan Google Cloud'a gecmek eski Cartesia ayarlarini silmez."
    ],
    items: [
      ["Active TTS", "Hangi TTS saglayicisinin kullanilacagini secer. OpenAI, Cartesia, ElevenLabs, Google AI ve Google Cloud ayri profillerdir."],
      ["PCM rate", "ESP'ye hedeflenen PCM sample rate bilgisidir. Bazilarinda saglayici kendi rate'ini verir; backend uygun metadata ile yollar."],
      ["ESP start buffer ms", "ESP'nin ses baslamadan once ne kadar tampon toplamasini istedigini belirler. Ilk saniye takilmalarini azaltabilir."],
      ["ESP silence prefix ms", "Sesin basina kisa sessizlik ekler. DAC/I2S/stream baslangicindaki tiklama ve kesilmeleri yumusatmak icindir."],
      ["Mic response", "Mikrofon testinden sonra sadece cevap, sadece duydugunu tekrar veya once tekrar sonra cevap davranisini secer."],
      ["TTS enabled", "Kapaliysa metin uretilebilir ama sese donusturme atlanir."],
      ["Stream TTS to ESP", "Aciksa ses ESP'ye WebSocket/audio protokoluyle gider. Kapaliysa backend TTS uretse bile robota okutmaz."],
      ["Barge-in cancel", "Kullanici konusurken mevcut cevabi kesmeye izin verir. Full-duplex hedefi icin onemli bir ayardir."],
      ["OpenAI TTS", "OpenAI API key, model, voice ve instructions alanlarini kullanir. Instructions ses tarzini yonlendirebilir."],
      ["Cartesia", "Cartesia API key, model ID, voice ID, language ve version ayarlaridir. Kredi/limit hatalari bu provider'dan gelebilir."],
      ["ElevenLabs", "API key, model, voice, output format ve latency mode ayarlaridir. Dusuk latency modlari kalite/gecikme dengesi kurar."],
      ["Google AI", "AI Studio API key, model ve voice name ile calisir. Prompt prefix, sese gidecek metni uslup olarak yonlendirebilir."],
      ["Google Cloud", "Service account JSON, voice name, language code ve gender alanlarini kullanir. Cloud TTS icin JSON kimligi gerekir."]
    ]
  }
};

const HELP_TARGETS = [
  [".connections-panel > header h2", "connections", "connectionsFields"],
  ["#esp-health > header h2", "espHealth", "espHealthFields"],
  ["#logs > header h2", "logs", "logsFields"],
  ["#radar > header h2", "radar", "radarFields"],
  ["#pipeline > header h2", "pipeline", "pipelineFields"],
  ["#commands > header h2", "commands", "commandsFields"],
  ["#latency-panel > header h2", "latency", "latencyFields"],
  ["#conversation > header h2", "conversation", "conversationFields"],
  ["#prompts > header h2", "prompts", "promptsFields"],
  ["#config > header h2", "config", "configFields"],
  ["#config .config-group:nth-of-type(1) h3", "panelEsp", "panelEspFields"],
  ["#config .config-group:nth-of-type(2) h3", "powerSleep", "powerSleepFields"],
  ["#config .config-group:nth-of-type(3) h3", "liveVoice", "liveVoiceFields"],
  ["#config .config-group:nth-of-type(4) h3", "sttVad", "sttVadFields"],
  ["#config .config-group:nth-of-type(5) h3", "homeAssistant", "homeAssistantFields"],
  ["#config .config-group:nth-of-type(6) h3", "llm", "llmFields"],
  ["#config .config-group:nth-of-type(7) h3", "tts", "ttsFields"]
];

Object.assign(HELP_TEXTS, {
  connections: {
    title: "Connections",
    body: [
      "Robot, canlı WebSocket, ses hattı ve Home Assistant köprüsünün o anki bağlantı özetidir.",
      "ESP satırı HTTP status poll tarafını, ESP WS satırı canlı event/audio/mikrofon kanalını gösterir. HA Bridge yalnızca allowlist içindeki Home Assistant varlıkları için hazır kabul edilir.",
      "Reconnects değeri otomatik yeniden bağlanma sayacıdır; limit dolarsa sistem manuel reconnect bekleyerek gereksiz ağ trafiğini keser."
    ]
  },
  logs: {
    title: "Logs",
    body: [
      "ESP, STT, LLM, TTS, Pipeline, HA ve sistem olaylarının tek canlı akışıdır.",
      "Filtreler sadece görünümü daraltır. Satıra tıklayınca ayrıntı JSON'u açılır; Pause, Download ve Clear günlük inceleme akışını yönetir.",
      "TTS sağlayıcı hataları, WebSocket kopmaları, barge-in iptalleri ve HA route kararları için ilk bakılacak panel burasıdır."
    ]
  },
  radar: {
    title: "Radar",
    body: [
      "RD-03D radar verisini teknik harita veya oda görünümüyle gösterir.",
      "Seçili hedef, yön, açı, mesafe ve güven bilgileri robotun kişiye dönme ve takip davranışını anlamak için kullanılır.",
      "Kalibrasyon butonları sadece panel görünümünü ve boş oda referansını düzeltir; robotun fiziksel montajını değiştirmez."
    ]
  },
  pipeline: {
    title: "Voice Pipeline",
    body: [
      "Ses ve metin hattının canlı özetidir: son durum, STT metni, LLM cevabı ve TTS'e giden son metin burada kalır.",
      "Üstteki test kontrolleri klasik LLM+TTS ve sadece TTS denemesi yapar. Oturum kontrolleri canlı konuşma, kesme ve barge-in davranışını test eder.",
      "Daha uzun konuşma geçmişi Conversation panelinde, milisaniye kırılımı ise Latency Timeline panelinde izlenir."
    ]
  },
  commands: {
    title: "Command Panel",
    body: [
      "Günlük robot kontrolü ve ileri seviye bakım komutlarının merkezidir.",
      "Daily sekmesi sürüş, ses seviyesi, uyku, wake, takip ve reaksiyon anahtarlarını daha temiz kullanmak içindir. Advanced sekmesi test, davranış, server bakım ve mikrofon debug işlerini tutar.",
      "ESP tarafında desteklenmeyen komutlar panelde kaybolmaz; loga düşer ve firmware hazır olana kadar güvenli şekilde cevap döner."
    ]
  },
  latency: {
    title: "Latency Timeline",
    body: [
      "Bir konuşma turunun nerede zaman kaybettiğini gösteren ayrıntılı zaman çizelgesidir.",
      "Wake'ten mikrofon paketine, STT sonucuna, LLM cevabına, Google TTS isteğine, ESP chunk aktarımına ve hoparlör başlangıcına kadar süreler ayrı ayrı ölçülür.",
      "TTS gecikmesi Google tarafında mı, bizim decode/resample tarafında mı, ESP aktarımında mı yoksa prebuffer/hoparlörde mi sorusunun cevabı burada aranır."
    ]
  },
  conversation: {
    title: "Conversation",
    body: [
      "Son kullanıcı ve asistan metinlerinin okunabilir konuşma geçmişidir.",
      "STT sonucu, OpenAI Realtime cevabı, HA route ile üretilen cevaplar ve TTS'e giden metinler burada sırayla kalır.",
      "Debug log gibi teknik değildir; konuşmanın gerçekten ne anlaşılıp ne söylendiğini hızlı kontrol etmek için kullanılır."
    ]
  },
  prompts: {
    title: "Prompt Editor",
    body: [
      "Alice'in genel karakter ve davranış profillerini yönetir.",
      "Live instructions veya LLM system prompt boşsa aktif prompt profili fallback olarak kullanılır; bu yüzden genel kişilik metnini burada tutmak temizdir.",
      "Profiller /data/prompts altında saklanır ve kaydetmek için server restart gerekmez."
    ]
  },
  config: {
    title: "Config",
    body: [
      "Add-on'un kalıcı ayar merkezidir: panel erişimi, ESP adresleri, sleep scheduler, live voice, STT/VAD, HA allowlist, LLM ve TTS provider profilleri buradan yönetilir.",
      "Kaydedilen değerler /data/alice_config.json altında kalır; API keyler repo içine yazılmaz.",
      "Import/Export yedek ve taşıma içindir. Secrets işaretlenmezse export gizli alanları maskeleyerek verir."
    ]
  },
  panelEsp: {
    title: "Panel & ESP",
    body: [
      "Panelin hangi porttan açılacağını, basit auth davranışını ve ESP ile hangi HTTP/WebSocket adreslerinden konuşacağını belirler.",
      "ESP base URL status ve komutlar içindir; ESP WebSocket URL canlı event, mikrofon debug ve audio stream için kullanılır."
    ]
  },
  powerSleep: {
    title: "Power & Sleep",
    body: [
      "Alice'in boşta kalınca veya gece saatlerinde daha sakin/güç tasarruflu moda geçmesini yönetir.",
      "Soft sleep gündüz boşta kalınca devreye girer; night sleep belirlenen saat aralığında daha derin uyku davranışı ister.",
      "Radar taze hedef/kişi görüyorsa soft sleep'e geçilmez; Alice soft sleep'teyken radar tekrar kişi görürse aktif moda uyandırılır. Night sleep saat kuralı ise daha baskın kalır."
    ]
  },
  liveVoice: {
    title: "Live Voice",
    body: [
      "Wake sonrası düşük gecikmeli karşılıklı konuşma hattını yönetir.",
      "OpenAI Live şu an aktif canlı konuşma yoludur. Gemini Live profili hazır tutulur; None seçilirse canlı hat devre dışı kalır.",
      "Live instructions canlı oturuma özel kişilik metnidir; boşsa LLM system prompt'a, o da boşsa aktif Prompt Editor profiline düşer."
    ]
  },
  sttVad: {
    title: "Classic STT & VAD",
    body: [
      "Live olmayan mikrofon yakalama, faster-whisper transkripsiyon ve yerel VAD ayarlarını tutar.",
      "Silero VAD konuşma başlangıç/bitişini modelle algılar; energy modu daha basit RMS eşiğiyle çalışır.",
      "Mikrofon dip gürültüsü varsa VAD eşikleri ve silence süreleri konuşma kalitesini doğrudan etkiler."
    ]
  },
  homeAssistant: {
    title: "Home Assistant",
    body: [
      "Alice'in Home Assistant varlıklarını kontrollü şekilde okumasını ve yönetmesini sağlar.",
      "Sistem beyaz listeyle çalışır: yalnızca Allowed entity IDs içine yazdığın entityler görünür.",
      "Route home control açıkken hava durumu ve bazı ev kontrol istekleri LLM cevabından önce HA bridge tarafından yakalanabilir."
    ]
  },
  llm: {
    title: "LLM",
    body: [
      "Klasik metin anlama ve cevap üretme sağlayıcısını seçer.",
      "OpenAI, OpenRouter, Groq, Gemini, OpenAI Compatible, Mock ve None profilleri ayrı saklanır; provider değiştirince eski bilgiler silinmez.",
      "Live Voice açıksa konuşmanın ana cevabı live modelden gelebilir; bu panel daha çok klasik LLM hattı ve fallback davranışı içindir."
    ]
  },
  tts: {
    title: "TTS",
    body: [
      "Yazıyı sese çeviren provider ve ESP audio aktarım ayarlarını yönetir.",
      "OpenAI, Cartesia, ElevenLabs, Google AI ve Google Cloud profilleri ayrı saklanır. Aktif TTS seçimi sadece hangi profilin kullanılacağını belirler.",
      "Buffer, silence prefix, streaming ve barge-in ayarları ilk ses gecikmesi, takılma ve konuşurken kesme davranışını etkiler."
    ]
  }
});

Object.assign(HELP_DETAIL_TEXTS, {
  connectionsFields: {
    title: "Connections detayları",
    body: ["Bu alanlar bağlantıların hangi katmanda sağlam veya sorunlu olduğunu ayırmak için kullanılır."],
    items: [
      ["ESP", "Robotun HTTP status poll sonucudur. Online ise /api/status okunuyor demektir."],
      ["ESP WS", "Robotun /ws canlı WebSocket bağlantısıdır. Audio, event, log ve mic debug için kritik yoldur."],
      ["STT", "Aktif konuşmayı metne çevirme motorunu gösterir; classic hatta genelde faster_whisper görünür."],
      ["LLM", "Aktif klasik LLM provider/model özetidir. Live Voice açıksa canlı model ayrıca Voice Pipeline'da görünür."],
      ["TTS", "Aktif TTS provider ve hedef PCM rate özetidir."],
      ["HA Bridge", "Home Assistant entegrasyonunun allowlist ile hazır olup olmadığını gösterir."],
      ["Reconnects", "ESP kopunca yapılan otomatik reconnect sayısı ve limitidir."]
    ]
  },
  logsFields: {
    title: "Logs kontrolleri",
    body: ["Log paneli canlıdır; filtreler ve presetler sadece görünümü değiştirir, backend ring buffer'ını bozmaz."],
    items: [
      ["Pause", "Canlı akışı ekranda dondurur. Arka planda yeni loglar gelmeye devam edebilir."],
      ["Download", "Mevcut log listesini dosya olarak indirir. TTS/HA hata detaylarını paylaşmak için en temiz yoldur."],
      ["Clear", "Paneldeki log buffer'ını temizler."],
      ["Search logs", "Mesaj, kategori veya detay içeriğinde metin arar."],
      ["Level filter", "ALL, DEBUG, INFO, WARN veya ERROR seviyesine göre süzer."],
      ["Category filter", "ESP, STT, LLM, TTS, PIPELINE, HA, SYSTEM gibi kaynağa göre süzer."],
      ["Summary chips", "Total, Errors, Warns ve son log kaynağı/saatini hızlı gösterir."],
      ["Preset buttons", "All, Errors, Warnings, Voice, ESP, HA gibi hazır filtreleri uygular."],
      ["Focus", "Gürültülü tekrar loglarını azaltıp daha önemli satırlara odaklanmak için kullanılan görünüm modudur."],
      ["Expandable rows", "Satırdaki + işaretine tıklayınca JSON detayını açar; provider error body, trace_id ve timing payload burada görünür."]
    ]
  },
  radarFields: {
    title: "Radar kontrolleri",
    body: ["Radar paneli hem teknik ham hedefleri hem de oda içi konum yorumunu gösterir."],
    items: [
      ["X", "Paneldeki X eksenini ters çevirir. Sağ/sol görüntüsü fiziksel yerleşime tersse kullanılır."],
      ["Y", "Paneldeki Y eksenini ters çevirir. İleri/geri görüntüsü tersse kullanılır."],
      ["180", "Radar görüntüsünü 180 derece çevirir."],
      ["Teknik", "Ham radar koordinatlarını ve hedef noktalarını teknik haritada gösterir."],
      ["Oda", "Aynı veriyi daha okunabilir oda/varlık görünümünde gösterir."],
      ["Targets", "Radarın o anda gördüğü hedef sayısıdır."],
      ["Direction", "Seçili hedefe göre SOL/SAG/ORTA gibi yön yorumudur."],
      ["Selected", "Robotun takip/karar için seçtiği hedef indeksidir."],
      ["Angle", "Seçili hedefin yaklaşık açısıdır."],
      ["Radar detail", "Karar mesafesi, x/y, açı, güven, frame ve boş oda kalibrasyon bilgisini yazar."]
    ]
  },
  pipelineFields: {
    title: "Voice Pipeline kontrolleri",
    body: ["Bu panel kısa canlı özet ve manuel pipeline testleri içindir; uzun konuşma geçmişi Conversation panelindedir."],
    items: [
      ["SESSION pill", "Canlı konuşma oturumunun açık/kapalı durumunu gösterir."],
      ["STREAM pill", "TTS/audio stream hattının aktif olup olmadığını gösterir."],
      ["Pipeline input", "Elle test metni yazılır. Wake veya mikrofon kullanmadan pipeline denenir."],
      ["LLM + TTS", "Yazdığın metni LLM'e gönderir, üretilen cevabı aktif TTS ile ESP'ye okutmaya çalışır."],
      ["TTS only", "Yazdığın metni LLM'e sokmadan doğrudan aktif TTS ile okutur."],
      ["Latency test", "Sabit kısa metinle TTS gecikme benchmark'ı çalıştırır ve Latency Timeline'a ölçüm düşürür."],
      ["Start session", "Canlı voice oturumunu manuel başlatır."],
      ["Stop session", "Canlı voice oturumunu kapatır."],
      ["Cancel response", "Devam eden Realtime/voice cevabını keser; barge-in davranışını test etmek için kullanılır."],
      ["Session meta", "Realtime bağlantı durumu, model ve son event özetini gösterir."],
      ["STATE", "Pipeline'ın idle, tts idle, live websocket sayısı ve son bağlantı durumunu gösterir."],
      ["STT", "Son kullanıcı transkripti veya manuel input sonucunu gösterir; yeni tur gelene kadar son değer kalır."],
      ["LLM", "Son asistan cevabını gösterir."],
      ["TTS", "Son TTS'e giden metni ve provider/rate durumunu gösterir."]
    ]
  },
  commandsFields: {
    title: "Command Panel kontrolleri",
    body: ["Daily günlük kullanım, Advanced tanılama ve bakım içindir."],
    items: [
      ["Daily tab", "Sürüş, ses, wake, reaksiyonlar, sleep ve temel bağlantı komutlarını gösterir."],
      ["Advanced tab", "Behavior efektleri, ESP test komutları, server bakım ve mic debug araçlarını gösterir."],
      ["Speaker Volume", "ESP hoparlör gain seviyesini yüzde olarak ayarlar. Panel son bilinen değeri hatırlar."],
      ["Mute", "Hoparlör sesini sessize alır veya önceki seviyeye geri getirir."],
      ["Motors", "Motor sürüş hattını aktif/pasif yapar."],
      ["Motion lock", "Panelden sürüş komutlarını kilitler; yanlışlıkla hareketi engeller."],
      ["D-pad", "Forward, Back, Left, Right ve Stop motor komutlarını gönderir."],
      ["Speed", "Motor komutlarına yavaş/normal/hızlı niyetini ekler. Firmware destekledikçe etkisi artar."],
      ["Step", "Kısa/orta/uzun hareket süresi niyetini ekler."],
      ["Listen", "Manuel dinleme oturumu başlatır veya durdurur."],
      ["Follow-up", "Cevap sonrası tekrar dinleme penceresini açar/kapatır."],
      ["Touch", "Dokunma reaksiyonlarını açar/kapatır."],
      ["Lift", "Havaya kaldırma reaksiyonlarını açar/kapatır."],
      ["Wake", "Wake word dinlemeyi açar/kapatır."],
      ["Sleep", "Soft sleep moduna alır veya aktif moda döndürür."],
      ["Reconnect", "ESP bağlantısını manuel yeniden dener."],
      ["Reboot", "ESP'yi yeniden başlatır."],
      ["Behavior buttons", "Göz/duygu davranışlarını test eder: happy, curious, thinking, love, normal gibi."],
      ["ESP commands", "Hoparlör, mikrofon, wake, amp, radar kalibrasyon ve diğer düşük seviye ESP komutlarını gönderir."],
      ["Server maintenance", "STT/TTS restart, prompt reload, voice session ve safe mode gibi add-on tarafı komutlarıdır."],
      ["Record L / Record R", "Sol veya sağ mikrofon kanalından kısa WAV debug kaydı ister."],
      ["Play L / Play R", "Son yakalanan sol/sağ kanal kaydını panelde çalar."],
      ["Download L / Download R", "Son debug WAV kaydını indirir."],
      ["Mic debug meta", "Kayıt sonrası duration, byte, RMS, peak, shift ve clip bilgisini gösterir."]
    ]
  },
  latencyFields: {
    title: "Latency Timeline detayları",
    body: ["Buradaki ölçümler süre için monotonic clock, okunabilir saat için wall-clock kullanır."],
    items: [
      ["Wake -> mic", "Wake/manual start sonrası backend'in ilk mikrofon paketini gördüğü süre."],
      ["Speech -> STT", "Konuşma bitişinden STT sonucuna kadar geçen süre."],
      ["STT -> LLM", "Transkriptin LLM'e gitmesi ve LLM'in cevap üretmeye başlaması arasındaki süre."],
      ["LLM -> TTS text", "LLM'in ilk/son kullanılabilir cevabından TTS metninin kuyruğa alınmasına kadar geçen süre."],
      ["TTS req -> headers", "Google/aktif TTS isteği başladıktan HTTP response header gelene kadar geçen süre."],
      ["TTS req -> first byte", "TTS isteğinden provider'ın ilk byte'ına kadar geçen süre."],
      ["TTS req -> audio", "TTS isteğinden ilk gerçek audio chunk'ın bulunmasına kadar geçen süre."],
      ["Audio -> ESP chunk", "Audio decode/convert sonrası ilk chunk'ın ESP'ye gönderilmesine kadar geçen süre."],
      ["ESP chunk -> speaker", "ESP'ye ilk chunk gittikten hoparlörün ilk PCM/speaker started bildirimi yapmasına kadar geçen süre."],
      ["ESP chunk -> finish", "İlk ESP chunk'tan speaker finished bildirimine kadar geçen süre."],
      ["TTS text -> speaker", "TTS metni hazırlandıktan hoparlörden ilk ses başlangıcına kadar geçen toplam TTS/ESP süresi."],
      ["TTS text -> finish", "TTS metninden ses oynatımının bitmesine kadar geçen süre."],
      ["Wake -> speaker", "Wake veya manuel turn başlangıcından hoparlörün ilk sesine kadar geçen gerçek algılanan gecikme."],
      ["Wake -> finish", "Turn başlangıcından ses bitimine kadar geçen süre."],
      ["Turn total", "Backend'in o turn için bildirdiği toplam süre."],
      ["Event list", "Her event için saat, turn başlangıcına göre +ms ve açıklama gösterir."],
      ["Recent turns", "Son turların kısa özetidir; hangi turu analiz ettiğini anlamaya yardım eder."]
    ]
  },
  conversationFields: {
    title: "Conversation kontrolleri",
    body: ["Teknik log değil, konuşma metni geçmişidir."],
    items: [
      ["USER rows", "STT veya manuel input ile gelen kullanıcı metnidir."],
      ["ASSISTANT rows", "Realtime veya klasik LLM cevabıdır."],
      ["TTS rows", "TTS'e gönderilen metin veya HA route gibi doğrudan seslendirilecek cevap olabilir."],
      ["Provider/source", "Satırda openai realtime, HA route veya tts gibi kaynağı gösterir."],
      ["Download", "Konuşma geçmişini metin dosyası olarak indirir."],
      ["Clear", "Conversation panelindeki geçmişi temizler; backend config veya promptları etkilemez."],
      ["Auto-scroll", "Yeni mesaj geldikçe log paneli gibi en alta kayar; kullanıcı yukarı kaydırdıysa konumu korumaya çalışır."]
    ]
  },
  promptsFields: {
    title: "Prompt Editor kontrolleri",
    body: ["Prompt profilleri Alice'in genel kişiliğini ve fallback system talimatını taşır."],
    items: [
      ["Profile select", "Düzenlenecek prompt profilini seçer."],
      ["Name", "Profilin dosya/başlık adıdır."],
      ["Description", "Profilin kısa açıklamasıdır; çalışma mantığını etkilemeyebilir ama yönetimi kolaylaştırır."],
      ["Prompt text", "Alice'in genel karakter ve davranış metnidir."],
      ["New", "Boş yeni profil oluşturur."],
      ["Copy", "Seçili profili yeni bir profile kopyalar."],
      ["Delete", "Seçili profili siler. Aktif profil silinirse başka profil seçmek gerekir."],
      ["Activate", "Seçili profili fallback aktif prompt yapar."],
      ["Save", "Profil metnini /data/prompts altına kaydeder."]
    ]
  },
  configFields: {
    title: "Config kontrolleri",
    body: ["Config panelindeki değişiklikler Save ile kalıcı hale gelir."],
    items: [
      ["Secrets", "Export sırasında gerçek API key/token değerlerini dahil eder. Kapalıyken secretlar maskelenir."],
      ["Import", "JSON config dosyası seçip içe aktarır."],
      ["Export", "Mevcut config'i indirir."],
      ["Save", "Ekrandaki config değişikliklerini /data/alice_config.json içine yazar."],
      ["Provider switches", "Live, LLM ve TTS kartlarında hangi provider alanlarının düzenleneceğini seçer."],
      ["Password/API key inputs", "Gizli değerleri password input olarak tutar; repo içine yazılmaz."],
      ["Checkboxes", "Özellikleri aç/kapatır; bazıları backend'de hemen etkili olur, bazıları ESP reconnect/yeniden komut bekleyebilir."]
    ]
  },
  panelEspFields: {
    title: "Panel & ESP alanları",
    body: ["Panel erişimi ve ESP bağlantısının temel adresleri burada tutulur."],
    items: [
      ["Panel port", "Web panelinin dinlediği porttur. Varsayılan 8099."],
      ["Panel token", "REST, WebSocket ve UI erişimi için basit token korumasıdır. Boşsa auth kapalı kalabilir."],
      ["Panel password", "UI için basit password korumasıdır."],
      ["ESP base URL", "Robotun HTTP API adresidir; status poll ve /api/command buradan gider."],
      ["ESP WebSocket URL", "Robotun canlı /ws adresidir; event, log, mic debug ve audio stream için kullanılır."],
      ["ESP max auto reconnects", "Otomatik reconnect deneme limitidir. Limit dolunca manuel reconnect beklenir."],
      ["ESP audio ACK timeout sec", "ESP audio start/chunk onayı için beklenecek süredir. Kısa olursa yavaş ağda erken hata verebilir."],
      ["Debug logs", "Daha ayrıntılı log üretir."],
      ["Safe mode", "Riskli/aktif davranışları azaltmak için güvenli çalışma anahtarıdır."]
    ]
  },
  powerSleepFields: {
    title: "Power & Sleep alanları",
    body: ["Bu ayarlar panelin ESP'ye otomatik uyku/uyanma komutu gönderip göndermeyeceğini belirler."],
    items: [
      ["Power scheduler", "Zamana veya idle durumuna göre otomatik sleep kararlarını tamamen açar/kapatır."],
      ["Soft sleep after idle", "Robot belirlenen süre boyunca kullanılmazsa soft sleep moduna geçmeyi dener."],
      ["Night sleep schedule", "Night start ve Night end arasındaki saatlerde night sleep modunu hedefler."],
      ["Soft idle minutes", "Soft sleep için kaç dakika aktivite olmaması gerektiğidir."],
      ["Night start", "Gece uyku penceresinin başlangıç saatidir."],
      ["Night end", "Gece uyku penceresinin bitiş saatidir."],
      ["Radar presence", "Radar taze hedef/kişi görüyorsa gündüz soft sleep engellenir. Soft sleep sırasında radar kişi görürse power manager sleep_mode_off göndererek uyandırır."],
      ["Öncelik", "Night sleep saat aralığı soft sleep ve radar presence kuralından daha baskındır; saat aralığı bitince aktif moda dönülür."]
    ]
  },
  liveVoiceFields: {
    title: "Live Voice alanları",
    body: ["Live Voice ayarları wake sonrası canlı konuşma oturumunun modelini, VAD kararını ve prompt fallback'ini belirler."],
    items: [
      ["OpenAI Live / Gemini Live / None", "Aktif canlı provider görünümünü seçer. None live hattı kapatır."],
      ["Active live", "Gerçekte kullanılacak realtime provider değeridir."],
      ["Input rate", "ESP mikrofon PCM sample rate değeridir. ESP ile uyumlu olmalıdır."],
      ["Response timeout ms", "Realtime cevap beklerken üst zaman sınırıdır."],
      ["Transcript wait ms", "STT transkripti geç gelirse kısa süre daha beklemek için kullanılır."],
      ["Turn detection", "Konuşma bitiş kararını server_vad veya semantic_vad ile belirler."],
      ["Semantic eagerness", "semantic_vad seçiliyken modelin turn kapatmaya ne kadar istekli olduğunu ayarlar."],
      ["VAD threshold", "server_vad hassasiyetidir. Düşük değer daha kolay tetikler, yüksek değer daha seçicidir."],
      ["Silence ms", "Konuşma bittikten sonra turn kapanması için gereken sessizlik süresidir."],
      ["Prefix padding ms", "Konuşma başındaki heceleri kaçırmamak için önceki kısa sesi de dahil eder."],
      ["Noise reduction", "Realtime input noise reduction modudur: near_field, far_field veya none."],
      ["Live instructions", "Canlı modele verilecek kişilik/davranış talimatıdır."],
      ["OpenAI API key", "OpenAI Realtime erişim anahtarıdır."],
      ["OpenAI Model", "Canlı konuşma modelidir; örneğin gpt-realtime-2."],
      ["Realtime WS URL", "OpenAI Realtime WebSocket endpointidir; özel proxy yoksa varsayılan kalabilir."],
      ["Realtime STT model", "OpenAI tarafındaki realtime transkripsiyon modelidir."],
      ["Realtime STT prompt", "Transkripsiyon ipucudur; Türkçe isimler ve sık yanlış duyulan kelimeler için kullanılır."],
      ["Gemini API key", "Gemini Live profili için API keydir."],
      ["Gemini model", "Gemini Live için model adıdır."],
      ["Voice name", "Gemini Live ses profili adıdır."],
      ["API version", "Gemini Live API versiyonudur."],
      ["Output rate", "Gemini Live audio output sample rate hedefidir."],
      ["Start/End sensitivity", "Gemini Live konuşma başlangıç ve bitiş hassasiyetidir."]
    ]
  },
  sttVadFields: {
    title: "Classic STT & VAD alanları",
    body: ["Classic STT ayarları faster-whisper ve yerel VAD denemeleri içindir."],
    items: [
      ["STT provider", "Kullanılacak classic STT motoru. Şu an faster_whisper hedeflenir."],
      ["STT model", "Whisper model boyutudur; küçük model hızlı, büyük model daha doğru olabilir."],
      ["Model cache", "Model dosyalarının saklanacağı dizindir; tekrar indirmeyi önler."],
      ["Language", "Transkripsiyon dili. Türkçe için tr kullanmak dil kaymasını azaltır."],
      ["Compute type", "Model hesaplama hassasiyetidir; int8 daha hafif, float türleri daha ağırdır."],
      ["Beam size", "STT arama genişliğidir; artırmak gecikmeyi yükseltebilir."],
      ["STT VAD filter", "Whisper/faster-whisper tarafındaki ek VAD filtresini açar."],
      ["Live VAD provider", "silero veya energy tabanlı canlı VAD kararını seçer."],
      ["Silero start prob", "Silero'nun konuşma başladı demesi için gereken olasılık eşiği."],
      ["Silero end prob", "Silero'nun konuşma bitti demesi için gereken olasılık eşiği."],
      ["Energy start RMS", "Energy VAD'de konuşma başlangıç RMS eşiği."],
      ["Energy end RMS", "Energy VAD'de konuşma bitiş RMS eşiği."],
      ["Live silence ms", "Konuşma bitişi için gereken sessizlik süresi."],
      ["Live max utterance ms", "Tek konuşma turn'ü için maksimum süre."],
      ["Live mic WS", "ESP'den canlı mikrofon WebSocket akışını kabul eder."],
      ["Live VAD", "Canlı mikrofon akışında VAD kararını aktif eder."]
    ]
  },
  homeAssistantFields: {
    title: "Home Assistant alanları",
    body: ["HA Bridge allowlist dışına çıkmadan Home Assistant kontrolü yapmayı hedefler."],
    items: [
      ["HA API base", "Add-on içinden Home Assistant API adresidir."],
      ["HA bridge", "Home Assistant köprüsünü açar/kapatır."],
      ["Route home control", "Ev kontrolü/hava durumu niyetlerini LLM'den önce HA bridge tarafında yakalamaya çalışır."],
      ["Allowed entity IDs", "Alice'in görebileceği tek entity listesidir. Burada olmayan entity okunmaz/kontrol edilmez."],
      ["Entity aliases", "Allowlist'teki entityler için Türkçe takma adlardır. Yeni izin vermez, sadece eşleştirmeyi iyileştirir."],
      ["Weather kullanımı", "weather.* entity allowlist içindeyse hava durumu sorularında state ve attribute bilgileri bu yoldan okunabilir."],
      ["Servis çağrıları", "Işık, switch ve benzeri komutlar allowlist'teki entityye bağlı güvenli servis çağrısına çevrilir."]
    ]
  },
  llmFields: {
    title: "LLM alanları",
    body: ["Bu alanlar klasik metin modeli hattını yönetir; live konuşma açıkken ana cevap OpenAI Live'dan gelebilir."],
    items: [
      ["Provider buttons", "Düzenlenecek provider kartını seçer."],
      ["Active LLM", "Klasik pipeline'ın kullanacağı provider değeridir."],
      ["Temperature", "Cevap yaratıcılığıdır; düşük değer daha tutarlı, yüksek değer daha serbesttir."],
      ["Streaming", "LLM cevabını parça parça almaya çalışır."],
      ["LLM system prompt", "Doluysa aktif Prompt Editor profilinin önüne geçer."],
      ["OpenAI", "OpenAI API key, model ve base URL bilgileri."],
      ["OpenRouter", "OpenRouter üzerinden farklı modelleri denemek için API key/model/base URL."],
      ["Groq", "Groq'un OpenAI uyumlu chat endpoint'i için API key/model/base URL."],
      ["Gemini", "Gemini classic text modeli için API key/model/base URL."],
      ["OpenAI Compatible", "LM Studio, Ollama proxy, vLLM veya başka uyumlu endpointler için genel profil."],
      ["Mock", "Gerçek LLM çağrısı yapmadan test cevabı üretir."],
      ["None", "Klasik LLM hattını kapatır veya sadece özel route/fallback kullanımına bırakır."]
    ]
  },
  ttsFields: {
    title: "TTS alanları",
    body: ["TTS profilleri ayrı saklanır; aktif provider dışında kalan key ve ayarlar korunur."],
    items: [
      ["Active TTS", "Kullanılacak ses sağlayıcısını seçer."],
      ["PCM rate", "ESP'ye hedeflenen PCM sample rate bilgisidir."],
      ["ESP start buffer ms", "ESP tarafında ses başlamadan önce toplanacak tampon süresidir."],
      ["ESP silence prefix ms", "Ses başına eklenecek kısa sessizliktir; ilk hece kesilmesini azaltabilir."],
      ["Mic response", "Mic capture sonrası asistan cevabı, echo veya echo + assistant davranışını seçer."],
      ["TTS enabled", "Kapalıysa metin üretilebilir ama seslendirme atlanır."],
      ["Stream TTS to ESP", "Sesin ESP'ye WebSocket üzerinden gönderilip gönderilmeyeceğini belirler."],
      ["Barge-in cancel", "Yeni konuşma algılanınca mevcut cevabı kesmeye izin verir."],
      ["OpenAI API/model/voice", "OpenAI TTS profilidir. Instructions seslendirme tarzını yönlendirebilir."],
      ["Cartesia API/model/voice", "Cartesia profilidir; kredi/limit hataları bu provider'dan gelebilir."],
      ["ElevenLabs API/model/voice", "ElevenLabs profilidir. Output format ve latency mode ses biçimi/gecikme dengesini ayarlar."],
      ["Google AI API/model/voice", "AI Studio tabanlı Gemini TTS profilidir. Stream destekli yanıtlarda ilk audio süresi Latency Timeline'da ölçülür."],
      ["Google AI prompt prefix", "Sadece seslendirme metninin üslubunu yönlendirmek içindir; uzun karakter promptunu buraya yığmak yerine Prompt Editor/Live instructions kullan."],
      ["Google Cloud credentials", "Google Cloud TTS service account JSON kimliğidir."],
      ["Google Cloud voice/language/gender", "Cloud TTS ses adı, dil kodu ve SSML gender seçimidir."]
    ]
  }
});

Object.assign(HELP_TEXTS, {
  connections: {
    title: "Connections",
    body: [
      "Canli WebSocket, ses hatti ve provider durumlarinin kisa baglanti ozetidir.",
      "Robot online ve Wi-Fi bilgisi ust durum kartlarinda, HA allowlist ise Config/HA ayarlarinda takip edilir; burada tekrar edilmez.",
      "Baglanti koparsa ESP WS satiri, reconnect sayaci, son hata satiri ve Logs paneli birlikte ipucu verir."
    ]
  },
  espHealth: {
    title: "ESP Health",
    body: [
      "ESP tarafindan bildirilen hafif sistem sagligi ozetidir. Sicaklik, CPU, internal RAM, PSRAM ve son reset sebebini tek bakista gosterir.",
      "Bu panel terminaldeki SYS_MON bilgisinin kisa web karsiligidir. Resetler, isiya bagli riskler veya bellek daralmasi gibi ipuclari icin kullanilir.",
      "OK normal, WARM/HOT sicaklik uyarisi, CHECK ise watchdog/brownout/panic gibi dikkat isteyen son reset sebebi anlamina gelir."
    ]
  }
});

Object.assign(HELP_DETAIL_TEXTS, {
  connectionsFields: {
    title: "Connections detaylari",
    body: [
      "Bu panel tekrar eden Robot/Wi-Fi/HA allowlist bilgisini tasimaz; yalnizca canli servis ve aktarim kanallarini ozetler.",
      "Kopma veya zaman asimi olursa kisa durum burada, ayrintili hata ise Logs panelinde gorunur."
    ],
    items: [
      ["ESP WS", "Robotun /ws canli WebSocket baglantisidir. Audio, event, log ve mic debug icin kritik yoldur."],
      ["STT", "Aktif konusmayi metne cevirme motorunu gosterir. Live Voice aciksa realtime STT modeli burada ozetlenir."],
      ["LLM", "Aktif cevap uretme hatti ve modelidir. Live Voice aciksa realtime model burada gorunur."],
      ["TTS", "Aktif TTS provider ve hedef PCM rate ozetidir."],
      ["Reconnects", "ESP kopunca yapilan otomatik reconnect sayisi ve limitidir."]
    ]
  }
});

const $ = (id) => document.getElementById(id);
const text = (id, value) => { const el = $(id); if (el) el.textContent = value ?? "-"; };

function isNearBottom(el, threshold = 28) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function initAutoScrollContainers() {
  document.querySelectorAll("[data-autoscroll]").forEach((el) => {
    if (autoScrollState.has(el)) return;
    const state = { pinned: true };
    autoScrollState.set(el, state);
    el.addEventListener("scroll", () => {
      state.pinned = isNearBottom(el);
    }, { passive: true });
  });
}

function keepAutoScrolled(el, mutate, force = false) {
  if (!el) return;
  const state = autoScrollState.get(el);
  const shouldStick = force || !state || state.pinned || isNearBottom(el);
  const distanceFromBottom = el.scrollHeight - el.scrollTop;
  mutate();
  if (!shouldStick) {
    window.requestAnimationFrame(() => {
      el.scrollTop = Math.max(0, el.scrollHeight - distanceFromBottom);
    });
    return;
  }
  window.requestAnimationFrame(() => {
    el.scrollTop = el.scrollHeight;
    if (state) state.pinned = true;
  });
}

function keepChildVisible(scroller, child, padding = 12) {
  if (!scroller || !child) return;
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      const scrollerRect = scroller.getBoundingClientRect();
      const childRect = child.getBoundingClientRect();
      const bottomOverflow = childRect.bottom - scrollerRect.bottom + padding;
      const topOverflow = scrollerRect.top - childRect.top + padding;
      if (bottomOverflow > 0) {
        scroller.scrollTop += bottomOverflow;
      } else if (topOverflow > 0) {
        scroller.scrollTop = Math.max(0, scroller.scrollTop - topOverflow);
      }
      const state = autoScrollState.get(scroller);
      if (state) state.pinned = isNearBottom(scroller);
    });
  });
}

function setAutoText(id, value) {
  const el = $(id);
  keepAutoScrolled(el, () => { el.textContent = value ?? "-"; });
}

function initHelpBubbles() {
  HELP_TARGETS.forEach(([selector, key, detailKey]) => {
    const heading = document.querySelector(selector);
    if (!heading || heading.dataset.helpAttached) return;
    const parent = heading.parentElement;
    if (!parent) return;
    const titleRow = document.createElement("div");
    titleRow.className = "help-title";
    parent.insertBefore(titleRow, heading);
    titleRow.appendChild(heading);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "help-trigger";
    button.dataset.help = key;
    button.setAttribute("aria-label", `${HELP_TEXTS[key]?.title || "Panel"} yardimi`);
    button.textContent = "?";
    button.onclick = (event) => {
      event.stopPropagation();
      toggleHelpBubble(key, button);
    };
    titleRow.appendChild(button);
    if (detailKey && HELP_DETAIL_TEXTS[detailKey]) {
      const detailButton = document.createElement("button");
      detailButton.type = "button";
      detailButton.className = "help-trigger help-trigger-detail";
      detailButton.dataset.help = detailKey;
      detailButton.setAttribute("aria-label", `${HELP_DETAIL_TEXTS[detailKey].title} detaylari`);
      detailButton.textContent = "??";
      detailButton.onclick = (event) => {
        event.stopPropagation();
        toggleHelpBubble(detailKey, detailButton, true);
      };
      titleRow.appendChild(detailButton);
    }
    heading.dataset.helpAttached = "true";
  });

  document.addEventListener("click", (event) => {
    if (helpPopover?.contains(event.target) || event.target.closest?.(".help-trigger")) return;
    closeHelpBubble();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeHelpBubble();
  });
  window.addEventListener("resize", closeHelpBubble);
}

function ensureHelpPopover() {
  if (helpPopover) return helpPopover;
  helpPopover = document.createElement("section");
  helpPopover.id = "help-popover";
  helpPopover.className = "help-popover hidden";
  helpPopover.setAttribute("role", "dialog");
  helpPopover.setAttribute("aria-live", "polite");
  document.body.appendChild(helpPopover);
  return helpPopover;
}

function toggleHelpBubble(key, anchor, isDetail = false) {
  const popover = ensureHelpPopover();
  if (!popover.classList.contains("hidden") && popover.dataset.helpKey === key) {
    closeHelpBubble();
    return;
  }
  openHelpBubble(key, anchor, isDetail);
}

function openHelpBubble(key, anchor, isDetail = false) {
  const doc = isDetail ? HELP_DETAIL_TEXTS[key] : HELP_TEXTS[key];
  if (!doc) return;
  const popover = ensureHelpPopover();
  popover.dataset.helpKey = key;
  popover.innerHTML = "";
  popover.classList.toggle("detail", Boolean(isDetail || doc.items?.length));

  const header = document.createElement("header");
  const title = document.createElement("h3");
  title.textContent = doc.title;
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Kapat";
  close.setAttribute("aria-label", "Yardimi kapat");
  close.onclick = closeHelpBubble;
  header.append(title, close);
  popover.appendChild(header);

  doc.body.forEach((paragraph) => {
    const p = document.createElement("p");
    p.textContent = paragraph;
    popover.appendChild(p);
  });

  if (doc.items?.length) {
    const list = document.createElement("ul");
    list.className = "help-detail-list";
    doc.items.forEach(([label, description]) => {
      const item = document.createElement("li");
      const name = document.createElement("strong");
      name.textContent = label;
      const detail = document.createElement("span");
      detail.textContent = description;
      item.append(name, detail);
      list.appendChild(item);
    });
    popover.appendChild(list);
  }

  popover.classList.remove("hidden");
  window.requestAnimationFrame(() => {
    const rect = anchor.getBoundingClientRect();
    const gap = 8;
    const margin = 12;
    const width = popover.offsetWidth;
    const height = popover.offsetHeight;
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - width - margin));
    let top = rect.bottom + gap;
    if (top + height > window.innerHeight - margin) {
      top = rect.top - height - gap;
    }
    if (top < margin) top = margin;
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
  });
}

function closeHelpBubble() {
  if (!helpPopover) return;
  helpPopover.classList.add("hidden");
  helpPopover.classList.remove("detail");
  helpPopover.dataset.helpKey = "";
}

function notice(value) {
  const el = $("notice");
  if (!value) { el.classList.add("hidden"); return; }
  el.textContent = value;
  el.classList.remove("hidden");
}

function rememberToken(value) {
  token = value || "";
  if (token) {
    localStorage.setItem("alice_panel_token", token);
    document.cookie = `alice_panel_token=${encodeURIComponent(token)}; path=/; SameSite=Lax`;
  } else {
    localStorage.removeItem("alice_panel_token");
    document.cookie = "alice_panel_token=; Max-Age=0; path=/; SameSite=Lax";
  }
}

async function guard(label, fn) {
  try {
    return await fn();
  } catch (err) {
    notice(`${label}: ${err.message}`);
    return null;
  }
}

function fmtSeconds(value) {
  const total = Number(value || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = Math.floor(total % 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtMs(value) {
  const total = Number(value);
  return Number.isFinite(total) ? `${Math.round(total)}ms` : "-";
}

function fmtClock(value, withMs = false) {
  const ts = Number(value || 0);
  if (!Number.isFinite(ts) || ts <= 0) return "--:--:--";
  const date = new Date(ts * 1000);
  if (!withMs) return date.toLocaleTimeString();
  const base = date.toLocaleTimeString();
  return `${base}.${String(date.getMilliseconds()).padStart(3, "0")}`;
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatBytes(value) {
  const bytes = finiteNumber(value);
  if (bytes === null) return "n/a";
  const units = ["B", "KB", "MB"];
  let current = bytes;
  let unitIndex = 0;
  while (Math.abs(current) >= 1024 && unitIndex < units.length - 1) {
    current /= 1024;
    unitIndex += 1;
  }
  const fixed = unitIndex === 0 ? current.toFixed(0) : current < 10 ? current.toFixed(2) : current.toFixed(1);
  return `${fixed} ${units[unitIndex]}`;
}

function tone(value) {
  const key = String(value || "").toLowerCase();
  if (key.includes("online") || key.includes("idle") || key.includes("ok")) return "good";
  if (key.includes("error") || key.includes("offline")) return "bad";
  if (key.includes("mock") || key.includes("warn")) return "warn";
  return "info";
}

function setPill(id, value, forcedTone) {
  const el = $(id);
  if (!el) return;
  el.textContent = value;
  el.className = `pill ${forcedTone || tone(value)}`;
}

async function api(path, options = {}, auth = token) {
  const headers = new Headers(options.headers || {});
  if (auth) headers.set("X-Alice-Token", auth);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
  const contentType = resp.headers.get("content-type") || "";
  return contentType.includes("application/json") ? resp.json() : resp.text();
}

function wsPath(path) {
  const url = new URL(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`);
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

function cacheBustedPath(path) {
  const url = new URL(path, location.origin);
  if (token) url.searchParams.set("token", token);
  url.searchParams.set("t", String(Date.now()));
  return url.pathname + url.search;
}

function readRadarCalibration() {
  try {
    const saved = JSON.parse(localStorage.getItem(RADAR_CALIBRATION_KEY) || "{}");
    return {
      invertX: Boolean(saved.invertX),
      invertY: Boolean(saved.invertY),
      rotate180: Boolean(saved.rotate180),
    };
  } catch {
    return { invertX: false, invertY: false, rotate180: false };
  }
}

function saveRadarCalibration() {
  localStorage.setItem(RADAR_CALIBRATION_KEY, JSON.stringify(radarCalibration));
}

function radarApplyCalibrationXY(xMm, yMm) {
  let x = Number.isFinite(xMm) ? xMm : 0;
  let y = Number.isFinite(yMm) ? yMm : 0;
  if (radarCalibration.invertX) x = -x;
  if (radarCalibration.invertY) y = -y;
  if (radarCalibration.rotate180) {
    x = -x;
    y = -y;
  }
  return { x_mm: x, y_mm: y };
}

function radarNeedsLegacyXFlip(info) {
  return info?.axis_x_inverted !== true;
}

function radarApplyBaseXY(info, xMm, yMm) {
  return {
    x_mm: radarNeedsLegacyXFlip(info) ? -xMm : xMm,
    y_mm: yMm,
  };
}

function radarNormalizeTarget(info, target) {
  if (!target) return null;
  const rawX = radarTargetNumber(target, "x_mm");
  const rawY = radarTargetY(target);
  const point = radarApplyBaseXY(info, rawX, rawY);
  return {
    ...target,
    raw_x_mm: rawX,
    raw_y_mm: rawY,
    x_mm: point.x_mm,
    y_mm: point.y_mm,
    distance_mm: radarTargetDistance(point),
    angle_deg: radarAngleDeg(point),
  };
}

function radarApplyRoomXY(xMm, yMm) {
  return { x_mm: xMm, y_mm: yMm };
}

function getDeep(obj, path) {
  return path.split(".").reduce((acc, key) => acc && acc[key], obj);
}

function setDeep(obj, path, value) {
  const keys = path.split(".");
  let cursor = obj;
  keys.slice(0, -1).forEach((key) => {
    if (!cursor[key] || typeof cursor[key] !== "object") cursor[key] = {};
    cursor = cursor[key];
  });
  cursor[keys[keys.length - 1]] = value;
}

function stripMasked(value) {
  if (Array.isArray(value)) return value.map(stripMasked);
  if (value && typeof value === "object") {
    const out = {};
    Object.entries(value).forEach(([key, item]) => {
      if (item !== "********") out[key] = stripMasked(item);
    });
    return out;
  }
  return value;
}

function syncRadarControls() {
  $("radar")?.classList.toggle("room-mode", radarView === "room");
  document.querySelectorAll("[data-radar-view]").forEach((button) => {
    const active = button.dataset.radarView === radarView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-radar-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.radarPanel === radarView);
  });
  document.querySelectorAll("[data-radar-cal]").forEach((button) => {
    const key = button.dataset.radarCal;
    button.classList.toggle("active", Boolean(radarCalibration[key]));
    button.setAttribute("aria-pressed", radarCalibration[key] ? "true" : "false");
  });
}

function initRadarControls() {
  if (!["tech", "room"].includes(radarView)) radarView = "tech";
  document.querySelectorAll("[data-radar-view]").forEach((button) => {
    button.onclick = () => {
      radarView = button.dataset.radarView || "tech";
      localStorage.setItem("alice_radar_view", radarView);
      syncRadarControls();
      if (latestRadarDraw) drawRadarViews(latestRadarDraw);
    };
  });
  document.querySelectorAll("[data-radar-cal]").forEach((button) => {
    button.onclick = () => {
      const key = button.dataset.radarCal;
      radarCalibration[key] = !radarCalibration[key];
      saveRadarCalibration();
      syncRadarControls();
      if (latestRadar) renderRadar(latestRadar);
    };
  });
  syncRadarControls();
}

function syncCommandTabs() {
  if (!["daily", "advanced"].includes(commandTab)) commandTab = "daily";
  document.querySelectorAll("[data-command-tab]").forEach((button) => {
    const active = button.dataset.commandTab === commandTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-command-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.commandPanel === commandTab);
  });
}

function initCommandTabs() {
  document.querySelectorAll("[data-command-tab]").forEach((button) => {
    button.onclick = () => {
      commandTab = button.dataset.commandTab || "daily";
      localStorage.setItem("alice_command_tab", commandTab);
      syncCommandTabs();
    };
  });
  syncCommandTabs();
}

function syncPipelineTabs() {
  pipelineView = "trace";
  const pipeline = $("pipeline");
  pipeline?.classList.remove("timing-expanded", "messages-view");
  window.requestAnimationFrame(() => {
    const messages = $("pipeline-messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  });
}

function initPipelineTabs() {
  localStorage.setItem("alice_pipeline_view", "trace");
  syncPipelineTabs();
}

function syncCollapsiblePanel(panel) {
  const button = panel.querySelector("[data-panel-toggle]");
  if (!button) return;
  const expanded = !panel.classList.contains("is-collapsed");
  button.textContent = expanded ? "-" : "+";
  button.title = expanded ? "Collapse panel" : "Expand panel";
  button.setAttribute("aria-label", expanded ? "Collapse panel" : "Expand panel");
  button.setAttribute("aria-expanded", expanded ? "true" : "false");
}

function expandCollapsiblePanel(panel) {
  if (!panel) return;
  panel.classList.remove("is-collapsed");
  syncCollapsiblePanel(panel);
}

function expandPanelFromHash() {
  const id = window.location.hash ? window.location.hash.slice(1) : "";
  if (!id) return;
  const target = document.getElementById(id);
  const panel = target?.closest("[data-collapsible-panel]");
  if (panel) expandCollapsiblePanel(panel);
}

function initCollapsiblePanels() {
  document.querySelectorAll("[data-collapsible-panel]").forEach((panel) => {
    const button = panel.querySelector("[data-panel-toggle]");
    if (!button) return;
    panel.classList.add("is-collapsed");
    button.onclick = () => {
      panel.classList.toggle("is-collapsed");
      syncCollapsiblePanel(panel);
      if (!panel.classList.contains("is-collapsed")) {
        window.requestAnimationFrame(() => {
          panel.scrollIntoView({ block: "nearest" });
          const autoscroll = panel.querySelector("[data-autoscroll]");
          if (autoscroll) autoscroll.scrollTo({ top: autoscroll.scrollHeight });
        });
      }
    };
    syncCollapsiblePanel(panel);
  });
  window.addEventListener("hashchange", expandPanelFromHash);
  expandPanelFromHash();
}

function readStoredIndex(key, fallback) {
  const value = Number(localStorage.getItem(key));
  return Number.isFinite(value) ? Math.max(0, Math.min(2, Math.round(value))) : fallback;
}

function syncDriveControls() {
  const speed = $("drive-speed");
  const step = $("drive-step");
  const lock = $("motion-lock");
  const driveCard = document.querySelector(".drive-card");
  if (speed) speed.value = String(driveSpeedIndex);
  if (step) step.value = String(driveStepIndex);
  text("drive-speed-label", DRIVE_SPEED_LABELS[driveSpeedIndex] || DRIVE_SPEED_LABELS[1]);
  text("drive-step-label", DRIVE_STEP_LABELS[driveStepIndex] || DRIVE_STEP_LABELS[1]);
  if (driveCard) driveCard.classList.toggle("motion-locked", motionLocked);
  if (lock) {
    lock.classList.toggle("active", motionLocked);
    lock.setAttribute("aria-pressed", motionLocked ? "true" : "false");
    lock.textContent = motionLocked ? "Motion locked" : "Motion lock off";
    lock.title = motionLocked ? "Panel drive commands are blocked" : "Panel drive commands are allowed";
  }
}

function initDriveControls() {
  const speed = $("drive-speed");
  const step = $("drive-step");
  const lock = $("motion-lock");
  if (speed) {
    speed.oninput = () => {
      driveSpeedIndex = readStoredIndexFromValue(speed.value, 1);
      localStorage.setItem(DRIVE_SPEED_KEY, String(driveSpeedIndex));
      syncDriveControls();
    };
  }
  if (step) {
    step.oninput = () => {
      driveStepIndex = readStoredIndexFromValue(step.value, 1);
      localStorage.setItem(DRIVE_STEP_KEY, String(driveStepIndex));
      syncDriveControls();
    };
  }
  if (lock) {
    lock.onclick = () => {
      motionLocked = !motionLocked;
      localStorage.setItem(MOTION_LOCK_KEY, motionLocked ? "1" : "0");
      syncDriveControls();
    };
  }
  syncDriveControls();
}

function readStoredIndexFromValue(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, Math.min(2, Math.round(parsed))) : fallback;
}

function isDriveMoveCommand(command) {
  return ["motor_forward", "motor_backward", "motor_left", "motor_right"].includes(command);
}

function driveCommandPayload() {
  return {
    speed_index: driveSpeedIndex,
    speed: DRIVE_SPEED_LABELS[driveSpeedIndex]?.toLowerCase() || "normal",
    step_index: driveStepIndex,
    step: DRIVE_STEP_LABELS[driveStepIndex]?.toLowerCase() || "medium",
  };
}

function initDailyCommandButtons() {
  document.querySelectorAll("[data-daily-command], [data-panel-command]").forEach((button) => {
    const command = button.dataset.dailyCommand || button.dataset.panelCommand;
    button.onclick = () => guard("Command failed", () => {
      if (isDriveMoveCommand(command)) {
        if (motionLocked) {
          notice("Motion lock is on; unlock Drive before moving.");
          return null;
        }
        return sendCommand(command, driveCommandPayload());
      }
      return sendCommand(command);
    });
    button.title = command;
  });
  document.querySelectorAll("[data-daily-toggle]").forEach((button) => {
    button.onclick = () => guard("Command failed", () => {
      const command = button.dataset.currentCommand || button.dataset.onCommand;
      return sendCommand(command);
    });
  });
  const mute = $("speaker-mute");
  if (mute) {
    mute.onclick = () => guard("Speaker mute toggle failed", async () => {
      const current = normalizeSpeakerVolume($("speaker-volume-slider")?.value) ?? speakerVolumeFromStatus(latestStatus.esp || {}) ?? rememberedSpeakerVolume ?? 50;
      if (current > 0) {
        rememberSpeakerVolumeBeforeMute(current);
        await setSpeakerVolume(0);
        return;
      }
      const previous = normalizeSpeakerVolume(localStorage.getItem(SPEAKER_VOLUME_BEFORE_MUTE_KEY)) || 50;
      await setSpeakerVolume(previous);
    });
  }
}

async function boot() {
  document.documentElement.dataset.aliceUiVersion = UI_VERSION;
  initAutoScrollContainers();
  initHelpBubbles();
  initRadarControls();
  initCommandTabs();
  initPipelineTabs();
  initCollapsiblePanels();
  initDriveControls();
  renderButtons();
  initDailyCommandButtons();
  initProviderSwitches();
  $("refresh-btn").onclick = () => guard("Refresh failed", loadStatus);
  $("unlock-btn").onclick = () => guard("Unlock failed", unlock);
  $("pipeline-send").onclick = () => guard("Pipeline failed", runPipeline);
  $("pipeline-tts-send").onclick = () => guard("TTS test failed", runTtsTest);
  $("pipeline-tts-benchmark").onclick = () => guard("TTS latency test failed", runTtsLatencyBenchmark);
  $("pipeline-messages-download").onclick = () => guard("Pipeline message download failed", downloadPipelineMessages);
  $("pipeline-messages-clear").onclick = () => guard("Clear pipeline messages failed", clearPipelineMessages);
  $("session-start").onclick = () => guard("Session start failed", startVoiceSession);
  $("session-stop").onclick = () => guard("Session stop failed", stopVoiceSession);
  $("response-cancel").onclick = () => guard("Response cancel failed", cancelResponse);
  initSpeakerVolumeControl();
  $("mic-record-left").onclick = () => guard("Left mic capture failed", () => recordMicDebug("left"));
  $("mic-record-right").onclick = () => guard("Right mic capture failed", () => recordMicDebug("right"));
  $("mic-play-left").onclick = () => guard("Left mic playback failed", () => playMicDebug("left"));
  $("mic-play-right").onclick = () => guard("Right mic playback failed", () => playMicDebug("right"));
  $("mic-download-left").onclick = () => guard("Left mic download failed", () => downloadMicDebug("left"));
  $("mic-download-right").onclick = () => guard("Right mic download failed", () => downloadMicDebug("right"));
  $("config-save").onclick = () => guard("Config save failed", saveConfig);
  $("config-export").onclick = () => guard("Config export failed", exportConfig);
  $("config-import").onclick = () => $("config-import-file").click();
  $("config-import-file").onchange = () => guard("Config import failed", importConfig);
  $("prompt-new").onclick = () => guard("Prompt create failed", createPrompt);
  $("prompt-copy").onclick = () => guard("Prompt copy failed", copyPrompt);
  $("prompt-delete").onclick = () => guard("Prompt delete failed", deletePrompt);
  $("prompt-save").onclick = () => guard("Prompt save failed", savePrompt);
  $("prompt-activate").onclick = () => guard("Prompt activate failed", activatePrompt);
  $("logs-download").onclick = () => guard("Log download failed", downloadLogs);
  $("logs-clear").onclick = () => guard("Clear logs failed", () => sendCommand("clear_logs"));
  $("logs-pause").onclick = () => {
    paused = !paused;
    $("logs-pause").textContent = paused ? "Resume" : "Pause";
    if (!paused) {
      loadLogSnapshot().catch(() => undefined);
      if (!logSocket || logSocket.readyState === WebSocket.CLOSED) connectLogs();
    }
  };
  $("log-search").oninput = () => renderLogs({ forceScroll: true });
  $("log-level").onchange = () => renderLogs({ forceScroll: true });
  $("log-category").onchange = () => renderLogs({ forceScroll: true });
  initLogListInteractions();
  document.querySelectorAll("[data-log-preset]").forEach((button) => {
    button.onclick = () => setLogPreset(button.dataset.logPreset || "all");
  });
  $("log-focus-toggle").onclick = () => {
    logFocusMode = !logFocusMode;
    localStorage.setItem("alice_log_focus", logFocusMode ? "1" : "0");
    renderLogControls();
    renderLogs({ forceScroll: true });
  };
  window.addEventListener("resize", () => {
    if (latestRadarDraw) drawRadarViews(latestRadarDraw);
  });

  try {
    const auth = await api("/api/auth/check", {}, "");
    if (auth.auth_required && !token) {
      $("login").classList.remove("hidden");
      return;
    }
    await loadAll();
    connectLogs();
    connectEvents();
    startStatusPolling();
  } catch (err) {
    notice(err.message);
  }
}

async function unlock() {
  const draft = $("token-input").value;
  try {
    await api("/api/status", {}, draft);
    rememberToken(draft);
    $("login").classList.add("hidden");
    await loadAll();
    connectLogs();
    connectEvents();
    startStatusPolling();
  } catch (err) {
    $("login-error").textContent = err.message;
  }
}

async function loadAll() {
  await loadStatus();
  await loadPrompts();
}

function startStatusPolling() {
  if (statusTimer) window.clearInterval(statusTimer);
  statusTimer = window.setInterval(() => loadStatus().catch(() => undefined), 5000);
}

function scheduleStatusRefresh(delay = 250) {
  if (statusRefreshTimer) window.clearTimeout(statusRefreshTimer);
  statusRefreshTimer = window.setTimeout(() => loadStatus().catch(() => undefined), delay);
}

async function loadStatus() {
  const data = await api("/api/status");
  latestStatus = data || {};
  const esp = data.esp || {};
  const pipe = data.pipeline || {};
  const health = (data.health || {}).system || {};
  const backend = data.health || {};
  if (!configDirty) currentConfig = data.config || {};

  const reconnectPaused = Boolean(esp.auto_reconnect_paused);
  $("summary").textContent = esp.online ? "Robot linked" : reconnectPaused ? "ESP offline, auto reconnect paused" : esp.mock_mode ? "ESP offline, mock mode active" : "Waiting for robot status";
  text("backend-version", `${backend.service || "alice_control_panel"} ${backend.version || ""} / ui ${UI_VERSION} - FastAPI backend online`);
  setPill("state-pill", pipe.state || "IDLE");
  const session = pipe.session || {};
  const liveMic = pipe.live_mic || {};
  const realtime = pipe.realtime || {};
  setPill("session-pill", session.active ? "SESSION ON" : "SESSION OFF", session.active ? "good" : "info");
  setPill("esp-pill", esp.online ? "ONLINE" : reconnectPaused ? "PAUSED" : esp.mock_mode ? "MOCK" : "OFFLINE");
  setPill("stream-pill", pipe.stream_active ? "STREAM ON" : "STREAM OFF", pipe.stream_active ? "good" : "info");
  text(
    "session-meta",
    session.active
      ? `${session.mode || "manual"} - ${fmtSeconds(session.uptime_sec)} - ${session.turns || 0} turns - ${session.last_event || "active"}`
      : realtime.active
        ? `realtime ${realtime.connected ? "connected" : "active"} - ${realtime.model || "model n/a"} - ${realtime.last_event || "active"}`
        : `session idle - ${session.last_event || "ready"} - live ws ${liveMic.clients || 0}`
  );
  text("robot-status", esp.online ? "ONLINE" : esp.mock_mode ? "MOCK" : "OFFLINE");
  text("robot-ip", esp.ip || "no ESP base URL");
  text("wifi-status", esp.wifi?.connected ? "connected" : "unknown");
  text("wifi-rssi", esp.wifi?.rssi ? `${esp.wifi.rssi} dBm` : "RSSI n/a");
  text("cpu-status", health.cpu_percent == null ? "n/a" : `${health.cpu_percent}%`);
  text("ram-status", `RAM ${health.ram_used_mb || "n/a"} MB`);
  text("heap-status", esp.heap_free || "n/a");
  text("heap-min", esp.heap_min ? `min ${esp.heap_min}` : "offline");
  text("server-uptime", fmtSeconds(health.uptime_sec));
  text("esp-uptime", `ESP ${fmtSeconds(esp.uptime_sec)}`);
  renderEspHealth(esp.system || {});
  const liveMode = Boolean(realtime.enabled || realtime.active || realtime.connected);
  const liveProvider = `${realtime.provider || "openai"} realtime`;
  text("conn-stt", liveMode ? `${liveProvider} / ${realtime.transcription_model || "stt n/a"}` : data.stt?.provider || "faster_whisper");
  text("conn-llm", liveMode ? `${liveProvider} / ${realtime.model || "model n/a"}` : `${data.llm?.provider || "openai"} / ${data.llm?.model || "n/a"}`);
  text("conn-tts", `${data.tts?.provider || "openai"} / ${data.tts?.pcm_sample_rate || "n/a"}`);
  text("conn-reconnects", formatReconnects(esp));
  text("conn-esp-ws", esp.ws_connected ? "connected" : "offline");
  text("hw-mic", esp.hardware?.mic || "unknown");
  text("hw-speaker", esp.hardware?.speaker || "unknown");
  updateSpeakerVolumeUi(esp);
  text("hw-radar", esp.hardware?.radar || esp.radar?.state || "unknown");
  text("hw-motion", formatMotionSensor(esp.hardware || {}));
  text("hw-touch", formatTouchSensor(esp.hardware || {}));
  const behavior = esp.hardware?.behavior || "normal";
  const behaviorSource = esp.hardware?.behavior_source || "";
  const behaviorEmotion = esp.hardware?.behavior_emotion || "";
  const behaviorLabel = behaviorSource ? `${behavior}/${behaviorSource}` : behavior;
  const idleEye = esp.idle_tracking || {};
  const behaviorParts = [behaviorLabel];
  if (behaviorEmotion) behaviorParts.push(behaviorEmotion);
  if (esp.hardware?.idle_eye_tracking_active || idleEye.active) {
    behaviorParts.push(`eye:${idleEye.direction || esp.hardware?.idle_eye_tracking || "tracking"}`);
  }
  text("hw-behavior", behaviorParts.join(" / "));
  text("hw-servo", esp.hardware?.servo_position || "center");
  text("hw-amp", esp.hardware?.amp_muted == null ? "unknown" : esp.hardware.amp_muted ? "muted" : "active");
  text("hw-wake", esp.hardware?.wake_enabled == null ? "unknown" : esp.hardware.wake_enabled ? "on" : "off");
  text("hw-state", formatPowerMode(esp));
  syncDailyBehaviorButtons(esp, pipe);
  renderPipelineTrace(pipe, realtime, data);
  renderPipelineMessages(pipe.messages || []);
  renderRealtimeLatency(realtime.latency || {});
  renderRadar(esp.radar || latestRadar || {});
  renderMicDebug(pipe.mic_debug || {});
  renderTurnTiming(realtime.latency || {}, pipe.timeline || []);
  if (!configDirty) fillConfig();
}

function renderEspHealth(system) {
  const temp = finiteNumber(system.temperature_c);
  const cpu = finiteNumber(system.cpu_percent);
  const cpuMhz = finiteNumber(system.cpu_mhz);
  const cores = Array.isArray(system.cpu_cores) ? system.cpu_cores.map(finiteNumber).filter((v) => v !== null) : [];
  const ram = system.ram && typeof system.ram === "object" ? system.ram : {};
  const psram = system.psram && typeof system.psram === "object" ? system.psram : {};
  const resetReason = String(system.reset_reason || "").trim();
  const resetRisk = String(system.reset_risk || "info").toLowerCase();
  const monitorReady = Boolean(system.monitor_ready);

  text("esp-temp", temp === null ? "n/a" : `${temp.toFixed(1)} C`);
  text("esp-cpu", cpu === null ? "n/a" : `${Math.round(cpu)}%${cores.length ? ` (${cores.join("/")})` : ""}`);
  text("esp-cpu-freq", cpuMhz === null ? "n/a" : `${Math.round(cpuMhz)} MHz`);
  text("esp-ram", formatMemoryBrief(ram));
  text("esp-psram", formatMemoryBrief(psram));
  text("esp-reset", resetReason || "n/a");

  let label = monitorReady ? "OK" : "N/A";
  let pillTone = monitorReady ? "good" : "info";
  if (temp !== null && temp >= 78) {
    label = "HOT";
    pillTone = "bad";
  } else if (temp !== null && temp >= 70) {
    label = "WARM";
    pillTone = "warn";
  } else if (resetRisk === "warn") {
    label = "CHECK";
    pillTone = "warn";
  }
  setPill("esp-health-pill", label, pillTone);
}

function formatMemoryBrief(memory) {
  const free = finiteNumber(memory.free);
  const total = finiteNumber(memory.total);
  if (free === null && total === null) return "n/a";
  if (free !== null && total !== null && total > 0) return `${formatBytes(free)}/${formatBytes(total)}`;
  if (free !== null) return `free ${formatBytes(free)}`;
  return formatBytes(total);
}

function formatMotionSensor(hw) {
  if (hw.motion_sensor_present === false || hw.motion_sensor === "missing") return "missing";
  if (hw.motion_sensor_ready === false || hw.motion_sensor === "not_ready") return "not ready";
  if (hw.motion_sensor_ready === true || hw.motion_sensor === "ready") {
    return hw.lift_reactions_enabled === false ? "ready / off" : "active";
  }
  return hw.motion_sensor || "unknown";
}

function formatTouchSensor(hw) {
  if (hw.touch_sensor_active === true || hw.touch_sensor === "touching") return "touching";
  if (hw.touch_sensor_ready === false || hw.touch_sensor === "not_ready") return "not ready";
  if (hw.touch_sensor_ready === true || hw.touch_sensor === "ready") {
    return hw.touch_reactions_enabled === false ? "ready / off" : "active";
  }
  return hw.touch_sensor || "unknown";
}

function formatPowerMode(esp) {
  const mode = String(esp?.power_mode || esp?.sleep_level || "").toLowerCase();
  if (mode === "soft_sleep") return "SOFT SLEEP";
  if (mode === "night_sleep") return "NIGHT SLEEP";
  if (esp?.sleep_mode) return "SLEEP";
  return esp?.state || "OFFLINE";
}

function radarStateTone(state, fresh, ready) {
  if (state === "sleep") return "info";
  if (!ready) return "bad";
  if (!fresh && state !== "clear") return "warn";
  if (state === "tracking") return "good";
  if (state === "clear") return "info";
  return "warn";
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeSpeakerVolume(raw) {
  if (raw == null || raw === "") return null;
  const volume = Number(raw);
  return Number.isFinite(volume) ? clamp(Math.round(volume), 0, 100) : null;
}

function rememberSpeakerVolume(volume) {
  const normalized = normalizeSpeakerVolume(volume);
  if (normalized == null) return null;
  rememberedSpeakerVolume = normalized;
  localStorage.setItem(SPEAKER_VOLUME_STORAGE_KEY, String(normalized));
  return normalized;
}

function rememberSpeakerVolumeBeforeMute(volume) {
  const normalized = normalizeSpeakerVolume(volume);
  if (normalized == null || normalized <= 0) return;
  localStorage.setItem(SPEAKER_VOLUME_BEFORE_MUTE_KEY, String(normalized));
}

async function setSpeakerVolume(volume) {
  const normalized = normalizeSpeakerVolume(volume);
  if (normalized == null) return;
  const slider = $("speaker-volume-slider");
  if (slider) slider.value = String(normalized);
  rememberSpeakerVolume(normalized);
  const result = await sendCommand("speaker_volume_set", { volume: normalized });
  rememberSpeakerVolume(
    result?.response?.volume_percent ??
    result?.response?.response?.volume_percent ??
    result?.volume_percent ??
    normalized
  );
  updateSpeakerVolumeUi(latestStatus.esp || {});
}

function speakerVolumeFromStatus(esp) {
  return normalizeSpeakerVolume(
    esp.hardware?.speaker_volume_percent ??
    esp.hardware?.speaker_volume ??
    esp.speaker_volume_percent ??
    esp.speaker_volume ??
    null
  );
}

function speakerGainFromStatus(esp) {
  const gain = Number(esp.hardware?.speaker_gain_q12 ?? esp.speaker_gain_q12 ?? null);
  return Number.isFinite(gain) ? gain : null;
}

function updateSpeakerVolumeText(volume, gainQ12 = null, source = "device") {
  text("speaker-volume-value", `${volume}%`);
  const gain = Number(gainQ12);
  const gainLabel = Number.isFinite(gain) ? `gain Q12 ${gain} (~${(gain / 4096).toFixed(3)}x)` : "gain waiting";
  const sourceLabel = source === "remembered" ? "last panel value" : "50% = current quiet baseline";
  text("speaker-volume-meta", `${gainLabel}; ${sourceLabel}`);
  const mute = $("speaker-mute");
  if (mute) {
    const muted = Number(volume) <= 0;
    mute.classList.toggle("active", muted);
    mute.setAttribute("aria-pressed", muted ? "true" : "false");
    mute.textContent = muted ? "Muted" : "Mute";
    mute.title = muted ? "Restore previous speaker volume" : "Mute speaker";
  }
}

function updateSpeakerVolumeUi(esp) {
  const slider = $("speaker-volume-slider");
  if (!slider) return;
  const statusVolume = speakerVolumeFromStatus(esp);
  const gainQ12 = speakerGainFromStatus(esp);
  const source = statusVolume == null ? "remembered" : "device";
  const displayVolume = statusVolume ?? rememberedSpeakerVolume ?? normalizeSpeakerVolume(slider.value) ?? 50;
  if (statusVolume != null) {
    rememberSpeakerVolume(statusVolume);
  }
  if (!speakerVolumeEditing) slider.value = String(displayVolume);
  updateSpeakerVolumeText(Number(slider.value || displayVolume), gainQ12, source);
}

function setDailyCommandActive(command, active) {
  document.querySelectorAll(`[data-daily-command="${command}"]`).forEach((button) => {
    button.classList.toggle("active", Boolean(active));
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function setDailyToggleState(key, enabled) {
  const labels = DAILY_TOGGLE_LABELS[key] || { on: key, off: key };
  document.querySelectorAll(`[data-daily-toggle="${key}"]`).forEach((button) => {
    const known = enabled != null;
    const active = Boolean(enabled);
    button.classList.toggle("active", known && active);
    button.setAttribute("aria-pressed", known && active ? "true" : "false");
    button.dataset.currentCommand = active ? button.dataset.offCommand : button.dataset.onCommand;
    button.textContent = active ? labels.on : labels.off;
    button.title = active ? button.dataset.offCommand : button.dataset.onCommand;
  });
}

function syncDailyBehaviorButtons(esp, pipe = {}) {
  const hw = esp?.hardware || {};
  const listening = Boolean(pipe?.session?.active || pipe?.live_mic?.clients || String(esp?.hardware?.mic || "").includes("streaming"));
  setDailyToggleState("listen", listening);
  setDailyToggleState("wake", hw.wake_enabled);
  setDailyToggleState("follow_up", hw.follow_up_enabled);
  setDailyToggleState("touch_reactions", hw.touch_reactions_enabled);
  setDailyToggleState("lift_reactions", hw.lift_reactions_enabled);
  setDailyToggleState("motors", hw.motors_enabled);
  setDailyToggleState("sleep_mode", esp?.sleep_mode ?? hw.sleep_mode ?? hw.eyes_sleeping);
}

function radarTargetNumber(target, key) {
  const value = Number(target?.[key]);
  return Number.isFinite(value) ? value : 0;
}

function radarTargetY(target) {
  const hasY = target && target.y_mm !== undefined && target.y_mm !== null && Number.isFinite(Number(target.y_mm));
  if (hasY) return radarTargetNumber(target, "y_mm");
  const distanceMm = radarTargetNumber(target, "distance_mm");
  return Math.max(0, distanceMm);
}

function radarTargetDistance(target) {
  const xMm = radarTargetNumber(target, "x_mm");
  const yMm = radarTargetY(target);
  return Math.round(Math.sqrt((xMm * xMm) + (yMm * yMm)));
}

function radarAngleDeg(target) {
  const xMm = radarTargetNumber(target, "x_mm");
  const yMm = radarTargetY(target);
  if (!xMm && !yMm) return null;
  return Math.round(Math.atan2(xMm, yMm || 1) * 180 / Math.PI);
}

function radarScaleMax(values, fallback, min, max, step) {
  const observed = Math.max(0, ...values.filter((value) => Number.isFinite(value)));
  const desired = observed > 0 ? observed * 1.35 : fallback;
  return clamp(Math.ceil(desired / step) * step, min, max);
}

function radarDistanceLabel(mm) {
  return mm >= 1000 ? `${(mm / 1000).toFixed(mm >= 3000 ? 0 : 1)}m` : `${Math.round(mm / 10)}cm`;
}

function radarDirectionFromX(xMm, previous = "BELIRSIZ") {
  if (previous === "SOL") {
    if (xMm > RADAR_DIRECTION_ENTER_MM) return "SAG";
    if (xMm > -RADAR_DIRECTION_EXIT_MM) return "ORTA";
    return "SOL";
  }
  if (previous === "SAG") {
    if (xMm < -RADAR_DIRECTION_ENTER_MM) return "SOL";
    if (xMm < RADAR_DIRECTION_EXIT_MM) return "ORTA";
    return "SAG";
  }
  if (xMm < -RADAR_DIRECTION_ENTER_MM) return "SOL";
  if (xMm > RADAR_DIRECTION_ENTER_MM) return "SAG";
  return "ORTA";
}

function updateRadarUiTrack(selected, fresh) {
  if (!fresh || !selected) {
    radarUiTrack = { valid: false, direction: "BELIRSIZ" };
    return null;
  }

  const now = Date.now();
  const rawX = radarTargetNumber(selected, "x_mm");
  const rawY = radarTargetY(selected);
  const lastX = Number(radarUiTrack.x_mm || 0);
  const lastY = Number(radarUiTrack.y_mm || 0);
  const jumpMm = radarUiTrack.valid ? Math.sqrt(((rawX - lastX) ** 2) + ((rawY - lastY) ** 2)) : 0;
  const staleMs = radarUiTrack.updated_at ? now - radarUiTrack.updated_at : 0;
  const reset = !radarUiTrack.valid ||
    radarUiTrack.target_id !== selected.id ||
    staleMs > RADAR_UI_RESET_MS ||
    jumpMm > RADAR_UI_RESET_JUMP_MM;
  const alpha = reset ? 1 : RADAR_UI_FILTER_ALPHA;
  const xMm = Math.round(reset ? rawX : (lastX * (1 - alpha)) + (rawX * alpha));
  const yMm = Math.round(reset ? rawY : (lastY * (1 - alpha)) + (rawY * alpha));
  const direction = radarDirectionFromX(xMm, reset ? "BELIRSIZ" : radarUiTrack.direction);

  radarUiTrack = {
    valid: true,
    target_id: selected.id,
    x_mm: xMm,
    y_mm: yMm,
    distance_mm: radarTargetDistance({ x_mm: xMm, y_mm: yMm }),
    angle_deg: radarAngleDeg({ x_mm: xMm, y_mm: yMm }),
    direction,
    raw_x_mm: rawX,
    raw_y_mm: rawY,
    raw_angle_deg: radarAngleDeg(selected),
    updated_at: now,
  };
  return radarUiTrack;
}

function renderRadarTargets(targets) {
  const box = $("radar-targets");
  if (!box) return;
  box.innerHTML = "";
  if (!targets.length) {
    box.classList.add("empty");
    box.textContent = "No targets";
    return;
  }
  box.classList.remove("empty");
  targets.forEach((target) => {
    const row = document.createElement("div");
    const distanceMm = radarTargetDistance(target);
    const angle = radarAngleDeg(target);
    const rawResolution = target.resolution_mm ?? target.distance_mm ?? 0;
    row.className = `radar-target-row${target.selected ? " selected" : ""}`;
    row.innerHTML = `
      <b class="radar-target-title">#${target.id}${target.selected ? " selected" : ""}</b>
      <span class="radar-target-stat"><i>d</i><strong>${radarDistanceLabel(distanceMm)}</strong></span>
      <span class="radar-target-stat"><i>x</i><strong>${radarTargetNumber(target, "x_mm")}mm</strong></span>
      <span class="radar-target-stat"><i>y</i><strong>${radarTargetNumber(target, "y_mm")}mm</strong></span>
      <span class="radar-target-stat"><i>a</i><strong>${angle === null ? "-" : `${angle > 0 ? "+" : ""}${angle}deg`}</strong></span>
      <span class="radar-target-stat"><i>v</i><strong>${radarTargetNumber(target, "speed_cms")}cm/s</strong></span>
      <span class="radar-target-stat"><i>res</i><strong>${rawResolution}mm</strong></span>
    `;
    box.appendChild(row);
  });
}

function renderRadar(info) {
  latestRadar = info || {};
  const ready = Boolean(latestRadar.ready);
  const fresh = Boolean(latestRadar.fresh);
  const state = String(latestRadar.state || (ready ? "waiting" : "offline"));
  const rawTargets = Array.isArray(latestRadar.targets) ? latestRadar.targets : [];
  const rawSelected = rawTargets.find((target) => target.selected) || rawTargets.find((target) => Number(target.id) === Number(latestRadar.selected_target));
  const normalizedTargets = rawTargets.map((target) => radarNormalizeTarget(latestRadar, target));
  const selected = rawSelected
    ? radarNormalizeTarget(latestRadar, rawSelected)
    : normalizedTargets.find((target) => Number(target.id) === Number(latestRadar.selected_target));
  const firmwareFilteredRaw = latestRadar.filtered?.valid ? {
    valid: true,
    target_id: latestRadar.filtered.target_id,
    x_mm: radarTargetNumber(latestRadar.filtered, "x_mm"),
    y_mm: radarTargetNumber(latestRadar.filtered, "y_mm"),
    distance_mm: radarTargetDistance(latestRadar.filtered),
    angle_deg: radarAngleDeg(latestRadar.filtered),
    direction: latestRadar.filtered.direction || radarDirectionFromX(radarTargetNumber(latestRadar.filtered, "x_mm")),
  } : null;
  const rawFiltered = firmwareFilteredRaw || updateRadarUiTrack(rawSelected, fresh);
  const filtered = rawFiltered?.valid ? {
    ...radarNormalizeTarget(latestRadar, rawFiltered),
    valid: true,
    target_id: rawFiltered.target_id,
  } : null;
  if (filtered?.valid) filtered.direction = radarDirectionFromX(filtered.x_mm, rawFiltered?.direction || "BELIRSIZ");
  const targets = normalizedTargets;
  const selectedAngle = filtered?.valid ? filtered.angle_deg : selected ? radarAngleDeg(selected) : null;
  const selectedDistance = filtered?.valid ? filtered.distance_mm : selected ? radarTargetDistance(selected) : 0;
  const selectedResolution = rawSelected ? (rawSelected.resolution_mm ?? rawSelected.distance_mm ?? 0) : 0;
  const direction = filtered?.valid
    ? filtered.direction
    : selected ? radarDirectionFromX(selected.x_mm, latestRadar.direction || "BELIRSIZ") : latestRadar.direction || "BELIRSIZ";
  const radarMeta = [];
  const confidence = Number(latestRadar.confidence ?? filtered?.confidence ?? 0);
  const stableFrames = Number(latestRadar.stable_frames ?? filtered?.stable_frames ?? 0);
  const frameCount = Number(latestRadar.frame_count || 0);
  const errorCount = Number(latestRadar.error_count || 0);
  const hasUartDiag = Object.prototype.hasOwnProperty.call(latestRadar, "uart_bytes");
  const uartBytes = hasUartDiag ? Number(latestRadar.uart_bytes || 0) : 0;
  if (ready && (fresh || confidence > 0)) radarMeta.push(`guven ${confidence}%`);
  if (!fresh || frameCount === 0 || errorCount > 0) {
    radarMeta.push(hasUartDiag ? `uart ${uartBytes}B` : "uart diag yok");
    radarMeta.push(`${frameCount} frame`);
    if (errorCount > 0) radarMeta.push(`${errorCount} hata`);
  }
  if (stableFrames > 0) radarMeta.push(`${stableFrames} frame stabil`);
  if (latestRadar.last_jump_rejected) radarMeta.push("sicrama reddedildi");
  if (Number(latestRadar.jump_rejects || 0) > 0) radarMeta.push(`${latestRadar.jump_rejects} sicrama red`);
  if (latestRadar.background_learning) radarMeta.push(`bos oda ogreniyor ${latestRadar.background_samples || 0}`);
  else if (latestRadar.background_active) radarMeta.push(`arka plan ${latestRadar.background_points || 0} nokta`);
  if (Number(latestRadar.background_suppressed || 0) > 0) radarMeta.push(`${latestRadar.background_suppressed} arka plan bastirildi`);
  const metaText = radarMeta.length ? ` | ${radarMeta.join(" | ")}` : "";
  setPill("radar-pill", state.toUpperCase(), radarStateTone(state, fresh, ready));
  text("radar-count", String(latestRadar.target_count ?? rawTargets.length ?? 0));
  text("radar-direction", direction);
  text(
    "radar-selected",
    selected
      ? `#${selected.id} ${radarDistanceLabel(selectedDistance)}`
      : "-"
  );
  text("radar-angle", selectedAngle === null ? "-" : `${selectedAngle > 0 ? "+" : ""}${selectedAngle} deg`);
  const age = Number(latestRadar.fresh_ms);
  const detail = !ready
    ? "RD-03D UART hazir degil."
      : !fresh
        ? age >= 0 ? `Son radar frame ${age}ms once; veri eski sayiliyor.${metaText}` : `RD-03D verisi bekleniyor.${metaText}`
      : selected
        ? `Karar d=${radarDistanceLabel(selectedDistance)} x=${filtered?.x_mm ?? selected.x_mm}mm y=${filtered?.y_mm ?? selected.y_mm}mm aci=${selectedAngle ?? "-"}deg | ham x=${rawSelected?.x_mm ?? "-"} y=${rawSelected?.y_mm ?? "-"} res=${selectedResolution}mm${metaText}`
        : `Radar taze, hedef yok.${metaText}`;
  text("radar-detail", detail);
  renderRadarTargets(targets);
  latestRadarDraw = { ...latestRadar, targets, ui_filtered: filtered };
  drawRadarViews(latestRadarDraw);
}

function drawRadarViews(info) {
  drawRadarMap(info);
  drawRadarRoom(info);
}

function drawRadarMap(info) {
  const canvas = $("radar-canvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(240, Math.round(rect.width || canvas.width));
  const height = Math.max(160, Math.round(rect.height || canvas.height));
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const cx = width / 2;
  const bottom = height - 24;
  const top = 18;
  const targets = Array.isArray(info?.targets) ? info.targets : [];
  const techPoints = targets.map((target) => radarApplyCalibrationXY(radarTargetNumber(target, "x_mm"), radarTargetY(target)));
  const filteredForScale = info?.ui_filtered?.valid
    ? radarApplyCalibrationXY(info.ui_filtered.x_mm, radarTargetY(info.ui_filtered))
    : null;
  if (filteredForScale) techPoints.push(filteredForScale);
  const maxY = radarScaleMax(techPoints.map((point) => Math.max(0, point.y_mm)), RADAR_DEFAULT_MAX_Y_MM, 1200, 6000, 400);
  const maxAbsX = radarScaleMax(techPoints.map((point) => Math.abs(point.x_mm)), 1600, 800, 3000, 200);
  const maxX = Math.max(maxAbsX, Math.round(maxY * 0.35), RADAR_DIRECTION_DEADZONE_MM * 2);
  const mapW = width - 34;
  const mapH = bottom - top;
  const mapLeft = cx - mapW / 2;
  const mapRight = cx + mapW / 2;

  ctx.fillStyle = "#0b111a";
  ctx.fillRect(0, 0, width, height);
  const deadzoneHalf = clamp(RADAR_DIRECTION_DEADZONE_MM / maxX, 0, 1) * (mapW / 2);
  ctx.fillStyle = "rgba(48,209,88,.06)";
  ctx.fillRect(cx - deadzoneHalf, top, deadzoneHalf * 2, mapH);
  ctx.strokeStyle = "rgba(100,169,255,.16)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    const y = bottom - (mapH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(14, y);
    ctx.lineTo(width - 14, y);
    ctx.stroke();
  }
  for (let i = -2; i <= 2; i += 1) {
    const x = cx + (mapW * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(48,209,88,.28)";
  ctx.beginPath();
  ctx.moveTo(cx - deadzoneHalf, top);
  ctx.lineTo(cx - deadzoneHalf, bottom);
  ctx.moveTo(cx + deadzoneHalf, top);
  ctx.lineTo(cx + deadzoneHalf, bottom);
  ctx.stroke();

  ctx.strokeStyle = "rgba(57,197,187,.28)";
  ctx.beginPath();
  ctx.moveTo(cx, bottom);
  ctx.lineTo(18, top + 18);
  ctx.moveTo(cx, bottom);
  ctx.lineTo(width - 18, top + 18);
  ctx.stroke();

  ctx.fillStyle = "#cbb8ff";
  ctx.beginPath();
  ctx.moveTo(cx, bottom - 12);
  ctx.lineTo(cx - 12, bottom + 8);
  ctx.lineTo(cx + 12, bottom + 8);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(229,221,255,.62)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "rgba(229,221,255,.82)";
  ctx.font = "11px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText("Alice", cx, bottom + 20);
  ctx.fillStyle = "rgba(215,236,255,.58)";
  ctx.textAlign = "left";
  ctx.fillText(radarDistanceLabel(maxY), mapLeft + 4, top + 12);
  ctx.fillText(`-${radarDistanceLabel(maxX)}`, mapLeft + 4, bottom - 5);
  ctx.textAlign = "right";
  ctx.fillText(`+${radarDistanceLabel(maxX)}`, mapRight - 4, bottom - 5);
  ctx.textAlign = "center";
  ctx.fillText("orta", cx, top + 12);

  const projectRadarPoint = (xMm, yMm) => {
    const point = radarApplyCalibrationXY(xMm, yMm);
    return {
      x: cx + clamp(point.x_mm / maxX, -1, 1) * (mapW / 2),
      y: bottom - clamp(point.y_mm / maxY, 0, 1) * mapH,
    };
  };
  const filtered = info?.ui_filtered?.valid ? info.ui_filtered : null;

  targets.forEach((target) => {
    const xMm = radarTargetNumber(target, "x_mm");
    const yMm = radarTargetY(target);
    const { x, y } = projectRadarPoint(xMm, yMm);
    const selected = Boolean(target.selected);
    const filteredSelected = selected && filtered && filtered.target_id === target.id;
    const angle = radarAngleDeg(target);
    const dotRadius = filteredSelected ? 4.5 : selected ? 6.5 : 5;
    const ringRadius = filteredSelected ? 8.5 : selected ? 12.5 : 9;
    const ringColor = filteredSelected ? "rgba(255,189,84,.2)" : selected ? "rgba(48,209,88,.2)" : "rgba(100,169,255,.16)";
    const glowColor = filteredSelected ? "rgba(255,189,84,.34)" : selected ? "rgba(48,209,88,.34)" : "rgba(100,169,255,.24)";
    ctx.fillStyle = filteredSelected ? "#ffbd54" : selected ? "#30d158" : "#64a9ff";
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = selected ? 2.5 : 2;
    ctx.beginPath();
    ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.save();
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = selected ? 8 : 5;
    ctx.beginPath();
    ctx.arc(x, y, ringRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#c4d0dc";
    ctx.textAlign = "left";
    ctx.fillText(`#${target.id}${filteredSelected ? " ham" : ""}`, x + 10, y - 8);
    if (angle !== null) ctx.fillText(`${angle > 0 ? "+" : ""}${angle}deg`, x + 10, y + 7);
  });

  if (filtered) {
    const { x, y } = projectRadarPoint(filtered.x_mm, filtered.y_mm);
    ctx.fillStyle = "#30d158";
    ctx.strokeStyle = "rgba(48,209,88,.22)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(x, y, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.save();
    ctx.shadowColor = "rgba(48,209,88,.38)";
    ctx.shadowBlur = 9;
    ctx.beginPath();
    ctx.arc(x, y, 13, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#d7ecff";
    ctx.textAlign = "left";
    ctx.fillText(`#${filtered.target_id} karar`, x + 10, y - 8);
    const filteredAngle = filtered.angle_deg ?? radarAngleDeg(filtered);
    if (filteredAngle !== null) ctx.fillText(`${filteredAngle > 0 ? "+" : ""}${filteredAngle}deg`, x + 10, y + 7);
  }

  if (!info?.fresh) {
    ctx.fillStyle = "rgba(11,17,26,.58)";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#ffbd54";
    ctx.textAlign = "center";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText(info?.ready ? "radar waiting" : "radar offline", cx, height / 2);
  }
}

function drawRadarRoom(info) {
  const canvas = $("radar-room-canvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(240, Math.round(rect.width || canvas.width));
  const height = Math.max(180, Math.round(rect.height || canvas.height));
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = 18;
  const cx = width / 2;
  const top = 18;
  const robotY = top + 30;
  const roomLeft = pad;
  const roomRight = width - pad;
  const roomBottom = height - 14;
  const roomW = roomRight - roomLeft;
  const roomH = roomBottom - top;
  const targets = Array.isArray(info?.targets) ? info.targets : [];
  const filtered = info?.ui_filtered?.valid ? info.ui_filtered : null;
  const roomPoints = targets.map((target) => radarApplyRoomXY(radarTargetNumber(target, "x_mm"), radarTargetY(target)));
  if (filtered) roomPoints.push(radarApplyRoomXY(filtered.x_mm, radarTargetY(filtered)));
  const yValues = roomPoints.map((point) => Math.abs(point.y_mm));
  const xValues = roomPoints.map((point) => Math.abs(point.x_mm));
  const maxY = radarScaleMax(yValues, 2600, 1200, 6000, 400);
  const maxX = Math.max(radarScaleMax(xValues, 1600, 900, 3600, 300), Math.round(maxY * 0.42));
  const projectRoomPoint = (xMm, yMm) => {
    const point = radarApplyRoomXY(xMm, yMm);
    return {
      x: cx + clamp(point.x_mm / maxX, -1, 1) * (roomW / 2),
      y: robotY + clamp(point.y_mm / maxY, -0.12, 1) * (roomBottom - robotY - 8),
    };
  };

  ctx.fillStyle = "#0d141d";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(100,169,255,.035)";
  ctx.fillRect(roomLeft, top, roomW, roomH);
  ctx.strokeStyle = "rgba(100,169,255,.22)";
  ctx.lineWidth = 1;
  ctx.strokeRect(roomLeft, top, roomW, roomH);

  ctx.strokeStyle = "rgba(100,169,255,.11)";
  for (let i = 1; i < 5; i += 1) {
    const x = roomLeft + (roomW * i) / 5;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, roomBottom);
    ctx.stroke();
  }
  for (let i = 1; i < 5; i += 1) {
    const y = top + (roomH * i) / 5;
    ctx.beginPath();
    ctx.moveTo(roomLeft, y);
    ctx.lineTo(roomRight, y);
    ctx.stroke();
  }

  const coneLeft = projectRoomPoint(-maxY * 0.58, maxY);
  const coneRight = projectRoomPoint(maxY * 0.58, maxY);
  const coneGrad = ctx.createLinearGradient(cx, robotY, cx, roomBottom);
  coneGrad.addColorStop(0, "rgba(200,184,255,.045)");
  coneGrad.addColorStop(1, "rgba(57,197,187,.01)");
  ctx.fillStyle = coneGrad;
  ctx.beginPath();
  ctx.moveTo(cx, robotY);
  ctx.lineTo(coneLeft.x, coneLeft.y);
  ctx.lineTo(coneRight.x, coneRight.y);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(200,184,255,.075)";
  ctx.beginPath();
  ctx.moveTo(cx, robotY);
  ctx.lineTo(coneLeft.x, coneLeft.y);
  ctx.moveTo(cx, robotY);
  ctx.lineTo(coneRight.x, coneRight.y);
  ctx.stroke();

  ctx.fillStyle = "rgba(215,236,255,.58)";
  ctx.font = "11px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText("on", cx, roomBottom - 8);
  ctx.textAlign = "left";
  ctx.fillText("sol", roomLeft + 8, robotY + 20);
  ctx.textAlign = "right";
  ctx.fillText("sag", roomRight - 8, robotY + 20);

  const selected = targets.find((target) => target.selected) || targets.find((target) => Number(target.id) === Number(info?.selected_target));
  const focus = filtered || selected;
  if (focus) {
    const { x, y } = projectRoomPoint(radarTargetNumber(focus, "x_mm"), radarTargetY(focus));
    ctx.fillStyle = "#30d158";
    ctx.strokeStyle = "rgba(48,209,88,.22)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(x, y, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.save();
    ctx.shadowColor = "rgba(48,209,88,.38)";
    ctx.shadowBlur = 9;
    ctx.beginPath();
    ctx.arc(x, y, 13, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  ctx.fillStyle = "rgba(16,23,32,.95)";
  ctx.strokeStyle = "rgba(200,184,255,.34)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, robotY, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#cbb8ff";
  ctx.beginPath();
  ctx.moveTo(cx, robotY + 22);
  ctx.lineTo(cx - 11, robotY - 6);
  ctx.lineTo(cx + 11, robotY - 6);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "rgba(229,221,255,.84)";
  ctx.textAlign = "center";
  ctx.fillText("Alice", cx, robotY - 13);

  if (!info?.fresh) {
    ctx.fillStyle = "rgba(11,17,26,.58)";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#ffbd54";
    ctx.textAlign = "center";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText(info?.ready ? "radar waiting" : "radar offline", cx, height / 2);
  }
}

function renderMicDebug(info) {
  micDebug = info || {};
  const captures = micDebug.captures || {};
  const left = captures.left;
  const right = captures.right;
  const latest = [left, right].filter(Boolean).sort((a, b) => Number(b.stored_at || 0) - Number(a.stored_at || 0))[0];
  const espWsConnected = Boolean(latestStatus.esp?.ws_connected);
  text(
    "mic-debug-status",
    latest
      ? `latest ${latest.channel || "mic"} - ${fmtSeconds(Math.round((Date.now() / 1000) - Number(latest.stored_at || 0)))} ago`
      : espWsConnected ? "No debug capture yet" : "ESP WS offline - recording needs WebSocket"
  );
  text(
    "mic-debug-meta",
    latest
      ? `${String(latest.channel || "mic").toUpperCase()} | ${latest.duration_sec || 0}s | ${latest.bytes_buffered || 0} bytes | rms ${latest.rms || 0} | peak ${latest.peak || 0} | shift ${latest.shift_bits ?? "-"} | clip ${latest.clip_pct ?? 0}%`
      : "RMS/peak bilgisi kayıt sonrası görünür."
  );
  ["left", "right"].forEach((channel) => {
    const available = Boolean(captures[channel]?.url);
    const play = $(`mic-play-${channel}`);
    const download = $(`mic-download-${channel}`);
    if (play) play.disabled = !available;
    if (download) download.disabled = !available;
  });
}

function formatReconnects(esp) {
  const count = Number(esp.reconnects || 0);
  const max = Number(esp.max_auto_reconnects || 0);
  const base = max ? `${count} / ${max}` : `${count}`;
  return esp.auto_reconnect_paused ? `${base} paused` : base;
}

function fillConfig() {
  document.querySelectorAll("[data-path]").forEach((el) => {
    const value = getDeep(currentConfig, el.dataset.path);
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value ?? "";
    const updateValue = () => {
      configDirty = true;
      const next = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
      setDeep(currentConfig, el.dataset.path, next);
      if (el.dataset.path === "realtime.enabled" && next && activeProvider("realtime") === "none") {
        setDeep(currentConfig, "realtime.provider", "openai");
      }
      if (el.dataset.providerSelect || el.dataset.path === "realtime.enabled") renderProviderSwitches();
    };
    el.oninput = updateValue;
    el.onchange = el.dataset.providerSelect
      ? () => guard("Provider switch failed", () => selectProvider(el.dataset.providerSelect, el.value))
      : updateValue;
  });
  renderProviderSwitches();
}

function initProviderSwitches() {
  document.querySelectorAll(".provider-switch button").forEach((button) => {
    const group = button.closest(".provider-switch");
    if (!group) return;
    button.onclick = () => guard("Provider switch failed", () => selectProvider(group.dataset.providerKind, button.dataset.provider));
  });
}

function activeProvider(kind) {
  if (kind === "realtime" && !getDeep(currentConfig, "realtime.enabled")) return "none";
  return String(getDeep(currentConfig, `${kind}.provider`) || "").toLowerCase();
}

function renderProviderSwitches() {
  const realtimeEnabled = document.querySelector('[data-path="realtime.enabled"]');
  if (realtimeEnabled) realtimeEnabled.checked = Boolean(getDeep(currentConfig, "realtime.enabled"));
  document.querySelectorAll(".provider-switch").forEach((group) => {
    const kind = group.dataset.providerKind;
    const active = activeProvider(kind);
    group.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", button.dataset.provider === active);
    });
  });
  document.querySelectorAll("[data-provider-select]").forEach((select) => {
    const kind = select.dataset.providerSelect;
    const active = activeProvider(kind);
    if (active) select.value = active;
  });
  document.querySelectorAll(".provider-card").forEach((card) => {
    const active = activeProvider(card.dataset.providerKind);
    card.classList.toggle("active", card.dataset.providerCard === active);
  });
}

async function selectProvider(kind, provider) {
  if (!kind || !provider) return;
  if (kind === "realtime") {
    setDeep(currentConfig, "realtime.enabled", provider !== "none");
  }
  setDeep(currentConfig, `${kind}.provider`, provider);
  configDirty = true;
  renderProviderSwitches();
  await saveConfig();
}

function compactPipelineText(value, fallback) {
  const textValue = stripEmotionTags(value);
  return textValue || fallback;
}

function lastPipelineMessage(messages, roles) {
  const list = Array.isArray(messages) ? messages : [];
  const accepted = Array.isArray(roles) ? roles : [roles];
  for (let index = list.length - 1; index >= 0; index -= 1) {
    const item = list[index] || {};
    if (accepted.includes(item.role) && stripEmotionTags(item.text)) return item;
  }
  return null;
}

function renderPipelineTrace(pipe, realtime, statusData = {}) {
  const box = $("pipeline-feed");
  if (!box) return;
  const session = pipe.session || {};
  const liveMic = pipe.live_mic || {};
  const capture = pipe.last_audio_capture || {};
  const messages = Array.isArray(pipe.messages) ? pipe.messages : [];
  const lastUser = lastPipelineMessage(messages, "user");
  const lastAssistant = lastPipelineMessage(messages, "assistant");
  const lastTts = lastPipelineMessage(messages, "tts");
  const realtimeMode = Boolean(realtime.enabled || realtime.active || realtime.connected || realtime.last_transcript || realtime.last_assistant_text);
  const realtimeProvider = `${realtime.provider || "openai"} realtime`;
  const sttText = compactPipelineText(pipe.stt_result || pipe.last_user_text || realtime.last_transcript || lastUser?.text, "No utterance yet");
  const llmText = compactPipelineText(pipe.llm_response || realtime.last_assistant_text || lastAssistant?.text, "No assistant response yet");
  const ttsText = compactPipelineText(
    pipe.last_tts_text || realtime.last_tts_text || realtime.last_assistant_text || lastTts?.text || lastAssistant?.text,
    pipe.tts_status ? `No TTS text captured; status is ${pipe.tts_status}.` : "No TTS text yet"
  );
  const llmProvider = realtimeMode ? `${realtimeProvider} / ${realtime.model || "model n/a"}` : `${statusData.llm?.provider || "LLM"} / ${statusData.llm?.model || "model n/a"}`;
  const ttsProvider = `${statusData.tts?.provider || "TTS"} / ${statusData.tts?.pcm_sample_rate || "rate n/a"}`;
  const rows = [
    {
      kind: "state",
      label: "State",
      meta: `${pipe.state || "IDLE"} / tts ${pipe.tts_status || "idle"} / live ws ${liveMic.clients || 0}`,
      text: session.active
        ? `${session.mode || "manual"} session, ${session.turns || 0} turns, ${session.last_event || "active"}`
        : realtime.active
          ? `Realtime ${realtime.connected ? "connected" : "active"}; ${realtime.model || "model n/a"}; ${realtime.last_event || "active"}`
          : "Waiting for text, wake word, or mic stream.",
    },
    {
      kind: "stt",
      label: "STT",
      meta: capture.duration_sec
        ? `audio ${capture.duration_sec}s / ${capture.bytes_buffered || 0} bytes / rms ${capture.rms || 0}`
        : realtimeMode
          ? `${realtimeProvider} / ${realtime.transcription_model || "stt n/a"}`
          : "incoming user text or transcript",
      text: sttText,
    },
    {
      kind: "llm",
      label: "LLM",
      meta: llmProvider,
      text: llmText,
    },
    {
      kind: "tts",
      label: "TTS",
      meta: `${ttsProvider} / ${pipe.stream_active ? "stream active" : pipe.tts_status || "idle"}`,
      text: ttsText,
    },
  ];
  keepAutoScrolled(box, () => {
    box.innerHTML = "";
    rows.forEach((item) => {
      const row = document.createElement("div");
      row.className = `pipeline-feed-row ${item.kind || "state"}${item.text.startsWith("No ") ? " idle" : ""}`;
      const label = document.createElement("b");
      const body = document.createElement("div");
      const meta = document.createElement("span");
      const textBlock = document.createElement("p");
      body.className = "pipeline-feed-body";
      meta.className = "pipeline-feed-meta";
      textBlock.className = "pipeline-feed-text";
      label.textContent = item.label;
      meta.textContent = item.meta;
      textBlock.textContent = item.text;
      body.append(meta, textBlock);
      row.append(label, body);
      box.appendChild(row);
    });
  });
}

function renderPipelineMessages(messages) {
  const box = $("pipeline-messages");
  if (!box) return;
  const rows = Array.isArray(messages) ? messages.slice(-120) : [];
  keepAutoScrolled(box, () => {
    box.innerHTML = "";
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "pipeline-message-empty";
      empty.textContent = "No pipeline messages yet.";
      box.appendChild(empty);
      return;
    }
    rows.forEach((item) => {
      const row = document.createElement("div");
      const role = String(item.role || "message").toLowerCase();
      row.className = `pipeline-message-row ${role}`;
      const timeEl = document.createElement("time");
      const roleEl = document.createElement("b");
      const sourceEl = document.createElement("span");
      const textEl = document.createElement("p");
      const ts = Number(item.ts || 0);
      timeEl.textContent = fmtClock(ts);
      roleEl.textContent = role.toUpperCase();
      sourceEl.textContent = String(item.source || "-").replaceAll("_", " ");
      textEl.textContent = stripEmotionTags(item.text);
      row.append(timeEl, roleEl, sourceEl, textEl);
      box.appendChild(row);
    });
  });
}

function renderRealtimeLatency(latency) {
  const box = $("realtime-latency");
  if (!box) return;
  const selected = selectTimingTurn(latency);
  const summary = selected.turn?.summary || latency.summary || {};
  const chips = [
    ["Wake -> mic", summary.wake_to_first_audio_ms],
    ["Speech -> STT", summary.speech_stop_to_transcript_ms ?? summary.speech_to_transcript_ms],
    ["STT -> LLM", summary.transcript_to_first_delta_ms],
    ["LLM -> TTS text", summary.first_delta_to_first_chunk_ms],
    ["TTS req -> headers", summary.tts_request_to_headers_ms],
    ["TTS req -> first byte", summary.tts_request_to_first_byte_ms],
    ["TTS req -> audio", summary.tts_request_to_first_audio_ms],
    ["Audio -> ESP chunk", summary.tts_decode_to_esp_chunk_ms],
    ["ESP chunk -> speaker", summary.esp_chunk_to_speaker_ms],
    ["ESP chunk -> finish", summary.esp_chunk_to_speaker_finished_ms],
    ["TTS text -> speaker", summary.tts_text_to_speaker_ms],
    ["TTS text -> finish", summary.tts_text_to_speaker_finished_ms],
    ["Wake -> speaker", summary.wake_to_speaker_ms],
    ["Wake -> finish", summary.wake_to_speaker_finished_ms],
    ["Wake -> TTS text", summary.wake_to_first_tts_ms],
    ["Turn total", summary.total_ms ?? summary.wake_to_complete_ms ?? summary.wake_to_session_completed_ms],
  ];
  box.innerHTML = "";
  chips.forEach(([label, value]) => {
    const chip = document.createElement("div");
    chip.className = "latency-chip";
    const title = document.createElement("span");
    const metric = document.createElement("b");
    title.textContent = label;
    metric.textContent = fmtMs(value);
    chip.append(title, metric);
    box.appendChild(chip);
  });
}

function turnHasSpeechData(turn) {
  if (!turn) return false;
  const summary = turn.summary || {};
  if (String(turn.transcript || turn.assistant_text || "").trim()) return true;
  return (
    summary.speech_to_transcript_ms != null ||
    summary.speech_stop_to_transcript_ms != null ||
    summary.transcript_to_first_delta_ms != null ||
    summary.first_delta_to_first_chunk_ms != null ||
    summary.wake_to_first_tts_ms != null ||
    summary.wake_to_speaker_ms != null
  );
}

function latencyAsTurn(latency = {}) {
  return {
    session_id: latency.session_id || "",
    summary: latency.summary || {},
    stages: latency.stages || fallbackLatencyStages(latency.events || []),
    events: latency.events || [],
    reason: "",
  };
}

function isCurrentTimingActive(latency = {}) {
  const events = Array.isArray(latency.events) ? latency.events : [];
  if (!events.length) return false;
  const last = String(events[events.length - 1]?.name || "");
  return !["session_completed", "speaker_audio_finished", "speaker_finished", "response_cancelled", "client_cancelled"].includes(last);
}

function selectTimingTurn(latency = {}) {
  const history = Array.isArray(latency.history) ? latency.history : [];
  const current = latencyAsTurn(latency);
  if (isCurrentTimingActive(latency) || turnHasSpeechData(current)) {
    return { turn: current, source: "current" };
  }
  const meaningful = history.slice().reverse().find(turnHasSpeechData);
  if (meaningful) return { turn: meaningful, source: "last_spoken" };
  const latest = history.length ? history[history.length - 1] : current;
  return { turn: latest, source: history.length ? "latest" : "current" };
}

function compactSnippet(value, fallback = "-") {
  const textValue = String(value || "").replace(/\s+/g, " ").trim();
  if (!textValue) return fallback;
  return textValue.length > 86 ? `${textValue.slice(0, 83)}...` : textValue;
}

function formatStageDetail(stage) {
  const detail = String(stage.detail || "").trim();
  return detail || "pending detail";
}

function fallbackLatencyStages(events) {
  const labels = {
    client_connected: "Client linked",
    start_received: "Turn start",
    openai_connected: "OpenAI connected",
    first_audio_chunk: "First mic packet",
    speech_started: "Speech started",
    speech_stopped: "Speech stopped",
    input_committed: "Audio committed",
    transcription_completed: "STT completed",
    stt_result_sent: "Transcript sent",
    response_requested: "LLM requested",
    first_llm_delta: "First LLM text",
    first_tts_chunk: "TTS text queued",
    tts_text_queued: "TTS backend queued",
    tts_worker_started: "TTS worker started",
    tts_relay_ws_connect_start: "TTS relay WS start",
    tts_relay_ws_connected: "TTS relay WS connected",
    google_tts_request_build_start: "Google build start",
    google_tts_request_built: "Google request built",
    google_tts_request_send_start: "Google request start",
    google_tts_request_sent: "Google request sent",
    google_tts_response_headers_received: "Google headers",
    google_tts_first_byte_received: "Google first byte",
    google_tts_response_body_buffered: "Google body buffered",
    google_tts_first_audio_chunk_received: "Google audio found",
    google_tts_first_audio_chunk_decoded: "Google audio decoded",
    audio_resample_start: "Audio convert start",
    audio_resample_done: "Audio convert done",
    google_tts_stream_completed: "Google stream done",
    first_chunk_sent_to_esp: "First ESP chunk",
    esp_first_pcm_reported: "ESP first PCM",
    speaker_started: "Speaker started",
    tts_relay_connected: "TTS relay connected",
    tts_relay_request_sent: "TTS request sent",
    tts_relay_started: "TTS stream started",
    speaker_finished: "Speaker finished",
    google_tts_error: "Google TTS error",
    response_done: "LLM done",
    session_completed: "Turn completed",
  };
  return (events || [])
    .filter((event) => labels[event.name])
    .map((event) => ({
      key: event.name,
      label: labels[event.name],
      ms: event.ms,
      detail: formatEventFallbackDetail(event),
    }));
}

function formatEventFallbackDetail(event) {
  const parts = [];
  if (event.reason) parts.push(String(event.reason).replaceAll("_", " "));
  if (event.trace_id) parts.push(String(event.trace_id));
  if (event.provider) parts.push(String(event.provider));
  if (event.transport) parts.push(String(event.transport));
  if (event.chars != null) parts.push(`${event.chars} chars`);
  if (event.text_chars != null) parts.push(`text ${event.text_chars} chars`);
  if (event.text_bytes != null) parts.push(`${event.text_bytes} text bytes`);
  if (event.provider_ms != null) parts.push(`provider +${event.provider_ms}ms`);
  if (event.payload_build_ms != null) parts.push(`payload ${event.payload_build_ms}ms`);
  if (event.request_payload_bytes != null) parts.push(`payload ${event.request_payload_bytes} bytes`);
  if (event.http_status != null) parts.push(`HTTP ${event.http_status}`);
  if (event.retry_after) parts.push(`retry-after ${event.retry_after}`);
  if (event.response_content_type) parts.push(String(event.response_content_type));
  if (event.response_content_length) parts.push(`content-length ${event.response_content_length}`);
  if (event.response_bytes != null) parts.push(`response ${event.response_bytes} bytes`);
  if (event.response_chunk_count != null) parts.push(`${event.response_chunk_count} response chunks`);
  if (event.first_chunk_bytes != null) parts.push(`first byte chunk ${event.first_chunk_bytes} bytes`);
  if (event.audio_bytes != null) parts.push(`audio ${event.audio_bytes} bytes`);
  if (event.decoded_audio_bytes != null) parts.push(`decoded ${event.decoded_audio_bytes} bytes`);
  if (event.audio_chunk_count != null) parts.push(`${event.audio_chunk_count} audio chunks`);
  if (event.response_buffered != null) parts.push(`buffered=${Boolean(event.response_buffered)}`);
  if (event.streaming_response != null) parts.push(`streaming=${Boolean(event.streaming_response)}`);
  if (event.operation) parts.push(String(event.operation));
  if (event.audio_format) parts.push(String(event.audio_format));
  if (event.resample != null) parts.push(`resample=${Boolean(event.resample)}`);
  if (event.pcm_bytes != null) parts.push(`pcm ${event.pcm_bytes} bytes`);
  if (event.total_audio_bytes != null) parts.push(`total audio ${event.total_audio_bytes} bytes`);
  if (event.chunk_bytes != null) parts.push(`chunk ${event.chunk_bytes} bytes`);
  if (event.initial_buffer_ms != null) parts.push(`initial buffer ${event.initial_buffer_ms}ms`);
  if (event.silence_prefix_ms != null) parts.push(`silence prefix ${event.silence_prefix_ms}ms`);
  if (event.audio_ms != null) parts.push(`audio ${event.audio_ms}ms`);
  if (event.audio_ts != null) parts.push(`audioTs ${event.audio_ts}ms`);
  if (event.esp_offset_ms != null) parts.push(`ESP stream +${event.esp_offset_ms}ms`);
  if (event.relay_ms != null) parts.push(`stage ${event.relay_ms}ms`);
  if (event.prebuffer_bytes != null) parts.push(`prebuffer ${event.prebuffer_bytes} bytes`);
  if (event.source_rate && event.target_rate) parts.push(`${event.source_rate}Hz -> ${event.target_rate}Hz`);
  if (event.sample_rate) parts.push(`${event.sample_rate}Hz x${event.channels || 1}`);
  if (event.note) parts.push(String(event.note));
  return parts.join("; ") || String(event.name || "event").replaceAll("_", " ");
}

function renderTurnSummary(latency, selected) {
  const box = $("turn-summary");
  if (!box) return;
  const source = selected.turn || {};
  const summary = source.summary || latency.summary || {};
  const totalMs = summary.total_ms ?? summary.wake_to_complete_ms ?? summary.wake_to_session_completed_ms;
  const speakerStartMissing = latency.speaker_first_audio?.available === false && summary.wake_to_speaker_finished_ms != null;
  const note = speakerStartMissing
    ? "ESP reports speaker finish, but speaker-start/first-PCM is missing; start metrics remain unavailable until firmware reports that event."
    : latency.speaker_first_audio?.available === false
      ? "Speaker first PCM is not reported by ESP yet; Wake -> TTS text is the current proxy."
      : "";
  const textParts = [
    selected.source === "last_spoken" ? "Showing last spoken turn" : "",
    source.transcript ? `STT: ${compactSnippet(source.transcript, "")}` : "",
    source.assistant_text ? `LLM: ${compactSnippet(source.assistant_text, "")}` : "",
    summary.wake_to_first_tts_ms != null ? `Wake -> TTS text ${fmtMs(summary.wake_to_first_tts_ms)}` : "",
    summary.tts_text_to_speaker_ms != null ? `TTS text -> speaker ${fmtMs(summary.tts_text_to_speaker_ms)}` : "",
    summary.wake_to_speaker_ms != null ? `Wake -> speaker ${fmtMs(summary.wake_to_speaker_ms)}` : "",
    summary.tts_text_to_speaker_finished_ms != null ? `TTS text -> finish ${fmtMs(summary.tts_text_to_speaker_finished_ms)}` : "",
    summary.wake_to_speaker_finished_ms != null ? `Wake -> finish ${fmtMs(summary.wake_to_speaker_finished_ms)}` : "",
    totalMs != null ? `Total ${fmtMs(totalMs)}` : "",
  ].filter(Boolean);
  box.innerHTML = "";
  const main = document.createElement("p");
  main.textContent = textParts.length ? textParts.join(" | ") : "No completed voice turn timing yet.";
  box.appendChild(main);
  if (note) {
    const small = document.createElement("small");
    small.textContent = note;
    box.appendChild(small);
  }
}

function turnHistoryText(turn) {
  const reason = String(turn.reason || "").replaceAll("_", " ").trim();
  const textValue = String(turn.transcript || turn.assistant_text || "").trim();
  if (textValue) return compactSnippet(textValue, "completed turn");
  return reason ? `no speech / ${reason}` : "no speech / completed turn";
}

function renderTurnHistory(history) {
  const box = $("turn-history");
  if (!box) return;
  const rows = Array.isArray(history) ? history.slice(-3).reverse() : [];
  box.innerHTML = "";
  if (!rows.length) return;
  const title = document.createElement("b");
  title.textContent = "Recent turns";
  box.appendChild(title);
  rows.forEach((turn) => {
    const row = document.createElement("div");
    const summary = turn.summary || {};
    const when = Number(turn.ended_at || 0);
    const timeText = fmtClock(when);
    const speakerMs = summary.wake_to_speaker_ms;
    const finishedMs = summary.wake_to_speaker_finished_ms ?? summary.wake_to_complete_ms;
    const ttsMs = summary.wake_to_first_tts_ms;
    const totalMs = summary.total_ms ?? finishedMs ?? summary.wake_to_session_completed_ms;
    const meta = document.createElement("span");
    const textLine = document.createElement("p");
    meta.textContent = `${timeText} | ${
      speakerMs != null ? `speaker ${fmtMs(speakerMs)}`
        : finishedMs != null ? `finish ${fmtMs(finishedMs)}`
          : `TTS text ${fmtMs(ttsMs)}`
    } | total ${fmtMs(totalMs)}`;
    textLine.textContent = turnHistoryText(turn);
    row.append(meta, textLine);
    box.appendChild(row);
  });
}

function renderTurnTiming(latency = {}, items = []) {
  const box = $("timeline");
  if (!box) return;
  const history = Array.isArray(latency.history) ? latency.history : [];
  const selected = selectTimingTurn(latency);
  const stages = selected.turn?.stages?.length
    ? selected.turn.stages
    : fallbackLatencyStages(selected.turn?.events || latency.events || []);
  renderTurnSummary(latency, selected);
  renderTurnHistory(history);
  box.innerHTML = "";
  const visibleStages = stages.slice(-24);
  if (!visibleStages.length) {
    const idle = document.createElement("div");
    idle.className = "turn-stage-row idle";
    idle.innerHTML = "<b>Idle</b><span>No timing data yet.</span><em>Wake the robot or start a session.</em>";
    box.appendChild(idle);
    return;
  }
  visibleStages.forEach((stage) => {
    const row = document.createElement("div");
    row.className = `turn-stage-row ${stage.key || "step"}`;
    const timeEl = document.createElement("b");
    const labelEl = document.createElement("span");
    const detailEl = document.createElement("em");
    timeEl.textContent = `${fmtClock(stage.at, true)}  +${fmtMs(stage.ms)}`;
    labelEl.textContent = stage.label || String(stage.key || "Step").replaceAll("_", " ");
    detailEl.textContent = formatStageDetail(stage);
    row.append(timeEl, labelEl, detailEl);
    box.appendChild(row);
  });
  if (Array.isArray(items) && items.length && visibleStages.length < 5) {
    items.slice(-2).forEach((item) => {
      const row = document.createElement("div");
      row.className = "turn-stage-row pipe";
      row.innerHTML = `<b>--</b><span>${item.category || "PIPE"}</span><em>${item.message || ""}</em>`;
      box.appendChild(row);
    });
  }
}

function renderButtons() {
  if (!$("esp-commands") || !$("server-commands")) return;
  $("esp-commands").innerHTML = "";
  espCommands.forEach((cmd) => {
    const btn = document.createElement("button");
    btn.textContent = commandLabels[cmd] || cmd.replaceAll("_", " ");
    btn.title = cmd.startsWith("motor_") ? `${cmd} firmware komutu; N20 motor testi icin kullaniliyor.` : cmd;
    btn.onclick = () => guard("Command failed", () => sendCommand(cmd));
    $("esp-commands").appendChild(btn);
  });
  $("server-commands").innerHTML = "";
  serverCommands.forEach((cmd) => {
    const btn = document.createElement("button");
    btn.textContent = cmd.replaceAll("_", " ");
    btn.onclick = () => guard("Command failed", () => sendCommand(cmd));
    $("server-commands").appendChild(btn);
  });
}

async function sendCommand(command, payload = {}) {
  const result = await api("/api/command", { method: "POST", body: JSON.stringify({ command, payload }) });
  if (command === "clear_logs") {
    logs = [];
    expandedLogKey = "";
  }
  notice(result.message || `${command} sent`);
  renderLogs({ forceScroll: command === "clear_logs" });
  await loadStatus();
  return result;
}

function initSpeakerVolumeControl() {
  const slider = $("speaker-volume-slider");
  if (!slider) return;
  slider.oninput = () => {
    speakerVolumeEditing = true;
    updateSpeakerVolumeText(Number(slider.value || 0), speakerGainFromStatus(latestStatus.esp || {}));
  };
  slider.onchange = () =>
    guard("Speaker volume failed", async () => {
      const volume = clamp(Number(slider.value || 0), 0, 100);
      if (volume > 0) rememberSpeakerVolumeBeforeMute(volume);
      rememberSpeakerVolume(volume);
      try {
        await setSpeakerVolume(volume);
      } finally {
        speakerVolumeEditing = false;
        updateSpeakerVolumeUi(latestStatus.esp || {});
      }
    });
  slider.onblur = () => {
    speakerVolumeEditing = false;
    updateSpeakerVolumeUi(latestStatus.esp || {});
  };
}

async function recordMicDebug(channel) {
  const command = channel === "right" ? "capture_mic_right" : "capture_mic_left";
  const label = channel === "right" ? "RIGHT" : "LEFT";
  const espWsConnected = Boolean(latestStatus.esp?.ws_connected);
  notice(espWsConnected ? `${label} mic recording requested; wait a few seconds.` : `${label} mic requested, but ESP WS is offline.`);
  text("mic-debug-status", `${label} mic recording...`);
  await sendCommand(command);
  micDebugRefreshTimers.forEach((timer) => window.clearTimeout(timer));
  micDebugRefreshTimers = [1200, 4200, 6500].map((delay) =>
    window.setTimeout(() => {
      refreshMicDebug().catch(() => undefined);
      loadStatus().catch(() => undefined);
    }, delay)
  );
}

async function refreshMicDebug() {
  const info = await api("/api/mic/debug");
  renderMicDebug(info);
  return info;
}

async function playMicDebug(channel) {
  const info = await refreshMicDebug();
  const capture = info.captures?.[channel];
  if (!capture?.url) {
    notice(`${channel.toUpperCase()} mic kaydı henüz yok`);
    return;
  }
  const audio = $("mic-debug-audio");
  audio.src = cacheBustedPath(capture.url);
  await audio.play();
}

async function downloadMicDebug(channel) {
  const info = await refreshMicDebug();
  const capture = info.captures?.[channel];
  if (!capture?.url) {
    notice(`${channel.toUpperCase()} mic kaydı henüz yok`);
    return;
  }
  const a = document.createElement("a");
  a.href = cacheBustedPath(capture.url);
  a.download = capture.filename || `alice_mic_${channel}.wav`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function saveConfig() {
  await api("/api/config", { method: "POST", body: JSON.stringify(stripMasked(currentConfig)) });
  const nextToken = getDeep(currentConfig, "panel.token") || getDeep(currentConfig, "panel.password");
  if (nextToken && nextToken !== "********") rememberToken(nextToken);
  configDirty = false;
  notice("Config saved");
  await loadStatus();
}

async function exportConfig() {
  const includeSecrets = $("config-export-secrets").checked ? "true" : "false";
  const data = await api(`/api/config/export?include_secrets=${includeSecrets}`);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = includeSecrets === "true" ? "alice_config_with_secrets.json" : "alice_config.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

async function importConfig() {
  const input = $("config-import-file");
  const file = input.files && input.files[0];
  if (!file) return;
  const doc = JSON.parse(await file.text());
  await api("/api/config/import", { method: "POST", body: JSON.stringify(doc) });
  input.value = "";
  configDirty = false;
  notice("Config imported");
  await loadStatus();
}

async function downloadLogs() {
  const body = await api("/api/logs/download");
  const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "alice_logs.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

async function clearPipelineMessages() {
  await api("/api/pipeline/messages", { method: "DELETE" });
  notice("Pipeline messages cleared");
  await loadStatus();
}

async function downloadPipelineMessages() {
  const body = await api("/api/pipeline/messages/download");
  const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "alice_pipeline_messages.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

async function loadPrompts(preferredSlug) {
  const data = await api("/api/prompts");
  const select = $("prompt-select");
  const profiles = data.profiles || [];
  select.innerHTML = "";
  profiles.forEach((profile) => {
    const opt = document.createElement("option");
    opt.value = profile.slug;
    opt.textContent = profile.name;
    select.appendChild(opt);
  });
  const desired = [preferredSlug, data.active_profile, profiles[0]?.slug, "alice"].find((slug) =>
    slug && profiles.some((profile) => profile.slug === slug)
  );
  select.value = desired || "";
  select.onchange = () => loadPrompt(select.value);
  if (select.value) await loadPrompt(select.value);
}

async function loadPrompt(slug) {
  currentPrompt = await api(`/api/prompts/${slug}`);
  $("prompt-name").value = currentPrompt.name || "";
  $("prompt-description").value = currentPrompt.description || "";
  $("prompt-text").value = currentPrompt.prompt || "";
}

async function createPrompt() {
  const name = window.prompt("New prompt profile name", "Alice Copy");
  if (!name) return;
  const result = await api("/api/prompts", {
    method: "POST",
    body: JSON.stringify({
      name,
      description: "Custom prompt profile",
      prompt: $("prompt-text").value || currentPrompt.prompt || "",
    }),
  });
  notice("Prompt created");
  await loadPrompts(result.prompt.slug);
}

async function copyPrompt() {
  if (!currentPrompt.slug) return;
  const name = window.prompt("Copied prompt profile name", `${currentPrompt.name || currentPrompt.slug} Copy`);
  if (!name) return;
  const result = await api(`/api/prompts/${currentPrompt.slug}/copy`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
  notice("Prompt copied");
  await loadPrompts(result.prompt.slug);
}

async function deletePrompt() {
  if (!currentPrompt.slug) return;
  if (!window.confirm(`Delete prompt profile "${currentPrompt.name || currentPrompt.slug}"?`)) return;
  await api(`/api/prompts/${currentPrompt.slug}`, { method: "DELETE" });
  notice("Prompt deleted");
  await loadPrompts();
}

async function savePrompt() {
  currentPrompt.name = $("prompt-name").value;
  currentPrompt.description = $("prompt-description").value;
  currentPrompt.prompt = $("prompt-text").value;
  await api(`/api/prompts/${currentPrompt.slug}`, { method: "POST", body: JSON.stringify(currentPrompt) });
  notice("Prompt saved");
  await loadPrompts(currentPrompt.slug);
}

async function activatePrompt() {
  await api(`/api/prompts/${currentPrompt.slug}/activate`, { method: "POST" });
  notice("Prompt activated");
  await loadPrompts();
}

async function runPipeline() {
  const input = $("pipeline-input");
  if (!input.value.trim()) return;
  await api("/api/pipeline/text", { method: "POST", body: JSON.stringify({ text: input.value }) });
  input.value = "";
  await loadStatus();
}

async function runTtsTest() {
  const input = $("pipeline-input");
  if (!input.value.trim()) return;
  await api("/api/pipeline/tts/text", { method: "POST", body: JSON.stringify({ text: input.value }) });
  input.value = "";
  await loadStatus();
}

async function runTtsLatencyBenchmark() {
  await api("/api/pipeline/tts/benchmark", { method: "POST", body: JSON.stringify({}) });
  notice("TTS latency test queued");
  await loadStatus();
}

async function startVoiceSession() {
  await api("/api/pipeline/session/start", { method: "POST", body: JSON.stringify({ mode: "manual" }) });
  notice("Voice session started");
  await loadStatus();
}

async function stopVoiceSession() {
  await api("/api/pipeline/session/stop", { method: "POST", body: JSON.stringify({ reason: "ui_stop" }) });
  notice("Voice session stopped");
  await loadStatus();
}

async function cancelResponse() {
  await api("/api/pipeline/cancel", { method: "POST", body: JSON.stringify({ reason: "ui_cancel" }) });
  notice("Response cancel requested");
  await loadStatus();
}

function connectLogs() {
  const seq = ++logSocketSeq;
  if (logSocket) {
    logSocket.onclose = null;
    logSocket.close();
  }
  loadLogSnapshot().catch(() => undefined);
  const socket = new WebSocket(wsPath("/api/ws/logs"));
  logSocket = socket;
  socket.onopen = () => notice("");
  socket.onmessage = (event) => {
    if (paused) return;
    const doc = JSON.parse(event.data);
    const incoming = doc.entries || [];
    if (!incoming.length) return;
    mergeLogs(incoming);
    renderLogCategories();
    renderLogs();
  };
  socket.onerror = () => {
    notice("Log WebSocket baglanamadi; HTTP log snapshot kullaniliyor.");
    loadLogSnapshot().catch(() => undefined);
  };
  socket.onclose = () => {
    window.setTimeout(() => {
      if (logSocketSeq === seq && !paused) connectLogs();
    }, 3000);
  };
}

async function loadLogSnapshot() {
  const data = await api("/api/logs?limit=250");
  mergeLogs(data.entries || []);
  renderLogCategories();
  renderLogs();
}

function mergeLogs(entries) {
  const map = new Map(logs.map((entry) => [entry.id, entry]));
  entries.forEach((entry) => {
    if (entry && entry.id) map.set(entry.id, entry);
  });
  logs = Array.from(map.values()).sort((a, b) => (a.ts || 0) - (b.ts || 0)).slice(-1000);
}

function connectEvents() {
  const seq = ++eventSocketSeq;
  if (eventSocket) {
    eventSocket.onclose = null;
    eventSocket.close();
  }
  const socket = new WebSocket(wsPath("/api/ws/events"));
  eventSocket = socket;
  socket.onmessage = (event) => {
    const doc = JSON.parse(event.data);
    if (doc.type === "snapshot") {
      if (doc.payload?.esp?.radar) renderRadar(doc.payload.esp.radar);
      scheduleStatusRefresh();
      return;
    }
    if (doc.type === "esp_event" && doc.payload?.type === "radar_targets") {
      renderRadar(doc.payload.payload || {});
      return;
    }
    if (doc.type === "esp_status" || doc.type === "pipeline_status" || doc.type === "config_updated" || doc.type === "esp_event") {
      scheduleStatusRefresh();
    }
  };
  socket.onclose = () => {
    window.setTimeout(() => {
      if (eventSocketSeq === seq) connectEvents();
    }, 4000);
  };
}

function renderLogCategories() {
  const select = $("log-category");
  const old = select.value;
  const cats = ["ALL", ...Array.from(new Set(logs.map((entry) => entry.category))).sort()];
  select.innerHTML = cats.map((cat) => `<option>${cat}</option>`).join("");
  select.value = cats.includes(old) ? old : "ALL";
  renderLogControls();
  renderLogSummary();
}

function setLogPreset(preset) {
  logPreset = LOG_PRESETS[preset] ? preset : "all";
  localStorage.setItem("alice_log_preset", logPreset);
  renderLogControls();
  renderLogs({ forceScroll: true });
}

function renderLogControls() {
  document.querySelectorAll("[data-log-preset]").forEach((button) => {
    button.classList.toggle("active", button.dataset.logPreset === logPreset);
  });
  const focus = $("log-focus-toggle");
  if (focus) {
    focus.classList.toggle("active", logFocusMode);
    focus.textContent = logFocusMode ? "Focus on" : "Focus";
  }
}

function renderLogSummary() {
  const total = logs.length;
  const errors = logs.filter((entry) => entry.level === "ERROR").length;
  const warns = logs.filter((entry) => entry.level === "WARN").length;
  const last = logs[logs.length - 1];
  $("log-total").textContent = String(total);
  $("log-errors").textContent = String(errors);
  $("log-warns").textContent = String(warns);
  $("log-last").textContent = last ? `${last.category || "-"} ${new Date(last.ts * 1000).toLocaleTimeString()}` : "-";
}

function isRoutineLog(entry) {
  if (!logFocusMode) return false;
  if (entry.level === "ERROR" || entry.level === "WARN") return false;
  const message = String(entry.message || "");
  if (LOG_FOCUS_NOISE.some((pattern) => message.includes(pattern))) return true;
  if (entry.category === "ESP" && /^(panel audio stream|ESP audio start|ESP audio stream acknowledged)/i.test(message)) return true;
  if (entry.category === "PIPELINE" && /session completed|completed without assistant text/i.test(message)) return true;
  return false;
}

function logMatchesPreset(entry) {
  const preset = LOG_PRESETS[logPreset] || LOG_PRESETS.all;
  if (preset.level && entry.level !== preset.level) return false;
  if (preset.categories && !preset.categories.includes(entry.category)) return false;
  return true;
}

function logEntryKey(entry, index) {
  if (entry?.id) return `id:${entry.id}`;
  const detailText = entry?.details ? JSON.stringify(entry.details) : "";
  return `${index}|${entry?.ts || 0}|${entry?.level || ""}|${entry?.category || ""}|${entry?.message || ""}|${detailText}`;
}

function formatLogDetails(entry) {
  const details = entry?.details && Object.keys(entry.details).length ? entry.details : null;
  if (details) return JSON.stringify(details, null, 2);
  return JSON.stringify({
    time: entry?.ts ? new Date(entry.ts * 1000).toLocaleString() : "",
    level: entry?.level || "",
    category: entry?.category || "",
    message: entry?.message || ""
  }, null, 2);
}

function toggleLogDetails(key) {
  if (!key) return;
  expandedLogKey = expandedLogKey === key ? "" : key;
  renderLogs({ revealExpanded: true });
}

function initLogListInteractions() {
  const list = $("log-list");
  if (!list || list.dataset.interactionsBound) return;
  list.dataset.interactionsBound = "1";
  list.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : event.target?.parentElement;
    if (!target || target.closest("pre")) return;
    const row = target.closest(".log-row");
    if (!row || !list.contains(row)) return;
    toggleLogDetails(row.dataset.logKey || "");
  });
  list.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const target = event.target instanceof Element ? event.target : event.target?.parentElement;
    const row = target?.closest(".log-row");
    if (!row || !list.contains(row)) return;
    event.preventDefault();
    toggleLogDetails(row.dataset.logKey || "");
  });
}

function renderLogs(options = {}) {
  const q = $("log-search").value.toLowerCase().trim();
  const level = $("log-level").value;
  const cat = $("log-category").value;
  const rows = logs.map((entry, index) => ({ entry, index })).filter(({ entry }) => {
    if (!logMatchesPreset(entry)) return false;
    if (isRoutineLog(entry)) return false;
    if (level !== "ALL" && entry.level !== level) return false;
    if (cat !== "ALL" && entry.category !== cat) return false;
    if (!q) return true;
    return `${entry.level} ${entry.category} ${entry.message} ${JSON.stringify(entry.details || {})}`.toLowerCase().includes(q);
  }).slice(-220);
  const list = $("log-list");
  let expandedTarget = null;
  renderLogSummary();
  renderLogControls();
  keepAutoScrolled(list, () => {
    list.innerHTML = "";
    rows.forEach(({ entry, index }) => {
      const row = document.createElement("div");
      const key = logEntryKey(entry, index);
      row.className = `log-row ${String(entry.level || "").toLowerCase()}`;
      row.dataset.logKey = key;
      const details = formatLogDetails(entry);
      const isExpanded = expandedLogKey === key;
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-expanded", isExpanded ? "true" : "false");
      row.innerHTML = `<time>${new Date(entry.ts * 1000).toLocaleTimeString()}</time><b>${entry.level}</b><span>${entry.category}</span><p></p>`;
      row.querySelector("p").textContent = entry.message || "";
      row.classList.add("has-details");
      if (isExpanded) {
        row.classList.add("expanded");
      }
      list.appendChild(row);
      if (isExpanded) {
        const detailBlock = document.createElement("pre");
        detailBlock.className = "log-detail";
        detailBlock.textContent = details;
        detailBlock.addEventListener("click", (event) => event.stopPropagation());
        list.appendChild(detailBlock);
        expandedTarget = detailBlock;
      }
    });
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "log-empty";
      empty.textContent = "No logs match the current filters.";
      list.appendChild(empty);
    }
  }, Boolean(options.forceScroll));
  if (expandedTarget && options.revealExpanded !== false) {
    window.requestAnimationFrame(() => keepChildVisible(list, expandedTarget, 24));
  }
}

window.addEventListener("load", boot);

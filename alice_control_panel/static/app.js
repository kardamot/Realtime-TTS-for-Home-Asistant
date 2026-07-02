const espCommands = [
  "test_speaker", "test_mic", "capture_mic", "wake_on", "wake_off",
  "motors_on", "motors_off", "amp_mute_on", "amp_mute_off", "radar_calibrate_empty", "radar_clear_empty", "reconnect", "reboot"
];
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
  "TTS relay websocket started",
  "TTS relay websocket disconnected",
  "Configuration updated",
];

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
  [".connections-panel > header h2", "connections"],
  ["#logs > header h2", "logs"],
  ["#radar > header h2", "hardware"],
  ["#pipeline > header h2", "pipeline"],
  ["#commands > header h2", "commands"],
  ["#prompts > header h2", "prompts"],
  ["#config > header h2", "config"],
  ["#config .config-group:nth-of-type(1) h3", "panelEsp", "panelEspFields"],
  ["#config .config-group:nth-of-type(2) h3", "liveVoice", "liveVoiceFields"],
  ["#config .config-group:nth-of-type(3) h3", "sttVad", "sttVadFields"],
  ["#config .config-group:nth-of-type(4) h3", "homeAssistant", "homeAssistantFields"],
  ["#config .config-group:nth-of-type(5) h3", "llm", "llmFields"],
  ["#config .config-group:nth-of-type(6) h3", "tts", "ttsFields"]
];

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
  if (!["trace", "messages", "timing"].includes(pipelineView)) pipelineView = "trace";
  document.querySelectorAll("[data-pipeline-tab]").forEach((button) => {
    const active = button.dataset.pipelineTab === pipelineView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-pipeline-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.pipelinePanel === pipelineView);
  });
  const pipeline = $("pipeline");
  pipeline?.classList.toggle("timing-expanded", pipelineView === "timing");
  pipeline?.classList.toggle("messages-view", pipelineView === "messages");
  if (pipelineView === "messages") {
    window.requestAnimationFrame(() => {
      const messages = $("pipeline-messages");
      if (messages) messages.scrollTop = messages.scrollHeight;
    });
  }
}

function initPipelineTabs() {
  document.querySelectorAll("[data-pipeline-tab]").forEach((button) => {
    button.onclick = () => {
      pipelineView = button.dataset.pipelineTab || "trace";
      localStorage.setItem("alice_pipeline_view", pipelineView);
      syncPipelineTabs();
    };
  });
  syncPipelineTabs();
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
  document.querySelectorAll("[data-daily-command]").forEach((button) => {
    const command = button.dataset.dailyCommand;
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
  initAutoScrollContainers();
  initHelpBubbles();
  initRadarControls();
  initCommandTabs();
  initPipelineTabs();
  initDriveControls();
  renderButtons();
  initDailyCommandButtons();
  initProviderSwitches();
  $("refresh-btn").onclick = () => guard("Refresh failed", loadStatus);
  $("unlock-btn").onclick = () => guard("Unlock failed", unlock);
  $("pipeline-send").onclick = () => guard("Pipeline failed", runPipeline);
  $("pipeline-tts-send").onclick = () => guard("TTS test failed", runTtsTest);
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
  text("backend-version", `${backend.service || "alice_control_panel"} ${backend.version || ""} - FastAPI backend online`);
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
  const liveMode = Boolean(realtime.enabled || realtime.active || realtime.connected);
  const liveProvider = `${realtime.provider || "openai"} realtime`;
  text("conn-esp", esp.online ? "online" : "offline");
  text("conn-stt", liveMode ? `${liveProvider} / ${realtime.transcription_model || "stt n/a"}` : data.stt?.provider || "faster_whisper");
  text("conn-llm", liveMode ? `${liveProvider} / ${realtime.model || "model n/a"}` : `${data.llm?.provider || "openai"} / ${data.llm?.model || "n/a"}`);
  text("conn-tts", `${data.tts?.provider || "openai"} / ${data.tts?.pcm_sample_rate || "n/a"}`);
  text(
    "conn-ha",
    data.ha_bridge?.connected
      ? data.ha_bridge?.entity_scope
        ? `allowlist ${data.ha_bridge?.explicit_entity_count || data.ha_bridge?.allowlist_count || 0}`
        : "no allowlist"
      : data.ha_bridge?.enabled
        ? "not ready"
        : "disabled"
  );
  text("conn-reconnects", formatReconnects(esp));
  text("conn-esp-ws", esp.ws_connected ? "connected" : "offline");
  setAutoText("last-error", esp.last_error || esp.last_ws_error || "");
  text("hw-mic", esp.hardware?.mic || "unknown");
  text("hw-speaker", esp.hardware?.speaker || "unknown");
  updateSpeakerVolumeUi(esp);
  text("hw-radar", esp.hardware?.radar || esp.radar?.state || "unknown");
  text("hw-motion", formatMotionSensor(esp.hardware || {}));
  text("hw-touch", formatTouchSensor(esp.hardware || {}));
  text("hw-servo", esp.hardware?.servo_position || "center");
  text("hw-amp", esp.hardware?.amp_muted == null ? "unknown" : esp.hardware.amp_muted ? "muted" : "active");
  text("hw-wake", esp.hardware?.wake_enabled == null ? "unknown" : esp.hardware.wake_enabled ? "on" : "off");
  text("hw-state", esp.state || "OFFLINE");
  syncDailyBehaviorButtons(esp, pipe);
  renderPipelineTrace(pipe, realtime, data);
  renderPipelineMessages(pipe.messages || []);
  renderRealtimeLatency(realtime.latency || {});
  renderRadar(esp.radar || latestRadar || {});
  renderMicDebug(pipe.mic_debug || {});
  renderTurnTiming(realtime.latency || {}, pipe.timeline || []);
  if (!configDirty) fillConfig();
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

function radarStateTone(state, fresh, ready) {
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
  const textValue = String(value || "").trim();
  return textValue || fallback;
}

function lastPipelineMessage(messages, roles) {
  const list = Array.isArray(messages) ? messages : [];
  const accepted = Array.isArray(roles) ? roles : [roles];
  for (let index = list.length - 1; index >= 0; index -= 1) {
    const item = list[index] || {};
    if (accepted.includes(item.role) && String(item.text || "").trim()) return item;
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
      timeEl.textContent = Number.isFinite(ts) && ts > 0 ? new Date(ts * 1000).toLocaleTimeString() : "--:--:--";
      roleEl.textContent = role.toUpperCase();
      sourceEl.textContent = String(item.source || "-").replaceAll("_", " ");
      textEl.textContent = String(item.text || "");
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
    ["LLM -> TTS", summary.first_delta_to_first_chunk_ms],
    ["Wake -> TTS text", summary.wake_to_first_tts_ms],
    ["Turn total", summary.wake_to_complete_ms ?? summary.total_ms],
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
    summary.wake_to_first_tts_ms != null
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
  return !["session_completed", "response_cancelled", "client_cancelled"].includes(last);
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
  if (event.chars != null) parts.push(`${event.chars} chars`);
  if (event.audio_ms != null) parts.push(`audio ${event.audio_ms}ms`);
  if (event.audio_ts != null) parts.push(`audioTs ${event.audio_ts}ms`);
  if (event.source_rate && event.target_rate) parts.push(`${event.source_rate}Hz -> ${event.target_rate}Hz`);
  return parts.join("; ") || String(event.name || "event").replaceAll("_", " ");
}

function renderTurnSummary(latency, selected) {
  const box = $("turn-summary");
  if (!box) return;
  const note = latency.speaker_first_audio?.available === false
    ? "Speaker first PCM is not reported by ESP yet; Wake -> TTS text is the current proxy."
    : "";
  const source = selected.turn || {};
  const summary = source.summary || latency.summary || {};
  const textParts = [
    selected.source === "last_spoken" ? "Showing last spoken turn" : "",
    source.transcript ? `STT: ${compactSnippet(source.transcript, "")}` : "",
    source.assistant_text ? `LLM: ${compactSnippet(source.assistant_text, "")}` : "",
    summary.wake_to_first_tts_ms != null ? `Wake -> TTS text ${fmtMs(summary.wake_to_first_tts_ms)}` : "",
    summary.wake_to_complete_ms != null ? `Total ${fmtMs(summary.wake_to_complete_ms)}` : "",
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
    const timeText = Number.isFinite(when) && when > 0 ? new Date(when * 1000).toLocaleTimeString() : "--:--:--";
    const meta = document.createElement("span");
    const textLine = document.createElement("p");
    meta.textContent = `${timeText} | TTS ${fmtMs(summary.wake_to_first_tts_ms)} | total ${fmtMs(summary.wake_to_complete_ms ?? summary.total_ms)}`;
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
  const visibleStages = stages.slice(-10);
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
    timeEl.textContent = `+${fmtMs(stage.ms)}`;
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
  if (command === "clear_logs") logs = [];
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

function renderLogs(options = {}) {
  const q = $("log-search").value.toLowerCase().trim();
  const level = $("log-level").value;
  const cat = $("log-category").value;
  const rows = logs.filter((entry) => {
    if (!logMatchesPreset(entry)) return false;
    if (isRoutineLog(entry)) return false;
    if (level !== "ALL" && entry.level !== level) return false;
    if (cat !== "ALL" && entry.category !== cat) return false;
    if (!q) return true;
    return `${entry.level} ${entry.category} ${entry.message} ${JSON.stringify(entry.details || {})}`.toLowerCase().includes(q);
  }).slice(-220);
  const list = $("log-list");
  renderLogSummary();
  renderLogControls();
  keepAutoScrolled(list, () => {
    list.innerHTML = "";
    rows.forEach((entry) => {
      const row = document.createElement("div");
      row.className = `log-row ${String(entry.level || "").toLowerCase()}`;
      const details = entry.details && Object.keys(entry.details).length ? JSON.stringify(entry.details, null, 2) : "";
      row.innerHTML = `<time>${new Date(entry.ts * 1000).toLocaleTimeString()}</time><b>${entry.level}</b><span>${entry.category}</span><p></p>${details ? "<pre></pre>" : ""}`;
      row.querySelector("p").textContent = entry.message || "";
      if (details) {
        row.classList.add("has-details");
        const detailBlock = row.querySelector("pre");
        detailBlock.textContent = details;
        detailBlock.addEventListener("click", (event) => event.stopPropagation());
        row.addEventListener("click", () => {
          row.classList.toggle("expanded");
          if (row.classList.contains("expanded")) keepChildVisible(list, row);
        });
      }
      list.appendChild(row);
    });
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "log-empty";
      empty.textContent = "No logs match the current filters.";
      list.appendChild(empty);
    }
  }, Boolean(options.forceScroll));
}

window.addEventListener("load", boot);

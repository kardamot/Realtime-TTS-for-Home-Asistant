const espCommands = [
  "test_speaker", "test_mic", "capture_mic", "wake_on", "wake_off", "servo_left", "servo_center",
  "servo_right", "amp_mute_on", "amp_mute_off", "reconnect", "reboot"
];
const serverCommands = [
  "restart_stt", "restart_tts", "reload_prompt", "clear_logs",
  "start_voice_session", "stop_voice_session", "cancel_response",
  "safe_mode_on", "safe_mode_off"
];
let token = localStorage.getItem("alice_panel_token") || "";
let currentConfig = {};
let currentPrompt = {};
let logs = [];
let paused = false;
let configDirty = false;
let logSocket = null;
let logSocketSeq = 0;
let eventSocket = null;
let eventSocketSeq = 0;
let statusTimer = null;
let statusRefreshTimer = null;
let micDebug = {};
const autoScrollState = new WeakMap();
let helpPopover = null;

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
      "Üst bölüm ESP komutlarıdır: hoparlör testi, mikrofon testi, wake aç/kapat, servo hareketleri, amfi mute, reconnect ve reboot gibi doğrudan robota giden işler burada durur.",
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
      ["Realtime STT prompt", "Transkripsiyon icin ipucu metnidir. Turkce, Alice, yerel isimler veya sik yanlis duyulan kelimeleri buraya yazmak tanimayi iyilestirebilir."],
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
  [".hardware-panel > header h2", "hardware"],
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

async function boot() {
  initAutoScrollContainers();
  initHelpBubbles();
  renderButtons();
  initProviderSwitches();
  $("refresh-btn").onclick = () => guard("Refresh failed", loadStatus);
  $("unlock-btn").onclick = () => guard("Unlock failed", unlock);
  $("pipeline-send").onclick = () => guard("Pipeline failed", runPipeline);
  $("pipeline-tts-send").onclick = () => guard("TTS test failed", runTtsTest);
  $("session-start").onclick = () => guard("Session start failed", startVoiceSession);
  $("session-stop").onclick = () => guard("Session stop failed", stopVoiceSession);
  $("response-cancel").onclick = () => guard("Response cancel failed", cancelResponse);
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
  text("conn-esp", esp.online ? "online" : "offline");
  text("conn-stt", data.stt?.provider || "faster_whisper");
  text("conn-llm", `${data.llm?.provider || "openai"} / ${data.llm?.model || "n/a"}`);
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
  text("hw-servo", esp.hardware?.servo_position || "center");
  text("hw-amp", esp.hardware?.amp_muted == null ? "unknown" : esp.hardware.amp_muted ? "muted" : "active");
  text("hw-wake", esp.hardware?.wake_enabled == null ? "unknown" : esp.hardware.wake_enabled ? "on" : "off");
  text("hw-state", esp.state || "OFFLINE");
  setAutoText("stt-text", pipe.stt_result || pipe.last_user_text || "No utterance yet");
  setAutoText("llm-text", pipe.llm_response || "FastAPI backend ready. Send a text test or configure providers.");
  renderMicDebug(pipe.mic_debug || {});
  renderTimeline(pipe.timeline || []);
  if (!configDirty) fillConfig();
}

function renderMicDebug(info) {
  micDebug = info || {};
  const captures = micDebug.captures || {};
  const left = captures.left;
  const right = captures.right;
  const latest = [left, right].filter(Boolean).sort((a, b) => Number(b.stored_at || 0) - Number(a.stored_at || 0))[0];
  text(
    "mic-debug-status",
    latest
      ? `latest ${latest.channel || "mic"} - ${fmtSeconds(Math.round((Date.now() / 1000) - Number(latest.stored_at || 0)))} ago`
      : "No debug capture yet"
  );
  text(
    "mic-debug-meta",
    latest
      ? `${String(latest.channel || "mic").toUpperCase()} | ${latest.duration_sec || 0}s | ${latest.bytes_buffered || 0} bytes | rms ${latest.rms || 0} | peak ${latest.peak || 0}`
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

function renderTimeline(items) {
  const box = $("timeline");
  const list = items.slice(-6);
  keepAutoScrolled(box, () => {
    box.innerHTML = list.length ? "" : "<div><b>IDLE</b><span>Waiting for audio/text</span></div>";
    list.forEach((item) => {
      const row = document.createElement("div");
      row.innerHTML = `<b>${item.category || "STEP"}</b><span>${item.message || ""}</span>`;
      box.appendChild(row);
    });
  });
}

function renderButtons() {
  $("esp-commands").innerHTML = "";
  espCommands.forEach((cmd) => {
    const btn = document.createElement("button");
    btn.textContent = cmd.replaceAll("_", " ");
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

async function sendCommand(command) {
  const result = await api("/api/command", { method: "POST", body: JSON.stringify({ command, payload: {} }) });
  if (command === "clear_logs") logs = [];
  notice(result.message || `${command} sent`);
  renderLogs({ forceScroll: command === "clear_logs" });
  await loadStatus();
}

async function recordMicDebug(channel) {
  const command = channel === "right" ? "capture_mic_right" : "capture_mic_left";
  notice(`${channel.toUpperCase()} mic recording requested`);
  await sendCommand(command);
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
    if (doc.type === "snapshot" || doc.type === "esp_status" || doc.type === "pipeline_status" || doc.type === "config_updated") {
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
}

function renderLogs(options = {}) {
  const q = $("log-search").value.toLowerCase().trim();
  const level = $("log-level").value;
  const cat = $("log-category").value;
  const rows = logs.filter((entry) => {
    if (level !== "ALL" && entry.level !== level) return false;
    if (cat !== "ALL" && entry.category !== cat) return false;
    if (!q) return true;
    return `${entry.level} ${entry.category} ${entry.message} ${JSON.stringify(entry.details || {})}`.toLowerCase().includes(q);
  }).slice(-220);
  const list = $("log-list");
  keepAutoScrolled(list, () => {
    list.innerHTML = "";
    rows.forEach((entry) => {
      const row = document.createElement("div");
      row.className = `log-row ${String(entry.level || "").toLowerCase()}`;
      row.innerHTML = `<time>${new Date(entry.ts * 1000).toLocaleTimeString()}</time><b>${entry.level}</b><span>${entry.category}</span><p></p>`;
      row.querySelector("p").textContent = entry.message || "";
      list.appendChild(row);
    });
  }, Boolean(options.forceScroll));
}

window.addEventListener("load", boot);

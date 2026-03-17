import streamlit as st
import pandas as pd
import time
import math
import re
import string
from urllib.parse import urlparse, parse_qs

from google_play_scraper import search as gp_search, app as gp_app
from sklearn.feature_extraction.text import CountVectorizer

# ---------- NLTK SAFE ----------
import nltk
for resource, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords"),
                        ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(path)
    except:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ============================================================
# COUNTRY / TIER DATA  (code, display_name)
# ============================================================
TIER1 = [
    ("us","United States"),("gb","United Kingdom"),("ca","Canada"),
    ("au","Australia"),("de","Germany"),("fr","France"),("jp","Japan"),
    ("kr","South Korea"),("nl","Netherlands"),("se","Sweden"),
    ("no","Norway"),("dk","Denmark"),("fi","Finland"),("ch","Switzerland"),
    ("ie","Ireland"),("nz","New Zealand"),("sg","Singapore"),("il","Israel"),
    ("at","Austria"),("be","Belgium"),
]
TIER2 = [
    ("in","India"),("br","Brazil"),("mx","Mexico"),("ru","Russia"),
    ("es","Spain"),("it","Italy"),("pl","Poland"),("tr","Turkey"),
    ("ar","Argentina"),("za","South Africa"),("ae","UAE"),("sa","Saudi Arabia"),
    ("ng","Nigeria"),("eg","Egypt"),("id","Indonesia"),("th","Thailand"),
    ("vn","Vietnam"),("ph","Philippines"),("pk","Pakistan"),("my","Malaysia"),
    ("co","Colombia"),("cl","Chile"),("pe","Peru"),("cz","Czech Republic"),
    ("ro","Romania"),("hu","Hungary"),("pt","Portugal"),("gr","Greece"),
    ("ua","Ukraine"),("kw","Kuwait"),
]
TIER3 = [
    ("bd","Bangladesh"),("np","Nepal"),("lk","Sri Lanka"),("mm","Myanmar"),
    ("kh","Cambodia"),("la","Laos"),("mn","Mongolia"),("uz","Uzbekistan"),
    ("kz","Kazakhstan"),("az","Azerbaijan"),("ge","Georgia"),("am","Armenia"),
    ("by","Belarus"),("md","Moldova"),("rs","Serbia"),("bg","Bulgaria"),
    ("hr","Croatia"),("si","Slovenia"),("sk","Slovakia"),("lt","Lithuania"),
    ("lv","Latvia"),("ee","Estonia"),("is","Iceland"),("mt","Malta"),
    ("cy","Cyprus"),("lu","Luxembourg"),("tn","Tunisia"),("ma","Morocco"),
    ("dz","Algeria"),("ly","Libya"),("gh","Ghana"),("ke","Kenya"),
    ("tz","Tanzania"),("et","Ethiopia"),("ug","Uganda"),("ci","Ivory Coast"),
    ("cm","Cameroon"),("sn","Senegal"),("ao","Angola"),("mz","Mozambique"),
    ("zw","Zimbabwe"),("zm","Zambia"),("bw","Botswana"),("rw","Rwanda"),
]
TIER4 = [
    ("sd","Sudan"),("so","Somalia"),("ye","Yemen"),("iq","Iraq"),
    ("sy","Syria"),("jo","Jordan"),("lb","Lebanon"),("qa","Qatar"),
    ("bh","Bahrain"),("om","Oman"),("bo","Bolivia"),("ec","Ecuador"),
    ("py","Paraguay"),("uy","Uruguay"),("ve","Venezuela"),("cr","Costa Rica"),
    ("gt","Guatemala"),("hn","Honduras"),("sv","El Salvador"),("ni","Nicaragua"),
    ("pa","Panama"),("do","Dominican Rep."),("jm","Jamaica"),("tt","Trinidad"),
    ("ht","Haiti"),("bb","Barbados"),("bs","Bahamas"),("gy","Guyana"),
    ("sr","Suriname"),("fj","Fiji"),("pg","Papua New Guinea"),("ws","Samoa"),
    ("to","Tonga"),("vu","Vanuatu"),("pw","Palau"),("fm","Micronesia"),
    ("mh","Marshall Islands"),("nr","Nauru"),("tv","Tuvalu"),("ki","Kiribati"),
]

ALL_TIERS = {
    "🌍 All Countries":                               TIER1+TIER2+TIER3+TIER4,
    "⭐ Tier 1 — High Income (US, UK, DE, JP...)":   TIER1,
    "🌐 Tier 2 — Emerging Markets (IN, BR, MX...)":  TIER2,
    "🌏 Tier 3 — Developing Markets (BD, GH, KE...)":TIER3,
    "🗺️ Tier 4 — Frontier Markets (BO, FJ, KI...)":  TIER4,
}
ALL_COUNTRY_PAIRS  = TIER1+TIER2+TIER3+TIER4
ALL_COUNTRY_CODES  = [c for c,_ in ALL_COUNTRY_PAIRS]
COUNTRY_LABEL_MAP  = {c: f"{c.upper()} — {n}" for c,n in ALL_COUNTRY_PAIRS}

# ============================================================
# LOCALIZATION: language code + stopwords lang + NLTK lang
# Scraped from Play Store in that language so keywords are native.
# Templates are pre-translated per language.
# ============================================================
LOCALES = {
    "🇺🇸 English":              {"code":"en","nltk":"english","name":"English"},
    "🇸🇦 Arabic":               {"code":"ar","nltk":"arabic","name":"Arabic"},
    "🇧🇩 Bengali":              {"code":"bn","nltk":"english","name":"Bengali"},
    "🇩🇪 German":               {"code":"de","nltk":"german","name":"German"},
    "🇪🇸 Spanish":              {"code":"es","nltk":"spanish","name":"Spanish"},
    "🇮🇷 Persian / Farsi":      {"code":"fa","nltk":"english","name":"Persian"},
    "🇫🇷 French":               {"code":"fr","nltk":"french","name":"French"},
    "🇮🇳 Hindi":                {"code":"hi","nltk":"english","name":"Hindi"},
    "🇮🇩 Indonesian":           {"code":"id","nltk":"english","name":"Indonesian"},
    "🇮🇹 Italian":              {"code":"it","nltk":"english","name":"Italian"},
    "🇯🇵 Japanese":             {"code":"ja","nltk":"english","name":"Japanese"},
    "🇰🇷 Korean":               {"code":"ko","nltk":"english","name":"Korean"},
    "🇲🇾 Malay":                {"code":"ms","nltk":"english","name":"Malay"},
    "🇳🇱 Dutch":                {"code":"nl","nltk":"english","name":"Dutch"},
    "🇧🇷 Portuguese (Brazil)":  {"code":"pt","nltk":"portuguese","name":"Portuguese"},
    "🇷🇺 Russian":              {"code":"ru","nltk":"russian","name":"Russian"},
    "🇵🇰 Urdu":                 {"code":"ur","nltk":"english","name":"Urdu"},
    "🇻🇳 Vietnamese":           {"code":"vi","nltk":"english","name":"Vietnamese"},
    "🇨🇳 Chinese Simplified":   {"code":"zh","nltk":"english","name":"Chinese"},
    "🇹🇷 Turkish":              {"code":"tr","nltk":"english","name":"Turkish"},
    "🇵🇱 Polish":               {"code":"pl","nltk":"english","name":"Polish"},
    "🇺🇦 Ukrainian":            {"code":"uk","nltk":"english","name":"Ukrainian"},
    "🇬🇷 Greek":                {"code":"el","nltk":"english","name":"Greek"},
    "🇸🇪 Swedish":              {"code":"sv","nltk":"english","name":"Swedish"},
    "🇹🇭 Thai":                 {"code":"th","nltk":"english","name":"Thai"},
    "🇵🇭 Filipino":             {"code":"tl","nltk":"english","name":"Filipino"},
}

# Pre-translated ASO template snippets per language
# Keys: intro, problem, features_header, why_header, perfect_header,
#       perfect_1, perfect_2, perfect_3, download_cta,
#       feat_fast, feat_smart, feat_clean, feat_offline, feat_updates, feat_battery
LANG_TEMPLATES = {
    "en": {
        "intro":          "{brand} is built for people who want real {p1} without the clutter.",
        "problem":        "Most apps in this space are bloated, full of ads, or simply don't deliver. {brand} is different — designed to give you smooth {p2}, reliable {s1}, and everything you need to get things done faster.",
        "use_case":       "Whether you need {lt1} or {lt2}, {brand} has you covered.",
        "features_header":"WHAT YOU GET",
        "why_header":     "WHY USERS CHOOSE {brand_upper}",
        "why_body":       "Users stick with {brand} because it combines powerful {p3} with a clean experience. The {s1} is seamless, {s2} speaks for itself, and you'll feel the difference from day one.",
        "perfect_header": "PERFECT FOR",
        "perfect_1":      "📱 Everyday users who want something reliable",
        "perfect_2":      "⚡ Power users who need fast {p2} without compromise",
        "perfect_3":      "🎯 Anyone tired of slow, ad-heavy alternatives",
        "download_cta":   "Download {brand} today and discover why it's the go-to {p1} app for thousands of users worldwide.",
        "feat_fast":      "Lightning-fast {p1} with zero lag",
        "feat_smart":     "Smart {p2} engine built for real users",
        "feat_clean":     "Clean, intuitive interface — no learning curve",
        "feat_offline":   "Works offline and optimised for all screen sizes",
        "feat_updates":   "Regular updates with new {p3} improvements",
        "feat_battery":   "Battery-friendly with low memory usage",
    },
    "ar": {
        "intro":          "تم تصميم {brand} للأشخاص الذين يريدون {p1} حقيقيًا بدون تعقيد.",
        "problem":        "معظم التطبيقات مليئة بالإعلانات وبطيئة. {brand} مختلف — صُمِّم ليمنحك {p2} سلسًا و{s1} موثوقًا.",
        "use_case":       "سواء كنت بحاجة إلى {lt1} أو {lt2}، فإن {brand} يلبي احتياجاتك.",
        "features_header":"ما الذي ستحصل عليه",
        "why_header":     "لماذا يختار المستخدمون {brand_upper}",
        "why_body":       "يبقى المستخدمون مع {brand} لأنه يجمع {p3} القوي مع تجربة نظيفة. {s1} سلس و{s2} يتحدث عن نفسه.",
        "perfect_header": "مثالي لـ",
        "perfect_1":      "📱 المستخدمين اليوميين الذين يريدون شيئًا موثوقًا",
        "perfect_2":      "⚡ المستخدمين المتقدمين الذين يحتاجون {p2} سريعًا",
        "perfect_3":      "🎯 كل من يعاني من التطبيقات البطيئة",
        "download_cta":   "حمّل {brand} الآن واكتشف لماذا هو التطبيق المفضل لـ{p1}.",
        "feat_fast":      "سرعة فائقة في {p1} بدون تأخير",
        "feat_smart":     "محرك {p2} ذكي للمستخدمين الحقيقيين",
        "feat_clean":     "واجهة نظيفة وسهلة الاستخدام",
        "feat_offline":   "يعمل بدون إنترنت على جميع الشاشات",
        "feat_updates":   "تحديثات منتظمة مع تحسينات {p3}",
        "feat_battery":   "صديق للبطارية مع استخدام منخفض للذاكرة",
    },
    "de": {
        "intro":          "{brand} wurde für Menschen entwickelt, die echte {p1}-Erfahrungen ohne Ablenkungen wollen.",
        "problem":        "Die meisten Apps in diesem Bereich sind überladen oder voller Werbung. {brand} ist anders — entwickelt für flüssiges {p2} und zuverlässiges {s1}.",
        "use_case":       "Ob {lt1} oder {lt2} — {brand} hat die Lösung.",
        "features_header":"DAS ERWARTET DICH",
        "why_header":     "WARUM NUTZER {brand_upper} WÄHLEN",
        "why_body":       "Nutzer bleiben bei {brand}, weil es starkes {p3} mit einer sauberen Oberfläche kombiniert. {s1} ist nahtlos, {s2} überzeugt von Anfang an.",
        "perfect_header": "PERFEKT FÜR",
        "perfect_1":      "📱 Alltägliche Nutzer, die etwas Zuverlässiges suchen",
        "perfect_2":      "⚡ Power-User, die schnelles {p2} brauchen",
        "perfect_3":      "🎯 Alle, die langsame, werbeüberladene Apps satt haben",
        "download_cta":   "Lade {brand} jetzt herunter und entdecke, warum es die erste Wahl für {p1} ist.",
        "feat_fast":      "Blitzschnelles {p1} ohne Verzögerung",
        "feat_smart":     "Smarte {p2}-Engine für echte Nutzer",
        "feat_clean":     "Klare, intuitive Oberfläche — kein Einarbeiten nötig",
        "feat_offline":   "Funktioniert offline und auf allen Bildschirmgrößen",
        "feat_updates":   "Regelmäßige Updates mit neuen {p3}-Verbesserungen",
        "feat_battery":   "Akkuschonend mit geringem Speicherbedarf",
    },
    "es": {
        "intro":          "{brand} está diseñado para quienes quieren {p1} real sin complicaciones.",
        "problem":        "La mayoría de las apps están llenas de anuncios o simplemente no funcionan bien. {brand} es diferente — diseñado para darte {p2} fluido y {s1} confiable.",
        "use_case":       "Tanto si necesitas {lt1} como {lt2}, {brand} lo tiene todo.",
        "features_header":"LO QUE OBTIENES",
        "why_header":     "POR QUÉ LOS USUARIOS ELIGEN {brand_upper}",
        "why_body":       "Los usuarios prefieren {brand} porque combina {p3} poderoso con una experiencia limpia. El {s1} es perfecto y {s2} habla por sí solo.",
        "perfect_header": "PERFECTO PARA",
        "perfect_1":      "📱 Usuarios cotidianos que quieren algo confiable",
        "perfect_2":      "⚡ Usuarios avanzados que necesitan {p2} rápido",
        "perfect_3":      "🎯 Cualquiera que esté harto de apps lentas con anuncios",
        "download_cta":   "Descarga {brand} hoy y descubre por qué es la app líder de {p1}.",
        "feat_fast":      "{p1} ultrarrápido sin retrasos",
        "feat_smart":     "Motor inteligente de {p2} para usuarios reales",
        "feat_clean":     "Interfaz limpia e intuitiva — sin curva de aprendizaje",
        "feat_offline":   "Funciona sin internet en todos los tamaños de pantalla",
        "feat_updates":   "Actualizaciones frecuentes con mejoras de {p3}",
        "feat_battery":   "Optimizado para la batería con bajo uso de memoria",
    },
    "fr": {
        "intro":          "{brand} est conçu pour ceux qui veulent une vraie expérience de {p1} sans complications.",
        "problem":        "La plupart des apps sont surchargées ou remplies de pubs. {brand} est différent — créé pour vous offrir un {p2} fluide et un {s1} fiable.",
        "use_case":       "Que vous ayez besoin de {lt1} ou de {lt2}, {brand} vous couvre.",
        "features_header":"CE QUE VOUS OBTENEZ",
        "why_header":     "POURQUOI LES UTILISATEURS CHOISISSENT {brand_upper}",
        "why_body":       "Les utilisateurs restent avec {brand} car il allie {p3} puissant à une interface claire. Le {s1} est fluide et {s2} parle de lui-même.",
        "perfect_header": "PARFAIT POUR",
        "perfect_1":      "📱 Les utilisateurs quotidiens qui veulent de la fiabilité",
        "perfect_2":      "⚡ Les utilisateurs avancés qui ont besoin de {p2} rapide",
        "perfect_3":      "🎯 Tous ceux qui en ont assez des apps lentes et remplies de pubs",
        "download_cta":   "Téléchargez {brand} aujourd'hui et découvrez pourquoi c'est l'app {p1} préférée.",
        "feat_fast":      "{p1} ultra-rapide sans latence",
        "feat_smart":     "Moteur {p2} intelligent pour les vrais utilisateurs",
        "feat_clean":     "Interface claire et intuitive — prise en main immédiate",
        "feat_offline":   "Fonctionne hors ligne sur tous les écrans",
        "feat_updates":   "Mises à jour régulières avec des améliorations de {p3}",
        "feat_battery":   "Économe en batterie et en mémoire",
    },
    "hi": {
        "intro":          "{brand} उन लोगों के लिए बना है जो बिना किसी झंझट के असली {p1} चाहते हैं।",
        "problem":        "ज़्यादातर ऐप्स विज्ञापनों से भरे और धीमे होते हैं। {brand} अलग है — यह आपको बेहतरीन {p2} और भरोसेमंद {s1} देने के लिए बनाया गया है।",
        "use_case":       "चाहे आपको {lt1} चाहिए या {lt2}, {brand} सब कुछ संभाल लेता है।",
        "features_header":"आपको क्या मिलेगा",
        "why_header":     "लोग {brand_upper} क्यों चुनते हैं",
        "why_body":       "यूज़र {brand} के साथ रहते हैं क्योंकि यह शक्तिशाली {p3} को साफ अनुभव के साथ जोड़ता है।",
        "perfect_header": "किसके लिए परफेक्ट है",
        "perfect_1":      "📱 रोज़ाना के यूज़र जो भरोसेमंद ऐप चाहते हैं",
        "perfect_2":      "⚡ पावर यूज़र जिन्हें तेज़ {p2} चाहिए",
        "perfect_3":      "🎯 वो लोग जो धीमे विज्ञापन-भरे ऐप्स से थक चुके हैं",
        "download_cta":   "आज ही {brand} डाउनलोड करें और {p1} का सबसे बेहतरीन अनुभव पाएं।",
        "feat_fast":      "{p1} में बिजली जैसी तेज़ी, बिना किसी देरी के",
        "feat_smart":     "असली यूज़र के लिए स्मार्ट {p2} इंजन",
        "feat_clean":     "साफ और सरल इंटरफेस — कोई सीखने की ज़रूरत नहीं",
        "feat_offline":   "बिना इंटरनेट के भी काम करता है",
        "feat_updates":   "नियमित अपडेट के साथ {p3} में सुधार",
        "feat_battery":   "बैटरी-फ्रेंडली, कम मेमोरी उपयोग",
    },
    "pt": {
        "intro":          "{brand} foi criado para quem quer {p1} de verdade, sem complicações.",
        "problem":        "A maioria dos apps é lenta ou cheia de anúncios. {brand} é diferente — feito para oferecer {p2} fluido e {s1} confiável.",
        "use_case":       "Seja para {lt1} ou {lt2}, {brand} tem tudo o que você precisa.",
        "features_header":"O QUE VOCÊ GANHA",
        "why_header":     "POR QUE OS USUÁRIOS ESCOLHEM {brand_upper}",
        "why_body":       "Os usuários ficam com {brand} porque ele combina {p3} poderoso com uma experiência limpa. O {s1} é impecável e {s2} fala por si só.",
        "perfect_header": "PERFEITO PARA",
        "perfect_1":      "📱 Usuários do dia a dia que querem confiabilidade",
        "perfect_2":      "⚡ Usuários avançados que precisam de {p2} rápido",
        "perfect_3":      "🎯 Quem está cansado de apps lentos com anúncios",
        "download_cta":   "Baixe {brand} hoje e descubra por que é o app líder em {p1}.",
        "feat_fast":      "{p1} ultrarrápido sem travamentos",
        "feat_smart":     "Motor {p2} inteligente para usuários reais",
        "feat_clean":     "Interface limpa e intuitiva — sem curva de aprendizado",
        "feat_offline":   "Funciona offline em todos os tamanhos de tela",
        "feat_updates":   "Atualizações frequentes com melhorias de {p3}",
        "feat_battery":   "Otimizado para bateria e baixo uso de memória",
    },
    "ru": {
        "intro":          "{brand} создан для тех, кто хочет настоящего {p1} без лишнего шума.",
        "problem":        "Большинство приложений перегружены рекламой. {brand} иначе — создан для плавного {p2} и надёжного {s1}.",
        "use_case":       "Нужен {lt1} или {lt2}? {brand} справится с любой задачей.",
        "features_header":"ЧТО ВЫ ПОЛУЧИТЕ",
        "why_header":     "ПОЧЕМУ ПОЛЬЗОВАТЕЛИ ВЫБИРАЮТ {brand_upper}",
        "why_body":       "Пользователи остаются с {brand}, потому что он сочетает мощный {p3} с чистым интерфейсом. {s1} работает без сбоев, а {s2} говорит сам за себя.",
        "perfect_header": "ИДЕАЛЬНО ДЛЯ",
        "perfect_1":      "📱 Обычных пользователей, которым нужна надёжность",
        "perfect_2":      "⚡ Продвинутых пользователей, которым нужен быстрый {p2}",
        "perfect_3":      "🎯 Тех, кто устал от медленных приложений с рекламой",
        "download_cta":   "Скачайте {brand} сегодня и убедитесь, почему это лучшее {p1}-приложение.",
        "feat_fast":      "Молниеносный {p1} без задержек",
        "feat_smart":     "Умный движок {p2} для реальных пользователей",
        "feat_clean":     "Чистый интуитивный интерфейс — без лишнего обучения",
        "feat_offline":   "Работает офлайн на всех размерах экрана",
        "feat_updates":   "Регулярные обновления с улучшениями {p3}",
        "feat_battery":   "Экономит батарею и память",
    },
    "ur": {
        "intro":          "{brand} ان لوگوں کے لیے بنایا گیا ہے جو بے فضول پیچیدگی کے بغیر اصلی {p1} چاہتے ہیں۔",
        "problem":        "زیادہ تر ایپس اشتہارات سے بھری اور سست ہیں۔ {brand} مختلف ہے — یہ آپ کو بہترین {p2} اور قابل اعتماد {s1} دینے کے لیے بنایا گیا ہے۔",
        "use_case":       "چاہے آپ کو {lt1} کی ضرورت ہو یا {lt2}، {brand} آپ کے لیے ہے۔",
        "features_header":"آپ کو کیا ملے گا",
        "why_header":     "لوگ {brand_upper} کیوں چنتے ہیں",
        "why_body":       "صارفین {brand} کے ساتھ رہتے ہیں کیونکہ یہ طاقتور {p3} کو صاف تجربے کے ساتھ جوڑتا ہے۔",
        "perfect_header": "کس کے لیے بہترین ہے",
        "perfect_1":      "📱 روزمرہ کے صارفین جو قابل اعتماد ایپ چاہتے ہیں",
        "perfect_2":      "⚡ پاور یوزرز جنہیں تیز {p2} چاہیے",
        "perfect_3":      "🎯 وہ لوگ جو سست، اشتہار بھری ایپس سے تھک چکے ہیں",
        "download_cta":   "آج ہی {brand} ڈاؤن لوڈ کریں اور {p1} کا بہترین تجربہ حاصل کریں۔",
        "feat_fast":      "{p1} میں بجلی کی رفتار، بغیر کسی تاخیر کے",
        "feat_smart":     "اصل صارفین کے لیے سمارٹ {p2} انجن",
        "feat_clean":     "صاف اور آسان انٹرفیس — کوئی سیکھنے کی ضرورت نہیں",
        "feat_offline":   "انٹرنیٹ کے بغیر بھی کام کرتا ہے",
        "feat_updates":   "باقاعدہ اپ ڈیٹس کے ساتھ {p3} میں بہتری",
        "feat_battery":   "بیٹری فرینڈلی، کم میموری استعمال",
    },
    "id": {
        "intro":          "{brand} dibuat untuk orang yang menginginkan {p1} nyata tanpa kerumitan.",
        "problem":        "Sebagian besar aplikasi penuh iklan atau lambat. {brand} berbeda — dirancang untuk memberi Anda {p2} yang lancar dan {s1} yang andal.",
        "use_case":       "Baik untuk {lt1} maupun {lt2}, {brand} siap membantu.",
        "features_header":"APA YANG ANDA DAPATKAN",
        "why_header":     "MENGAPA PENGGUNA MEMILIH {brand_upper}",
        "why_body":       "Pengguna tetap bersama {brand} karena menggabungkan {p3} yang kuat dengan tampilan bersih. {s1} mulus dan {s2} berbicara sendiri.",
        "perfect_header": "SEMPURNA UNTUK",
        "perfect_1":      "📱 Pengguna sehari-hari yang butuh keandalan",
        "perfect_2":      "⚡ Pengguna mahir yang butuh {p2} cepat",
        "perfect_3":      "🎯 Siapa pun yang bosan dengan aplikasi lambat dan penuh iklan",
        "download_cta":   "Unduh {brand} sekarang dan rasakan mengapa ini adalah aplikasi {p1} terbaik.",
        "feat_fast":      "{p1} sangat cepat tanpa lag",
        "feat_smart":     "Mesin {p2} cerdas untuk pengguna nyata",
        "feat_clean":     "Antarmuka bersih dan intuitif",
        "feat_offline":   "Berfungsi offline di semua ukuran layar",
        "feat_updates":   "Pembaruan rutin dengan peningkatan {p3}",
        "feat_battery":   "Hemat baterai dan memori",
    },
    "tr": {
        "intro":          "{brand}, gerçek {p1} deneyimi isteyen kullanıcılar için tasarlandı.",
        "problem":        "Çoğu uygulama reklam dolu ve yavaş. {brand} farklı — size akıcı {p2} ve güvenilir {s1} sunmak için geliştirildi.",
        "use_case":       "{lt1} veya {lt2} için ne gerekiyorsa {brand} hazır.",
        "features_header":"NELER SUNUYORUZ",
        "why_header":     "KULLANICILAR NEDEN {brand_upper} SEÇIYOR",
        "why_body":       "Kullanıcılar {brand}'ı seçiyor çünkü güçlü {p3}'yı temiz bir deneyimle birleştiriyor.",
        "perfect_header": "KİM İÇİN MÜKEMMEL",
        "perfect_1":      "📱 Güvenilir bir şey isteyen günlük kullanıcılar",
        "perfect_2":      "⚡ Hızlı {p2} isteyen güçlü kullanıcılar",
        "perfect_3":      "🎯 Yavaş, reklam dolu uygulamalardan bıkmışlar",
        "download_cta":   "{brand}'ı bugün indirin ve {p1} için neden tercih edilen uygulama olduğunu keşfedin.",
        "feat_fast":      "Gecikme olmadan {p1}'da şimşek hızı",
        "feat_smart":     "Gerçek kullanıcılar için akıllı {p2} motoru",
        "feat_clean":     "Temiz, sezgisel arayüz",
        "feat_offline":   "Tüm ekran boyutlarında çevrimdışı çalışır",
        "feat_updates":   "{p3} iyileştirmeleriyle düzenli güncellemeler",
        "feat_battery":   "Pil dostu, düşük bellek kullanımı",
    },
    "zh": {
        "intro":          "{brand} 专为追求真正{p1}体验的用户而生，简洁高效。",
        "problem":        "市面上大多数应用广告满屏、运行迟缓。{brand} 不同——专为流畅的{p2}和可靠的{s1}而设计。",
        "use_case":       "无论您需要{lt1}还是{lt2}，{brand} 都能满足。",
        "features_header":"您将获得什么",
        "why_header":     "为什么用户选择 {brand_upper}",
        "why_body":       "用户选择留在 {brand}，因为它将强大的{p3}与简洁的体验完美结合。{s1}流畅无阻，{s2}不言而喻。",
        "perfect_header": "适合人群",
        "perfect_1":      "📱 需要可靠工具的日常用户",
        "perfect_2":      "⚡ 需要快速{p2}的进阶用户",
        "perfect_3":      "🎯 厌倦了广告泛滥、速度缓慢应用的所有人",
        "download_cta":   "立即下载 {brand}，发现为何它是{p1}领域的首选应用。",
        "feat_fast":      "{p1}极速，零延迟",
        "feat_smart":     "为真实用户打造的智能{p2}引擎",
        "feat_clean":     "界面简洁直观，无需学习",
        "feat_offline":   "离线可用，适配所有屏幕尺寸",
        "feat_updates":   "定期更新，持续优化{p3}",
        "feat_battery":   "省电省内存",
    },
    "ja": {
        "intro":          "{brand}は、複雑な操作なしに本格的な{p1}を求めるユーザーのために作られました。",
        "problem":        "多くのアプリは広告だらけで動作が遅いです。{brand}は違います——スムーズな{p2}と信頼できる{s1}を提供するために設計されました。",
        "use_case":       "{lt1}でも{lt2}でも、{brand}がサポートします。",
        "features_header":"特徴・機能",
        "why_header":     "なぜユーザーは{brand_upper}を選ぶのか",
        "why_body":       "ユーザーが{brand}を選ぶ理由は、強力な{p3}とクリーンなUIを両立しているからです。",
        "perfect_header": "こんな方におすすめ",
        "perfect_1":      "📱 信頼性を求める日常ユーザー",
        "perfect_2":      "⚡ 高速な{p2}を必要とするパワーユーザー",
        "perfect_3":      "🎯 遅くて広告だらけのアプリに嫌気がさした方",
        "download_cta":   "今すぐ{brand}をダウンロードして、{p1}の最高体験を。",
        "feat_fast":      "ラグゼロの超高速{p1}",
        "feat_smart":     "実ユーザー向けスマート{p2}エンジン",
        "feat_clean":     "直感的でシンプルなUI",
        "feat_offline":   "オフラインでも動作、全画面サイズ対応",
        "feat_updates":   "{p3}の改善を含む定期アップデート",
        "feat_battery":   "バッテリーとメモリに優しい設計",
    },
    "ko": {
        "intro":          "{brand}은 복잡함 없이 진정한 {p1} 경험을 원하는 사람들을 위해 만들어졌습니다.",
        "problem":        "대부분의 앱은 광고로 가득 차 있거나 느립니다. {brand}는 다릅니다 — 원활한 {p2}와 신뢰할 수 있는 {s1}을 제공하도록 설계되었습니다.",
        "use_case":       "{lt1}이든 {lt2}이든 {brand}가 해결해 드립니다.",
        "features_header":"제공 기능",
        "why_header":     "사용자들이 {brand_upper}를 선택하는 이유",
        "why_body":       "사용자들이 {brand}에 머무는 이유는 강력한 {p3}와 깔끔한 UX를 결합했기 때문입니다.",
        "perfect_header": "이런 분께 최적",
        "perfect_1":      "📱 신뢰할 수 있는 앱을 원하는 일반 사용자",
        "perfect_2":      "⚡ 빠른 {p2}가 필요한 파워 유저",
        "perfect_3":      "🎯 느리고 광고 가득한 앱에 지친 모든 분",
        "download_cta":   "지금 {brand}를 다운로드하고 최고의 {p1} 앱을 경험해 보세요.",
        "feat_fast":      "지연 없는 초고속 {p1}",
        "feat_smart":     "실제 사용자를 위한 스마트 {p2} 엔진",
        "feat_clean":     "직관적이고 깔끔한 인터페이스",
        "feat_offline":   "오프라인에서도 작동, 모든 화면 크기 지원",
        "feat_updates":   "{p3} 개선 포함 정기 업데이트",
        "feat_battery":   "배터리 및 메모리 절약",
    },
}

# Fallback: use English for any lang not in LANG_TEMPLATES
def get_template(lang_code):
    return LANG_TEMPLATES.get(lang_code, LANG_TEMPLATES["en"])

# ============================================================
# HELPERS
# ============================================================
def extract_app_id(url):
    return parse_qs(urlparse(url).query).get("id", [None])[0]

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def tokenize(text, extra_stopwords=None):
    if not text:
        return []
    sw = STOP_WORDS.copy()
    if extra_stopwords:
        sw.update(extra_stopwords)
    try:
        return [t for t in word_tokenize(clean_text(text)) if t not in sw and len(t) > 2]
    except:
        return clean_text(text).split()

def extract_seed_keywords(text):
    tokens = tokenize(text)
    important = [t for t in tokens if len(t) > 3]
    phrases = []
    for i in range(len(important) - 1):
        phrases.append(f"{important[i]} {important[i+1]}")
    return phrases[:5]

# ============================================================
# SCRAPER
# ============================================================
def get_app_data(app_id, country, lang="en"):
    try:
        d = gp_app(app_id, lang=lang, country=country)
        return {
            "title":      d.get("title", "") or "",
            "short_desc": d.get("summary", "") or "",
            "desc":       d.get("description", "") or "",
            "rating":     float(d.get("score") or 0),
            "reviews":    int(d.get("reviews") or 0),
            "installs":   d.get("installs", "") or "",
            "country":    country,
            "lang":       lang,
        }
    except:
        return None

def get_app_by_title(title, country, lang="en"):
    try:
        res = gp_search(title, lang=lang, country=country, n_hits=1)
        if res:
            return get_app_data(res[0]["appId"], country, lang)
    except:
        pass
    return None

def get_competitors(seed_keywords, country, lang="en"):
    ids = []
    for kw in seed_keywords:
        try:
            res = gp_search(kw, lang=lang, country=country, n_hits=6)
            ids.extend([r["appId"] for r in res])
        except:
            continue
    return list(dict.fromkeys(ids))[:8]

# ============================================================
# KEYWORD EXTRACTION
# ============================================================
def extract_keywords(apps, lang="en"):
    docs = [
        (a.get("title") or "") + " " +
        (a.get("short_desc") or "") + " " +
        (a.get("desc") or "")
        for a in apps
    ]
    if not docs or all(d.strip() == "" for d in docs):
        return pd.DataFrame()
    try:
        sw = "english" if lang == "en" else None
        vectorizer = CountVectorizer(
            stop_words=sw,
            ngram_range=(1, 3),
            max_features=120,
            min_df=1,
        )
        X = vectorizer.fit_transform(docs)
        freqs = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        data = []
        for term, freq in zip(terms, freqs):
            difficulty  = math.log2(freq + 10)
            opportunity = freq / (difficulty + 1)
            data.append({"keyword": term, "difficulty": round(difficulty, 2), "opportunity": round(opportunity, 2)})
        return pd.DataFrame(sorted(data, key=lambda x: x["opportunity"], reverse=True))
    except:
        return pd.DataFrame()

# ============================================================
# ASO COPY GENERATORS  (language-aware)
# ============================================================
def generate_title(app_name, primary_keywords):
    brand       = app_name.split(":")[0].split("-")[0].strip()
    short_brand = brand[:14].strip()
    for kw in primary_keywords:
        kw_t = kw.title()
        cand = f"{short_brand}: {kw_t}"
        if len(cand) <= 30:
            return cand
        if len(kw_t) <= 30:
            return kw_t
    return brand[:30]

def generate_short(primary, secondary, app_name, lang="en"):
    brand = app_name.split(":")[0].split("-")[0].strip()
    t = get_template(lang)
    # Build a short line from the template snippets, fall back to English pattern
    p1 = primary[0]   if primary   else "features"
    s1 = secondary[0] if secondary else "tools"
    candidates = [
        f"{p1.title()} & {s1} — {t.get('feat_fast','fast & free').split('{')[0].strip()}.",
        f"{p1.title()} · {s1} · {brand}",
        f"{brand}: {p1.title()}",
    ]
    for c in candidates:
        if len(c) <= 80:
            return c
    return f"{brand}: {p1}"[:80]

def generate_long_desc(app_name, primary, secondary, long_tail, core_features, lang="en"):
    p1  = primary[0]   if len(primary)   > 0 else "features"
    p2  = primary[1]   if len(primary)   > 1 else "performance"
    p3  = primary[2]   if len(primary)   > 2 else "tools"
    s1  = secondary[0] if len(secondary) > 0 else "ease of use"
    s2  = secondary[1] if len(secondary) > 1 else "speed"
    lt1 = long_tail[0] if len(long_tail) > 0 else p1
    lt2 = long_tail[1] if len(long_tail) > 1 else p2
    brand       = app_name.split(":")[0].split("-")[0].strip()
    brand_upper = brand.upper()

    t = get_template(lang)

    def fmt(s):
        return s.format(brand=brand, brand_upper=brand_upper,
                        p1=p1, p2=p2, p3=p3, s1=s1, s2=s2, lt1=lt1, lt2=lt2)

    if core_features and core_features.strip():
        feature_lines   = [f.strip() for f in core_features.strip().splitlines() if f.strip()]
        feature_bullets = "\n".join([f"✅ {f}" for f in feature_lines[:8]])
    else:
        feature_bullets = "\n".join([
            f"✅ {fmt(t['feat_fast'])}",
            f"✅ {fmt(t['feat_smart'])}",
            f"✅ {fmt(t['feat_clean'])}",
            f"✅ {fmt(t['feat_offline'])}",
            f"✅ {fmt(t['feat_updates'])}",
            f"✅ {fmt(t['feat_battery'])}",
        ])

    use_case_line = fmt(t["use_case"]) + "\n\n" if lt1 != p1 else ""

    desc = (
        f"{fmt(t['intro'])}\n\n"
        f"{fmt(t['problem'])}\n\n"
        f"{use_case_line}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{t['features_header']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{feature_bullets}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{fmt(t['why_header'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{fmt(t['why_body'])}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{t['perfect_header']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{fmt(t['perfect_1'])}\n"
        f"{fmt(t['perfect_2'])}\n"
        f"{fmt(t['perfect_3'])}\n\n"
        f"{fmt(t['download_cta'])}"
    )
    return desc.strip()[:4000]

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="ASO Intelligence Dashboard", layout="wide")
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 ASO Intelligence Dashboard")
st.caption("Keyword research & store listing optimizer for Google Play — 100% free, no API key required")

# ============================================================
# INPUT FORM
# ============================================================

# ── Tier selector lives OUTSIDE the form so it reacts instantly ──
tier_options = list(ALL_TIERS.keys())

def _on_tier_change():
    chosen = st.session_state["tier_selector"]
    st.session_state["country_multiselect"] = [c for c, _ in ALL_TIERS[chosen]]

# Initialise session defaults on first load
if "tier_selector" not in st.session_state:
    st.session_state["tier_selector"] = tier_options[0]
if "country_multiselect" not in st.session_state:
    st.session_state["country_multiselect"] = [c for c, _ in ALL_TIERS[tier_options[0]]]

st.selectbox(
    "Quick-select countries by Tier",
    options=tier_options,
    key="tier_selector",
    on_change=_on_tier_change,
    help=(
        "Tier 1: High-income Western/East-Asian markets (best monetisation)\n"
        "Tier 2: Large fast-growing emerging markets\n"
        "Tier 3: Developing markets — high volume, lower ARPU\n"
        "Tier 4: Frontier markets — niche but growing"
    )
)

st.multiselect(
    "Countries to analyze (auto-filled by tier — edit freely)",
    options=ALL_COUNTRY_CODES,
    key="country_multiselect",
    format_func=lambda c: COUNTRY_LABEL_MAP.get(c, c.upper()),
    help="Changing the tier above replaces this list. You can still add/remove individual countries."
)

with st.form("aso_form"):
    input_type = st.radio("Input type", ["Play Store URL", "App Title / Name"], horizontal=True)
    app_input  = st.text_input(
        "Play Store URL" if input_type == "Play Store URL" else "App Title or Name",
        placeholder=(
            "https://play.google.com/store/apps/details?id=com.example.app"
            if input_type == "Play Store URL"
            else "e.g. Volume Booster Increase Sound"
        )
    )
    core_features = st.text_area(
        "Core Features of Your App (one per line — optional but recommended)",
        placeholder="e.g.\nBoost volume up to 200%\nEqualizer with bass & treble control\nWorks without internet",
        height=120
    )
    submitted = st.form_submit_button("🔍 Analyze")

# ============================================================
# ANALYSIS
# ============================================================
if submitted:
    if not app_input.strip():
        st.error("Please enter a Play Store URL or app title.")
        st.stop()

    country_list = st.session_state.get("country_multiselect") or ["us"]

    progress = st.progress(0)
    status   = st.empty()

    status.info("🔍 Fetching your app data...")
    progress.progress(10)

    if input_type == "Play Store URL":
        app_id = extract_app_id(app_input)
        if not app_id:
            st.error("Invalid Play Store URL. Could not extract app ID.")
            st.stop()
        own = get_app_data(app_id, country_list[0])
    else:
        own = get_app_by_title(app_input.strip(), country_list[0])

    if not own or not own.get("title"):
        st.error("Failed to fetch app data. Check the URL/title or try a different country.")
        st.stop()

    progress.progress(20)
    status.info("🌱 Extracting seed keywords...")

    seed = extract_seed_keywords((own.get("title") or "") + " " + (own.get("desc") or ""))
    if not seed:
        seed = [own.get("title", "")]

    all_apps_by_country = {c: [] for c in country_list}
    all_apps = [own]
    total    = len(country_list)

    for idx, c in enumerate(country_list):
        status.info(f"🌍 Scraping competitors — **{COUNTRY_LABEL_MAP.get(c, c.upper())}** ({idx+1}/{total})...")
        ids = get_competitors(seed, c)
        for aid in ids:
            data = get_app_data(aid, c)
            if data:
                all_apps_by_country[c].append(data)
                all_apps.append(data)
            time.sleep(0.15)
        pct = 20 + int(55 * (idx + 1) / total)
        progress.progress(pct)

    if len(all_apps) < 2:
        st.error("Not enough competitor data. Try adding more countries.")
        st.stop()

    progress.progress(78)
    status.info("🧠 Extracting keywords & building ASO copy...")

    kw_df    = extract_keywords(all_apps)
    if kw_df.empty:
        st.error("Keyword extraction failed.")
        st.stop()

    keywords  = kw_df["keyword"].tolist()
    if len(keywords) < 3:
        st.error("Not enough keywords extracted.")
        st.stop()

    primary   = keywords[:3]
    secondary = keywords[3:6]
    long_tail = keywords[6:15] if len(keywords) > 6 else []
    app_name  = own.get("title", "My App")

    en_title  = generate_title(app_name, primary)
    en_short  = generate_short(primary, secondary, app_name, "en")
    en_long   = generate_long_desc(app_name, primary, secondary, long_tail, core_features, "en")

    progress.progress(100)
    status.success("✅ Analysis complete!")
    time.sleep(0.4)
    status.empty()
    progress.empty()

    # Store in session for localization tab
    st.session_state["aso_done"]        = True
    st.session_state["all_apps"]        = all_apps
    st.session_state["all_apps_by_c"]   = all_apps_by_country
    st.session_state["country_list"]    = country_list
    st.session_state["kw_df"]           = kw_df
    st.session_state["primary"]         = primary
    st.session_state["secondary"]       = secondary
    st.session_state["long_tail"]       = long_tail
    st.session_state["app_name"]        = app_name
    st.session_state["own"]             = own
    st.session_state["core_features"]   = core_features
    st.session_state["en_title"]        = en_title
    st.session_state["en_short"]        = en_short
    st.session_state["en_long"]         = en_long
    st.session_state["seed"]            = seed

# ============================================================
# RESULTS (shown if analysis done in this or previous run)
# ============================================================
if st.session_state.get("aso_done"):

    all_apps           = st.session_state["all_apps"]
    all_apps_by_c      = st.session_state["all_apps_by_c"]
    country_list       = st.session_state["country_list"]
    kw_df              = st.session_state["kw_df"]
    primary            = st.session_state["primary"]
    secondary          = st.session_state["secondary"]
    long_tail          = st.session_state["long_tail"]
    app_name           = st.session_state["app_name"]
    own                = st.session_state["own"]
    core_features      = st.session_state["core_features"]
    en_title           = st.session_state["en_title"]
    en_short           = st.session_state["en_short"]
    en_long            = st.session_state["en_long"]
    seed               = st.session_state["seed"]

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔑 Keywords", "✍️ ASO Copy", "🌐 Localization"])

    # ── TAB 1: Overview ──────────────────────────────────────
    with tab1:
        st.subheader("Your App")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Title",    own.get("title", "—"))
        c2.metric("Rating",   f"{own.get('rating', 0):.1f} ⭐")
        c3.metric("Reviews",  f"{own.get('reviews', 0):,}")
        c4.metric("Installs", own.get("installs", "—"))
        if own.get("short_desc"):
            st.caption(f"**Short description:** {own.get('short_desc')}")
        st.divider()

        tier_filter = st.selectbox(
            "Filter by Tier",
            ["All Tiers"] + [k for k in ALL_TIERS.keys() if k != "🌍 All Countries"],
            key="ov_tier"
        )
        tier_filter_codes = (
            ALL_COUNTRY_CODES if tier_filter == "All Tiers"
            else [c for c,_ in ALL_TIERS[tier_filter]]
        )
        sel_country = st.selectbox(
            "Filter by Country",
            ["All"] + [c for c in country_list if c in tier_filter_codes],
            key="ov_country"
        )
        st.subheader("Competitor Apps")
        rows = []
        for c, apps_in_c in all_apps_by_c.items():
            if c not in tier_filter_codes:
                continue
            if sel_country != "All" and c != sel_country:
                continue
            for a in apps_in_c:
                rows.append({
                    "Country":           c.upper(),
                    "Title":             a.get("title", ""),
                    "Short Description": a.get("short_desc", ""),
                    "Rating":            round(float(a.get("rating") or 0), 1),
                    "Reviews":           a.get("reviews", 0),
                    "Installs":          a.get("installs", ""),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No competitor data for the selected filter.")

    # ── TAB 2: Keywords ───────────────────────────────────────
    with tab2:
        st.subheader("Keyword Opportunities")
        tier_kw = st.selectbox(
            "Filter by Tier",
            ["All Tiers"] + [k for k in ALL_TIERS.keys() if k != "🌍 All Countries"],
            key="kw_tier"
        )
        tier_kw_codes = (
            ALL_COUNTRY_CODES if tier_kw == "All Tiers"
            else [c for c,_ in ALL_TIERS[tier_kw]]
        )
        kw_country = st.selectbox(
            "Filter by Country",
            ["All"] + [c for c in country_list if c in tier_kw_codes],
            key="kw_country"
        )
        if kw_country == "All":
            display_apps = [a for a in all_apps if a.get("country") in tier_kw_codes]
        else:
            display_apps = [a for a in all_apps if a.get("country") == kw_country]
        if own not in display_apps:
            display_apps = [own] + display_apps

        fkw_df = extract_keywords(display_apps)
        if not fkw_df.empty:
            st.dataframe(fkw_df, use_container_width=True)
            st.bar_chart(fkw_df.set_index("keyword")["opportunity"].head(15))
        else:
            st.info("Not enough data for this filter.")

    # ── TAB 3: ASO Copy ───────────────────────────────────────
    with tab3:
        st.subheader("Generated ASO Listing (English)")

        st.markdown("#### 📌 Title")
        st.caption("Google Play limit: **30 characters**")
        tc, cc = st.columns([5, 1])
        with tc:
            t_edit = st.text_input("Title", value=en_title, max_chars=30, label_visibility="collapsed")
        with cc:
            st.markdown(f"{'🟢' if len(t_edit)<=30 else '🔴'} `{len(t_edit)}/30`")

        st.divider()
        st.markdown("#### 📝 Short Description")
        st.caption("Google Play limit: **80 characters**")
        sc, scc = st.columns([5, 1])
        with sc:
            s_edit = st.text_area("Short", value=en_short, max_chars=80, height=80, label_visibility="collapsed")
        with scc:
            st.markdown(f"{'🟢' if len(s_edit)<=80 else '🔴'} `{len(s_edit)}/80`")

        st.divider()
        st.markdown("#### 📄 Long Description")
        st.caption("Google Play limit: **4,000 characters**")
        lc, lcc = st.columns([5, 1])
        with lc:
            l_edit = st.text_area("Long", value=en_long, max_chars=4000, height=520, label_visibility="collapsed")
        with lcc:
            st.markdown(f"{'🟢' if len(l_edit)<=4000 else '🔴'} `{len(l_edit)}/4000`")

        st.divider()
        st.markdown("#### 🔑 Top Keywords Embedded")
        all_used = primary + secondary + (long_tail[:5] if long_tail else [])
        kc = st.columns(3)
        for i, kw in enumerate(all_used):
            kc[i % 3].markdown(f"- `{kw}`")

    # ── TAB 4: Localization ────────────────────────────────────
    with tab4:
        st.subheader("🌐 Localized ASO Listing")
        st.caption(
            "Select a language. The tool will scrape Play Store listings in that language "
            "from the most relevant countries, extract native keywords, then generate a "
            "fully localized Title, Short Description, and Long Description — no API needed."
        )

        locale_label = st.selectbox(
            "Select Target Language",
            options=list(LOCALES.keys()),
            index=0,
        )
        locale       = LOCALES[locale_label]
        lang_code    = locale["code"]
        lang_name    = locale["name"]

        # Pick best countries for this language (use first 5 from selected list, fallback to us)
        locale_countries = [c for c in country_list] or ["us"]
        locale_countries = locale_countries[:6]   # limit for speed

        if st.button(f"🚀 Generate {lang_name} ASO"):
            with st.spinner(f"Scraping Play Store in {lang_name}..."):
                loc_apps = []
                for c in locale_countries:
                    ids = get_competitors(seed, c, lang=lang_code)
                    for aid in ids:
                        d = get_app_data(aid, c, lang=lang_code)
                        if d:
                            loc_apps.append(d)
                        time.sleep(0.15)

                # Also include own app description for seed vocabulary
                if own:
                    loc_apps.insert(0, own)

            if len(loc_apps) < 2:
                st.warning("Not enough localized data found. Generating from English keywords with translated templates.")
                loc_primary   = primary
                loc_secondary = secondary
                loc_long_tail = long_tail
                loc_kw_df     = kw_df.copy()
            else:
                loc_kw_df = extract_keywords(loc_apps, lang=lang_code)
                if loc_kw_df.empty:
                    loc_primary   = primary
                    loc_secondary = secondary
                    loc_long_tail = long_tail
                else:
                    loc_kws       = loc_kw_df["keyword"].tolist()
                    loc_primary   = loc_kws[:3]
                    loc_secondary = loc_kws[3:6]
                    loc_long_tail = loc_kws[6:15] if len(loc_kws) > 6 else []

            loc_title = generate_title(app_name, loc_primary)
            loc_short = generate_short(loc_primary, loc_secondary, app_name, lang_code)
            loc_long  = generate_long_desc(app_name, loc_primary, loc_secondary, loc_long_tail, core_features, lang_code)

            st.success(f"✅ {lang_name} ASO listing generated!")

            st.markdown(f"#### 📌 Title — {lang_name}")
            ltc, lcc2 = st.columns([5, 1])
            with ltc:
                lt_edit = st.text_input(f"loc_title_{lang_code}", value=loc_title, max_chars=30, label_visibility="collapsed")
            with lcc2:
                st.markdown(f"{'🟢' if len(lt_edit)<=30 else '🔴'} `{len(lt_edit)}/30`")

            st.divider()
            st.markdown(f"#### 📝 Short Description — {lang_name}")
            lsc, lscc = st.columns([5, 1])
            with lsc:
                ls_edit = st.text_area(f"loc_short_{lang_code}", value=loc_short, max_chars=80, height=80, label_visibility="collapsed")
            with lscc:
                st.markdown(f"{'🟢' if len(ls_edit)<=80 else '🔴'} `{len(ls_edit)}/80`")

            st.divider()
            st.markdown(f"#### 📄 Long Description — {lang_name}")
            llc, llcc = st.columns([5, 1])
            with llc:
                ll_edit = st.text_area(f"loc_long_{lang_code}", value=loc_long, max_chars=4000, height=520, label_visibility="collapsed")
            with llcc:
                st.markdown(f"{'🟢' if len(ll_edit)<=4000 else '🔴'} `{len(ll_edit)}/4000`")

            if not loc_kw_df.empty:
                st.divider()
                st.markdown(f"#### 🔑 Top Keywords in {lang_name}")
                st.dataframe(loc_kw_df.head(20), use_container_width=True)
                st.bar_chart(loc_kw_df.set_index("keyword")["opportunity"].head(15))
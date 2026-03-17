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

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ---------- HELPERS ----------
def extract_app_id(url):
    return parse_qs(urlparse(url).query).get("id", [None])[0]

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def tokenize(text):
    if not text:
        return []
    try:
        return [t for t in word_tokenize(clean_text(text)) if t not in STOP_WORDS and len(t) > 2]
    except:
        return []

# ---------- SEED ----------
def extract_seed_keywords(text):
    tokens = tokenize(text)
    important = [t for t in tokens if len(t) > 3]

    phrases = []
    for i in range(len(important) - 1):
        phrases.append(f"{important[i]} {important[i+1]}")

    return phrases[:5]

# ---------- SCRAPER ----------
def get_app_data(app_id, country):
    try:
        d = gp_app(app_id, lang="en", country=country)
        return {
            "title": d.get("title", ""),
            "desc": d.get("description", ""),
            "rating": d.get("score", 0),
            "reviews": d.get("reviews", 0)
        }
    except:
        return None

def get_competitors(seed_keywords, country):
    ids = []
    for kw in seed_keywords:
        try:
            res = gp_search(kw, lang="en", country=country, n_hits=6)
            ids.extend([r["appId"] for r in res])
        except:
            continue
    return list(dict.fromkeys(ids))[:8]

# ---------- KEYWORDS ----------
def extract_keywords(apps):
    docs = [(a.get("title") or "") + " " + (a.get("desc") or "") for a in apps]

    if not docs:
        return pd.DataFrame()

    try:
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(2, 3), max_features=100)
        X = vectorizer.fit_transform(docs)

        freqs = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        data = []

        for term, freq in zip(terms, freqs):
            difficulty = math.log2(freq + 10)
            opportunity = freq / (difficulty + 1)

            data.append({
                "keyword": term,
                "difficulty": round(difficulty, 2),
                "opportunity": round(opportunity, 2)
            })

        return pd.DataFrame(sorted(data, key=lambda x: x["opportunity"], reverse=True))
    except:
        return pd.DataFrame()

# ---------- TITLE ----------
def generate_title(app_name, primary):
    base = app_name.split(" ")[0] if app_name else "App"

    for kw in primary:
        candidate = f"{base}: {kw.title()}"
        if len(candidate) <= 30:
            return candidate

    return base[:30]

# ---------- SHORT ----------
def generate_short(primary, secondary):
    if not primary:
        return "Best app experience for your needs"

    if len(secondary) < 2:
        return primary[0]

    return f"Discover {primary[0]}, {secondary[0]} & {secondary[1]} easily!"[:80]

# ---------- LONG ----------
def generate_long_desc(app_name, primary, secondary, long_tail):
    p = primary[0] if primary else "features"
    s1 = secondary[0] if len(secondary) > 0 else "tools"
    s2 = secondary[1] if len(secondary) > 1 else "experience"

    sections = []

    sections.append(
        f"Tired of poor experience or limited features?\n\n"
        f"{app_name} helps you explore {p}, improve {s1}, "
        f"and get the best out of {s2} effortlessly."
    )

    sections.append(
        "Designed for modern users, this app delivers powerful performance "
        "and smooth experience every time."
    )

    sections.append("━━━━━━━━━━━━━━━━━━━━\nKEY FEATURES\n━━━━━━━━━━━━━━━━━━━━")

    features = [
        "Smart interface",
        "Fast performance",
        "Advanced tools",
        "Multiple use cases",
        "Optimized for all devices",
        "Regular updates"
    ]

    for f in features:
        sections.append(f"✅ {f}")

    sections.append("━━━━━━━━━━━━━━━━━━━━\nPERFECT FOR\n━━━━━━━━━━━━━━━━━━━━")
    sections.append("📱 Mobile users\n⚡ Fast workflows\n🎯 Efficient results")

    for i in range(3):
        sections.append(
            f"\nExplore {p}, improve {s1}, and enhance {s2} with this powerful app."
        )

    sections.append(f"\nDownload {app_name} now and experience next-level performance!")

    sections.append("Tags: " + ", ".join(primary + secondary + long_tail))

    return "\n\n".join(sections)[:4000]

# ---------- UI ----------
st.set_page_config(page_title="ASO Tool", layout="wide")
st.title("🚀 ASO Intelligence Dashboard")

url = st.text_input("Enter Play Store URL")
countries = st.text_input("Countries (us,in,pk)", "us,in")

if st.button("Analyze"):

    app_id = extract_app_id(url)
    if not app_id:
        st.error("Invalid Play Store URL")
        st.stop()

    country_list = [c.strip() for c in countries.split(",")]

    own = get_app_data(app_id, country_list[0])

    if not own or not own.get("title"):
        st.error("Failed to fetch app. Check URL or try again.")
        st.stop()

    desc = own.get("desc") or ""
    title_text = own.get("title") or ""

    seed = extract_seed_keywords(title_text + " " + desc)
    if not seed:
        seed = [title_text]

    apps = [own]

    for c in country_list:
        ids = get_competitors(seed, c)
        for aid in ids:
            data = get_app_data(aid, c)
            if data:
                apps.append(data)
            time.sleep(0.2)

    if len(apps) < 2:
        st.error("Not enough competitor data. Try again.")
        st.stop()

    df = pd.DataFrame(apps)
    kw_df = extract_keywords(apps)

    if kw_df.empty:
        st.error("Keyword extraction failed.")
        st.stop()

    keywords = kw_df["keyword"].tolist()

    if len(keywords) < 3:
        st.error("Not enough keywords.")
        st.stop()

    primary = keywords[:3]
    secondary = keywords[3:6]
    long_tail = keywords[6:15] if len(keywords) > 6 else []

    app_name = own.get("title", "My App")

    title = generate_title(app_name, primary)
    short = generate_short(primary, secondary)
    long = generate_long_desc(app_name, primary, secondary, long_tail)

    tab1, tab2, tab3 = st.tabs(["Overview", "Keywords", "ASO"])

    with tab1:
        st.dataframe(df)

    with tab2:
        st.dataframe(kw_df)
        st.bar_chart(kw_df.set_index("keyword")["opportunity"].head(10))

    with tab3:
        st.text_area("Title", title)
        st.text_area("Short", short)
        st.text_area("Long", long, height=500)
import streamlit as st
import pandas as pd
import time
import math
import re
import string
from urllib.parse import urlparse, parse_qs

from google_play_scraper import search as gp_search, app as gp_app
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# ---------- HELPERS ----------
def extract_app_id(url):
    return parse_qs(urlparse(url).query).get("id", [None])[0]

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def tokenize(text):
    return [t for t in word_tokenize(clean_text(text)) if t not in STOP_WORDS and len(t) > 2]

# ---------- DYNAMIC SEED ----------
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
            res = gp_search(kw, lang="en", country=country, n_hits=8)
            ids.extend([r["appId"] for r in res])
        except:
            continue
    return list(dict.fromkeys(ids))[:8]

# ---------- KEYWORDS ----------
def extract_keywords(apps):
    docs = [a["title"] + " " + a["desc"] for a in apps]

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(2, 3), max_features=120)
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

# ---------- TITLE ----------
def generate_title(app_name, primary):
    base = app_name.split(" ")[0]

    for kw in primary:
        candidate = f"{base}: {kw.title()}"
        if len(candidate) <= 30:
            return candidate

    return app_name[:30]

# ---------- SHORT DESC ----------
def generate_short(primary, secondary):
    if len(secondary) < 2:
        return primary[0]

    return f"Discover {primary[0]}, {secondary[0]} & {secondary[1]} easily!"[:80]

# ---------- LONG DESC ----------
def generate_long_desc(app_name, primary, secondary, long_tail):
    sections = []

    sections.append(
        f"Tired of poor experience or limited features?\n\n"
        f"{app_name} helps you explore {primary[0]}, improve {secondary[0]}, "
        f"and get the best out of {secondary[1]} effortlessly."
    )

    sections.append(
        f"Designed for modern users, this app delivers powerful performance, "
        f"smooth experience, and high-quality results every time."
    )

    sections.append("━━━━━━━━━━━━━━━━━━━━\nKEY FEATURES\n━━━━━━━━━━━━━━━━━━━━")

    features = [
        "Smart and intuitive interface",
        "Fast and reliable performance",
        "Advanced tools and customization",
        "Works across multiple use cases",
        "Optimized for all devices",
        "Regular updates and improvements"
    ]

    for f in features:
        sections.append(f"✅ {f}")

    sections.append("━━━━━━━━━━━━━━━━━━━━\nPERFECT FOR\n━━━━━━━━━━━━━━━━━━━━")
    sections.append(
        "👤 Everyday users\n📱 Mobile users\n⚡ Fast workflows\n🎯 Efficient results"
    )

    # SEO Boost
    for i in range(4):
        sections.append(
            f"\nExplore {primary[0]}, improve {secondary[0]}, and enhance {secondary[1]} "
            f"with this powerful app designed for performance and ease."
        )

    sections.append(
        f"\nDownload {app_name} now and experience next-level performance!"
    )

    sections.append("Tags: " + ", ".join(primary + secondary + long_tail))

    return "\n\n".join(sections)[:4000]

# ---------- UI ----------
st.set_page_config(page_title="ASO SaaS Tool", layout="wide")
st.title("🚀 ASO Intelligence Dashboard (Dynamic AI)")

url = st.text_input("Enter Play Store URL")
countries = st.text_input("Countries (us,in,pk)", "us,in")

if st.button("Analyze"):

    app_id = extract_app_id(url)
    country_list = [c.strip() for c in countries.split(",")]

    with st.spinner("Fetching app..."):
        own = get_app_data(app_id, country_list[0])

    if not own:
        st.error("Failed to fetch app")
        st.stop()

    # 🔥 Dynamic seed
    seed = extract_seed_keywords(own["title"] + " " + own["desc"])

    apps = [own]

    with st.spinner("Scraping competitors..."):
        for c in country_list:
            ids = get_competitors(seed, c)
            for aid in ids:
                data = get_app_data(aid, c)
                if data:
                    apps.append(data)
                time.sleep(0.3)

    df = pd.DataFrame(apps)
    kw_df = extract_keywords(apps)

    # -------- TABS -------- #
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Keywords", "🧠 ASO Output"])

    # -------- OVERVIEW -------- #
    with tab1:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "competitors.csv")

    # -------- KEYWORDS -------- #
    with tab2:
        st.dataframe(kw_df)

        st.bar_chart(kw_df.set_index("keyword")["opportunity"].head(10))
        st.bar_chart(kw_df.set_index("keyword")["difficulty"].head(10))

        st.download_button("Download Keywords", kw_df.to_csv(index=False), "keywords.csv")

    # -------- ASO -------- #
    with tab3:
        keywords = kw_df["keyword"].tolist()

        primary = keywords[:3]
        secondary = keywords[3:6]
        long_tail = keywords[6:15]

        title = generate_title(own["title"], primary)
        short = generate_short(primary, secondary)
        long = generate_long_desc(own["title"], primary, secondary, long_tail)

        st.text_area("Title", title)
        st.text_area("Short Description", short)
        st.text_area("Long Description", long, height=500)

        st.download_button("Download ASO", long, "aso.txt")
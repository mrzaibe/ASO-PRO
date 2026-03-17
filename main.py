import re
import time
import string
import math
import logging
from collections import defaultdict
from urllib.parse import urlparse, parse_qs

import pandas as pd
from google_play_scraper import search as gp_search, app as gp_app
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- CONFIG ---------------- #
STOP_WORDS = set(stopwords.words("english"))
SCRAPE_DELAY = 1.0
TOP_APPS_PER_COUNTRY = 10

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------- HELPERS ---------------- #
def extract_app_id(url):
    return parse_qs(urlparse(url).query).get("id", [None])[0]

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def tokenize(text):
    return [t for t in word_tokenize(clean_text(text)) if t not in STOP_WORDS and len(t) > 2]

# ---------------- SCRAPER ---------------- #
def get_app_data(app_id, country):
    try:
        d = gp_app(app_id, lang="en", country=country)
        return {
            "app_id": app_id,
            "title": d.get("title", ""),
            "desc": d.get("description", ""),
            "rating": d.get("score", 0),
            "reviews": d.get("reviews", 0),
            "country": country.upper()
        }
    except:
        return None

def get_competitors(seed_keywords, country):
    ids = []
    for kw in seed_keywords:
        try:
            res = gp_search(kw, lang="en", country=country, n_hits=TOP_APPS_PER_COUNTRY)
            ids.extend([r["appId"] for r in res])
        except:
            continue
    return list(dict.fromkeys(ids))[:TOP_APPS_PER_COUNTRY]

# ---------------- KEYWORD METRICS ---------------- #
def calculate_metrics(apps, term, freq):
    total_reviews = sum(a["reviews"] for a in apps if term in (a["title"] + a["desc"]).lower())
    avg_rating = sum(a["rating"] for a in apps if term in (a["title"] + a["desc"]).lower()) / max(1, len(apps))

    difficulty = math.log2(total_reviews + 10)
    popularity = freq * avg_rating

    opportunity = popularity / (difficulty + 1)
    probability = min(100, (opportunity / (difficulty + 1)) * 10)

    return round(difficulty, 2), round(opportunity, 2), round(probability, 2)

# ---------------- KEYWORD ENGINE ---------------- #
def extract_keywords(apps):
    docs = [a["title"] + " " + a["desc"] for a in apps]

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(2, 3), max_features=150)
    X = vectorizer.fit_transform(docs)

    freqs = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    keywords = []

    for term, freq in zip(terms, freqs):
        if len(term.split()) < 2:
            continue

        diff, opp, prob = calculate_metrics(apps, term, freq)

        keywords.append({
            "keyword": term,
            "difficulty": diff,
            "opportunity": opp,
            "probability": prob
        })

    keywords.sort(key=lambda x: x["probability"], reverse=True)
    return keywords[:50]

# ---------------- GAP ANALYSIS ---------------- #
def keyword_gap(keywords, own_text):
    own_text = own_text.lower()
    return [k for k in keywords if k["keyword"] not in own_text][:20]

# ---------------- COMPETITOR SCORING ---------------- #
def score_competitors(apps):
    scored = []
    for a in apps:
        strength = (a["rating"] * 2) + math.log2(a["reviews"] + 1)
        scored.append({**a, "strength": round(strength, 2)})
    return sorted(scored, key=lambda x: x["strength"], reverse=True)

# ---------------- FEATURE EXTRACTION ---------------- #
def extract_features(text):
    lines = text.split("\n")
    return [l.strip() for l in lines if len(l) > 20][:10]

# ---------------- KEYWORD SELECTION ---------------- #
def select_keywords(keywords):
    high = [k for k in keywords if k["probability"] > 70]
    mid = [k for k in keywords if 40 < k["probability"] <= 70]
    low = [k for k in keywords if k["probability"] <= 40]

    primary = [k["keyword"] for k in high[:3]]
    secondary = [k["keyword"] for k in mid[:5]]
    long_tail = [k["keyword"] for k in low[:10]]

    return primary, secondary, long_tail

# ---------------- ASO GENERATION ---------------- #
def generate_title():
    return "Volume Booster - Bass & EQ"

def generate_short(primary, secondary):
    return f"Boost {primary[0]}, {secondary[0]} & {secondary[1]} with powerful equalizer!"

def generate_long(app_name, primary, secondary, long_tail, features):
    sections = []

    sections.append(
        f"Tired of low volume and weak sound on your phone?\n\n"
        f"{app_name} boosts volume, enhances bass, and improves sound instantly."
    )

    sections.append(
        f"Enjoy louder music, deeper bass, and better audio for videos, games, and calls."
    )

    sections.append("━━━━━━━━━━━━━━━━━━━━\nFEATURES\n━━━━━━━━━━━━━━━━━━━━")
    for f in features:
        sections.append(f"✅ {f}")

    sections.append("━━━━━━━━━━━━━━━━━━━━\nWHY USERS LOVE IT\n━━━━━━━━━━━━━━━━━━━━")
    sections.append(
        f"• Boost {primary[0]}\n"
        f"• Improve {secondary[0]} & {secondary[1]}\n"
        f"• Works with headphones & speakers\n"
        f"• Easy to use"
    )

    sections.append("━━━━━━━━━━━━━━━━━━━━\nUSE CASES\n━━━━━━━━━━━━━━━━━━━━")
    sections.append("🎧 Music\n🎬 Movies\n🎮 Gaming\n📱 Low volume")

    sections.append(f"\nDownload {app_name} now!")

    sections.append("Tags: " + ", ".join(primary + secondary + long_tail))

    return "\n\n".join(sections)[:3990]

# ---------------- MAIN ---------------- #
def main():
    print("\n🔥 FINAL ASO ENGINE\n")

    url = input("Enter app URL:\n> ").strip()
    countries = input("Countries (us,in,pk):\n> ").split(",")

    app_id = extract_app_id(url)
    own = get_app_data(app_id, countries[0])

    print("⚡ Extracting seed keywords...")
    seed = [
        "volume booster",
        "bass booster",
        "equalizer app",
        "sound amplifier",
        "increase volume"
    ]

    apps = [own]
    seen = {app_id}

    print("⚡ Scraping competitors...")
    for c in countries:
        ids = get_competitors(seed, c)

        for aid in ids:
            if aid in seen:
                continue

            seen.add(aid)
            data = get_app_data(aid, c)

            if data and data["rating"] >= 4.0:
                apps.append(data)

            time.sleep(SCRAPE_DELAY)

    print("⚡ Keyword intelligence...")
    keywords = extract_keywords(apps)

    print("⚡ Gap analysis...")
    gaps = keyword_gap(keywords, own["title"] + own["desc"])

    print("⚡ Competitor scoring...")
    scored = score_competitors(apps)

    print("⚡ Feature extraction...")
    features = extract_features(own["desc"])

    primary, secondary, long_tail = select_keywords(keywords)

    title = generate_title()
    short = generate_short(primary, secondary)
    long = generate_long(own["title"], primary, secondary, long_tail, features)

    pd.DataFrame(scored).to_csv("competitors.csv", index=False)
    pd.DataFrame(keywords).to_csv("keyword_ranking.csv", index=False)
    pd.DataFrame(gaps).to_csv("keyword_gap.csv", index=False)

    pd.DataFrame([{
        "App Name": own["title"],
        "Title": title,
        "Short Description": short,
        "Long Description": long
    }]).to_csv("aso_output.csv", index=False)

    print("\n✅ DONE (FINAL ENGINE)")
    print("• competitors.csv")
    print("• keyword_ranking.csv")
    print("• keyword_gap.csv")
    print("• aso_output.csv")

if __name__ == "__main__":
    main()
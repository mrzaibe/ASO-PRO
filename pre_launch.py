import re
import time
import string
import logging
from collections import defaultdict

import pandas as pd
from google_play_scraper import search as gp_search, app as gp_app

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- CONFIG ---------------- #
STOP_WORDS = set(stopwords.words("english"))
SCRAPE_DELAY = 1.5
MAX_RESULTS = 20

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------- TEXT CLEAN ---------------- #
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def tokenize(text):
    tokens = word_tokenize(clean_text(text))
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

# ---------------- INPUT CLEAN ---------------- #
def clean_keywords_input(text):
    words = [w.strip().lower() for w in text.split(",") if w.strip()]
    corrections = {"birhtday": "birthday"}
    return [corrections.get(w, w) for w in words]

def clean_countries_input(text):
    return [c.strip().lower() for c in text.split(",") if c.strip()]

# ---------------- SCRAPER ---------------- #
def get_competitors(keyword, country):
    try:
        results = gp_search(keyword, lang="en", country=country, n_hits=MAX_RESULTS)
        return [r["appId"] for r in results]
    except Exception as e:
        log.warning(f"Search failed for {keyword}-{country}: {e}")
        return []

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

# ---------------- KEYWORD ENGINE ---------------- #
def extract_keywords(apps):
    counter = defaultdict(int)

    for app in apps:
        tokens = tokenize(app["title"] + " " + app["desc"])
        for t in tokens:
            counter[t] += 1

    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [{"keyword": k, "score": v} for k, v in ranked[:50]]

# ---------------- INTENT CLUSTER ---------------- #
def cluster_keywords(keywords):
    clusters = {
        "occasion": [],
        "action": [],
        "design": [],
        "general": []
    }

    for k in keywords:
        word = k["keyword"]

        if any(x in word for x in ["birthday", "wedding", "party", "baby"]):
            clusters["occasion"].append(word)
        elif any(x in word for x in ["maker", "create", "design", "edit"]):
            clusters["action"].append(word)
        elif any(x in word for x in ["card", "invitation", "poster"]):
            clusters["design"].append(word)
        else:
            clusters["general"].append(word)

    return clusters

# ---------------- COPYWRITING ---------------- #
def generate_title(app_name, keywords):
    top = [k["keyword"] for k in keywords[:5]]

    for kw in top:
        candidate = f"{app_name}: {kw.title()} Maker"
        if len(candidate) <= 30:
            return candidate

    return app_name[:30]

def generate_short_desc(app_name, keywords):
    top = [k["keyword"] for k in keywords[:5]]

    options = [
        f"Create stunning {top[0]}s in seconds — easy & fast!",
        f"Design {top[0]}s for {top[1]}s & events instantly!",
        f"Beautiful {top[0]} maker for every occasion!"
    ]

    for o in options:
        if len(o) <= 80:
            return o

    return options[0][:80]

# ---------------- LONG DESCRIPTION ---------------- #
def generate_long_desc(app_name, clusters, keywords):
    top_keywords = [k["keyword"] for k in keywords[:20]]

    def kw(i):
        return top_keywords[i] if i < len(top_keywords) else "design"

    sections = []

    sections.append(
        f"Looking for the best {kw(0)} maker? {app_name} helps you create stunning {kw(0)}s "
        f"for {kw(1)}s, {kw(2)}s, and {kw(3)}s in seconds — no design skills needed."
    )

    sections.append(
        f"{app_name} is an all-in-one {kw(0)} creator designed for speed, simplicity, and beauty. "
        f"Whether you're creating a {kw(1)} {kw(0)}, a {kw(2)} card, or a {kw(3)} poster, "
        f"this app makes everything effortless."
    )

    sections.append("━━━━━━━━━━━━━━━━━━━━\nKEY FEATURES\n━━━━━━━━━━━━━━━━━━━━")
    features = [
        f"Create {kw(0)}s instantly with ready-made templates",
        f"Design {kw(1)} invitations and {kw(2)} cards easily",
        f"Customize text, fonts, colors, and images",
        f"AI-powered suggestions for better {kw(0)} design",
        f"High-quality export for print and social media",
        f"Share your {kw(0)} via WhatsApp, Instagram & more",
    ]
    sections.append("\n".join([f"✅ {f}" for f in features]))

    sections.append("━━━━━━━━━━━━━━━━━━━━\nUSE CASES\n━━━━━━━━━━━━━━━━━━━━")
    use_cases = [
        f"🎂 {kw(1).title()} {kw(0)}s",
        f"💍 {kw(2).title()} invitations",
        f"🎉 Party and event {kw(0)}s",
        f"📢 Business flyers and posters",
        f"👶 Baby shower invitations",
        f"🎓 Graduation cards",
    ]
    sections.append("\n".join(use_cases))

    sections.append("━━━━━━━━━━━━━━━━━━━━\nHOW IT WORKS\n━━━━━━━━━━━━━━━━━━━━")
    sections.append(
        f"1️⃣ Choose a {kw(0)} template\n"
        f"2️⃣ Customize text and design\n"
        f"3️⃣ Preview and download\n"
        f"4️⃣ Share instantly"
    )

    sections.append("━━━━━━━━━━━━━━━━━━━━\nWHY USERS LOVE IT\n━━━━━━━━━━━━━━━━━━━━")
    sections.append(
        f"Thousands of users trust {app_name} to create beautiful {kw(0)}s quickly and easily."
    )

    sections.append(
        f"Download {app_name} now and start creating stunning {kw(0)}s today!"
    )

    sections.append("Tags: " + ", ".join(top_keywords[:15]))

    final = "\n\n".join(sections)
    return final[:3990]

# ---------------- MAIN ---------------- #
def main():
    print("\n🔥 ASO Generator (PRO VERSION)\n")

    app_name = input("Enter your app name:\n> ").strip()

    keywords_input = input("Enter keywords (comma-separated):\n> ")
    keywords_list = clean_keywords_input(keywords_input)

    countries_input = input("Enter countries (comma-separated):\n> ")
    countries = clean_countries_input(countries_input)

    apps = []
    seen_ids = set()

    for country in countries:
        for keyword in keywords_list:
            log.info(f"Searching '{keyword}' in {country}")

            app_ids = get_competitors(keyword, country)

            for aid in app_ids:
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                log.info(f"Scraping {aid}")
                data = get_app_data(aid, country)

                if data:
                    apps.append(data)

                time.sleep(SCRAPE_DELAY)

    if not apps:
        log.error("No data collected.")
        return

    # SAVE FILES
    pd.DataFrame(apps).to_csv("competitors.csv", index=False)

    keywords = extract_keywords(apps)
    pd.DataFrame(keywords).to_csv("keyword_ranking.csv", index=False)

    clusters = cluster_keywords(keywords)

    title = generate_title(app_name, keywords)
    short = generate_short_desc(app_name, keywords)
    long = generate_long_desc(app_name, clusters, keywords)

    pd.DataFrame([{
        "App Name": app_name,
        "Title": title,
        "Short Description": short,
        "Long Description": long
    }]).to_csv("aso_output.csv", index=False)

    print("\n✅ DONE! Files saved:")
    print("• competitors.csv")
    print("• keyword_ranking.csv")
    print("• aso_output.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Stopped by user")
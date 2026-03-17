import re
import time
import string
import logging
import math
from collections import Counter
from urllib.parse import urlparse, parse_qs

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from google_play_scraper import app as gp_app, search as gp_search

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STOP_WORDS = set(stopwords.words("english"))
FILLER_WORDS = STOP_WORDS | {
    "app", "apps", "get", "use", "new", "one", "also", "make", "like",
    "best", "good", "great", "just", "much", "many", "using", "used",
    "can", "will", "way", "even", "well", "made", "every", "let",
    "need", "may", "try", "take", "com", "play", "google", "android",
    "version", "update", "download", "free", "open", "http", "https",
}
MAX_SEARCH_RESULTS = 30
SCRAPE_DELAY = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, min_len: int = 3) -> list[str]:
    tokens = word_tokenize(clean_text(text))
    return [t for t in tokens if t not in FILLER_WORDS and len(t) >= min_len and t.isalpha()]


def extract_features_from_description(description: str) -> str:
    if not description:
        return ""
    feature_lines = []
    for line in description.split("\n"):
        stripped = line.strip()
        if re.match(r"^[•✓✔★▸►●\-\*]\s+", stripped):
            feature_lines.append(re.sub(r"^[•✓✔★▸►●\-\*]\s+", "", stripped))
    return "; ".join(feature_lines) if feature_lines else ""


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def fetch_app_details(app_id: str, country: str) -> dict | None:
    try:
        d = gp_app(app_id, lang="en", country=country)
        return {
            "app_id": app_id,
            "Country": country.upper(),
            "App Title": d.get("title", ""),
            "Developer": d.get("developer", ""),
            "Rating": d.get("score") or 0,
            "Reviews": d.get("reviews") or 0,
            "Short Description": d.get("summary", ""),
            "Long Description": d.get("description", ""),
            "Core Features": extract_features_from_description(d.get("description", "")),
        }
    except Exception as exc:
        log.warning("  Failed to fetch %s for %s: %s", app_id, country.upper(), exc)
        return None


def search_competitors(keyword: str, country: str) -> list[str]:
    try:
        results = gp_search(keyword, lang="en", country=country, n_hits=MAX_SEARCH_RESULTS)
        return [r["appId"] for r in results]
    except Exception as exc:
        log.warning("  Search failed for '%s' in %s: %s", keyword, country.upper(), exc)
        return []


def scrape_country(keyword: str, country: str) -> list[dict]:
    log.info("Searching for '%s' in %s …", keyword, country.upper())
    app_ids = search_competitors(keyword, country)
    log.info("  Found %d competitor app(s) in %s", len(app_ids), country.upper())

    rows: list[dict] = []
    seen: set[str] = set()
    for i, app_id in enumerate(app_ids, 1):
        if app_id in seen:
            continue
        seen.add(app_id)
        log.info("  [%s] Scraping %d/%d – %s", country.upper(), i, len(app_ids), app_id)
        details = fetch_app_details(app_id, country)
        if details:
            rows.append(details)
        time.sleep(SCRAPE_DELAY)

    log.info("  Collected %d app(s) for %s", len(rows), country.upper())
    return rows


# ---------------------------------------------------------------------------
# Keyword ranking
# ---------------------------------------------------------------------------

def rank_keywords_for_country(apps: list[dict], top_n: int = 40) -> list[dict]:
    documents = []
    for a in apps:
        documents.append(
            f"{a['App Title']} {a['Short Description']} {a['Long Description']}"
        )

    tfidf_map: dict[str, float] = {}
    if documents:
        vec = TfidfVectorizer(stop_words="english", max_features=800, min_df=1, max_df=0.95)
        try:
            matrix = vec.fit_transform(documents)
            scores = matrix.sum(axis=0).A1
            for term, score in zip(vec.get_feature_names_out(), scores):
                tfidf_map[term] = score
        except ValueError:
            pass

    keyword_stats: dict[str, dict] = {}

    for a in apps:
        title_tokens = set(tokenize(a["App Title"]))
        desc_tokens = set(tokenize(
            f"{a['Short Description']} {a['Long Description']}"
        ))
        all_tokens = title_tokens | desc_tokens
        rating = a.get("Rating") or 0
        reviews = a.get("Reviews") or 0

        for token in all_tokens:
            if token not in keyword_stats:
                keyword_stats[token] = {
                    "keyword": token,
                    "title_freq": 0,
                    "desc_freq": 0,
                    "total_reviews": 0,
                    "ratings": [],
                }
            stats = keyword_stats[token]
            if token in title_tokens:
                stats["title_freq"] += 1
            if token in desc_tokens:
                stats["desc_freq"] += 1
            stats["total_reviews"] += reviews
            stats["ratings"].append(rating)

    ranked = []
    for token, stats in keyword_stats.items():
        avg_rating = sum(stats["ratings"]) / len(stats["ratings"]) if stats["ratings"] else 0
        review_w = math.log2(stats["total_reviews"] + 2)
        tfidf_s = tfidf_map.get(token, 0.1)
        presence = 3 * stats["title_freq"] + stats["desc_freq"]
        final_score = presence * avg_rating * review_w * tfidf_s

        ranked.append({
            "keyword": token,
            "score": round(final_score, 2),
            "title_appearances": stats["title_freq"],
            "desc_appearances": stats["desc_freq"],
            "avg_competitor_rating": round(avg_rating, 2),
            "total_reviews": stats["total_reviews"],
            "tfidf": round(tfidf_s, 4),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]


def analyse_keywords_all_countries(
    competitor_df: pd.DataFrame,
) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    """Run keyword ranking per country. Returns (dict, combined DataFrame)."""
    country_kw_data: dict[str, list[dict]] = {}
    all_kw_rows: list[dict] = []

    for country, group in competitor_df.groupby("Country"):
        apps = group.to_dict("records")
        ranked = rank_keywords_for_country(apps, top_n=40)
        country_kw_data[str(country)] = ranked

        for rank, entry in enumerate(ranked, 1):
            row = {"Country": country, "Rank": rank}
            row.update(entry)
            all_kw_rows.append(row)

        top5 = [e["keyword"] for e in ranked[:5]]
        log.info("Top keywords for %s: %s", country, ", ".join(top5))

    return country_kw_data, pd.DataFrame(all_kw_rows)


# ---------------------------------------------------------------------------
# ASO Content Generation
# ---------------------------------------------------------------------------

def _kw(keywords: list[dict], start: int, end: int) -> list[str]:
    return [k["keyword"] for k in keywords[start:end]]


def generate_title(app_name: str, keywords: list[dict]) -> str:
    top = _kw(keywords, 0, 5)
    base = app_name.split(" - ")[0].strip()

    for kw in top:
        candidate = f"{base} - {kw.title()}"
        if len(candidate) <= 30:
            return candidate

    return base[:30]


def generate_short_description(
    app_name: str, features: list[str], keywords: list[dict]
) -> str:
    brand = app_name.split(" - ")[0].strip()
    top = _kw(keywords, 0, 6)
    if not top:
        top = ["featured", "new", "top", "easy", "fast", "best"]

    templates = [
        f"{brand}: Create stunning {top[0]}s & {top[1]} {top[2]}s — fast, easy & beautiful!",
        f"Design {top[0]}s for {top[1]}s, {top[2]}s & more. {brand} makes it effortless!",
        f"Beautiful {top[0]} {top[1]} for every occasion. {features[0]} with {brand}!" if features else f"Beautiful {top[0]} {top[1]} for every occasion with {brand}!",
        f"{brand} — AI-powered {top[0]} maker for {top[1]}s, {top[2]}s & every event!",
        f"Create pro {top[0]}s in seconds. {top[1].title()}, {top[2].title()} & more with {brand}!",
    ]

    for t in templates:
        if len(t) <= 80:
            return t

    return f"{brand}: Design beautiful {top[0]}s for every occasion!"[:80]


def generate_long_description(
    app_name: str,
    features: list[str],
    keywords: list[dict],
    competitor_apps: list[dict],
) -> str:
    brand = app_name.split(" - ")[0].strip()
    top = _kw(keywords, 0, 20)
    kw = lambda i: top[i] if i < len(top) else "design"  # noqa: E731

    feature_bullets = ""
    for feat in features:
        feat = feat.strip()
        if not feat:
            continue
        matching_kws = [k for k in top[:10] if k in feat.lower()]
        if matching_kws:
            feature_bullets += f"✅ {feat} — powered by advanced {matching_kws[0]} technology\n"
        else:
            feature_bullets += f"✅ {feat}\n"

    use_cases = [
        f"🎂 {kw(1).title()} {kw(0)}s — surprise your loved ones with personalized designs",
        f"💒 {kw(2).title()} {kw(0)}s — elegant layouts for your special day",
        f"🎉 {kw(3).title()} {kw(0)}s — vibrant designs for celebrations & events",
        f"💼 Corporate & business {kw(0)}s — professional templates for formal occasions",
        f"🎊 Baby shower, engagement, anniversary & holiday {kw(0)}s",
        f"📋 {kw(4).title()} for events, meetups & community gatherings",
    ]

    sections = []
    sections.append(
        f"Looking for the perfect {kw(0)} {kw(1)} maker? "
        f"{brand} is the all-in-one {kw(0)} designer that helps you "
        f"create stunning, professional-quality {kw(0)}s for {kw(1)}s, "
        f"{kw(2)}s, {kw(3)}s, and every special occasion — right from "
        f"your phone!"
    )
    sections.append(
        f"Whether you need a quick {kw(1)} {kw(0)}, an elegant "
        f"{kw(2)} {kw(0)}, or a fun {kw(3)} {kw(0)}, {brand} gives "
        f"you hundreds of beautiful, ready-to-use templates so you can "
        f"design and share in minutes, not hours."
    )
    sections.append(f"━━━━━━━━━━━━━━━━━━━━\nWHY CHOOSE {brand.upper()}?\n━━━━━━━━━━━━━━━━━━━━")
    sections.append(
        f"{brand} combines powerful {kw(0)} design tools with an "
        f"intuitive interface that anyone can use. No design experience "
        f"needed — just pick a template, customize it, and share your "
        f"masterpiece!"
    )
    sections.append(f"━━━━━━━━━━━━━━━━━━━━\nKEY FEATURES\n━━━━━━━━━━━━━━━━━━━━")
    sections.append(feature_bullets.strip() if feature_bullets else "✅ Hundreds of custom templates\n✅ Simple, intuitive interface\n✅ HD export and easy sharing")
    sections.append(f"━━━━━━━━━━━━━━━━━━━━\nWHAT YOU CAN CREATE\n━━━━━━━━━━━━━━━━━━━━")
    sections.append("\n".join(use_cases))

    # --- Keyword footer for indexing ---
    kw_line = ", ".join(top[:15])
    sections.append(f"Tags: {kw_line}")

    description = "\n\n".join(sections)
    if len(description) > 4000:
        description = description[:3997] + "..."
    return description


def generate_aso_content(
    country: str,
    keywords: list[dict],
    user_features: list[str],
    app_name: str,
    competitor_apps: list[dict],
) -> dict:
    title = generate_title(app_name, keywords)
    short = generate_short_description(app_name, user_features, keywords)
    long = generate_long_description(app_name, user_features, keywords, competitor_apps)
    kw_string = ", ".join(k["keyword"] for k in keywords[:15])

    return {
        "Country": country,
        "Generated Title": title,
        "Generated Short Description": short,
        "Generated Long Description": long,
        "Target Keywords": kw_string,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Google Play Store – ASO Generator (Unreleased App)")
    print("=" * 60 + "\n")

    app_name = input("1. Enter your Brand / App Name:\n   > ").strip()
    countries_raw = input("2. Enter country codes (comma-separated, e.g. us,pk,in,gb):\n   > ").strip()
    keyword = input("3. Enter target seed keyword for competition analysis:\n   > ").strip()
    user_features_raw = input("4. Enter your app's core features (comma-separated):\n   > ").strip()

    countries = [c.strip().lower() for c in countries_raw.split(",") if c.strip()]
    user_features = [f.strip() for f in user_features_raw.split(",") if f.strip()]

    log.info("App Name    : %s", app_name)
    log.info("Countries   : %s", ", ".join(c.upper() for c in countries))
    log.info("Seed Keyword: %s", keyword)

    # ---- Scrape competitors per country ----
    all_rows: list[dict] = []
    for country in countries:
        rows = scrape_country(keyword, country)
        all_rows.extend(rows)

    if not all_rows:
        log.error("No competitor data collected. Exiting.")
        return

    competitor_df = pd.DataFrame(all_rows)

    kw_col = competitor_df.apply(
        lambda r: ", ".join(tokenize(
            f"{r['App Title']} {r['Short Description']} {r['Long Description']}"
        )[:20]),
        axis=1,
    )
    competitor_df["Keywords"] = kw_col

    export_cols = [
        "Country", "App Title", "Developer", "Rating", "Reviews",
        "Short Description", "Long Description", "Core Features", "Keywords",
    ]
    competitor_csv = "competitor_apps_data.csv"
    competitor_df[export_cols].to_csv(competitor_csv, index=False, encoding="utf-8-sig")
    log.info("Saved competitor data → %s  (%d rows)", competitor_csv, len(competitor_df))

    # ---- Keyword ranking per country ----
    country_kw_data, kw_ranking_df = analyse_keywords_all_countries(competitor_df)

    kw_csv = "keyword_rankings.csv"
    kw_ranking_df.to_csv(kw_csv, index=False, encoding="utf-8-sig")
    log.info("Saved keyword rankings → %s  (%d rows)", kw_csv, len(kw_ranking_df))

    # ---- ASO content generation ----
    aso_rows: list[dict] = []
    for country in countries:
        cu = country.upper()
        kws = country_kw_data.get(cu, [])
        if not kws:
            log.warning("No keywords for %s – skipping ASO generation.", cu)
            continue
        country_apps = competitor_df[competitor_df["Country"] == cu].to_dict("records")
        aso = generate_aso_content(cu, kws, user_features, app_name, country_apps)
        aso_rows.append(aso)
        log.info("Generated ASO content for %s", cu)

    if aso_rows:
        aso_df = pd.DataFrame(aso_rows)
        aso_csv = "aso_generated_content.csv"
        aso_df.to_csv(aso_csv, index=False, encoding="utf-8-sig")
        log.info("Saved ASO content → %s  (%d rows)", aso_csv, len(aso_df))
    else:
        log.warning("No ASO content generated.")

    print("\n" + "=" * 60)
    print("  Done! Output files:")
    print(f"    1. {competitor_csv}")
    print(f"    2. {kw_csv}")
    print(f"    3. {aso_csv}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

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
            "short_desc": d.get("summary", ""),
            "desc": d.get("description", ""),
            "rating": d.get("score", 0),
            "reviews": d.get("reviews", 0),
            "installs": d.get("installs", ""),
            "country": country
        }
    except:
        return None

def get_app_by_title(title, country):
    """Search Play Store by app title and return top result."""
    try:
        res = gp_search(title, lang="en", country=country, n_hits=1)
        if res:
            return get_app_data(res[0]["appId"], country)
    except:
        pass
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
    docs = [
        (a.get("title") or "") + " " +
        (a.get("short_desc") or "") + " " +
        (a.get("desc") or "")
        for a in apps
    ]

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

# ---------- TITLE (max 30 chars) ----------
def generate_title(app_name, primary_keywords):
    brand = app_name.split(":")[0].split("-")[0].strip()
    short_brand = brand[:14].strip()

    for kw in primary_keywords:
        kw_title = kw.title()
        candidate = f"{short_brand}: {kw_title}"
        if len(candidate) <= 30:
            return candidate
        if len(kw_title) <= 30:
            return kw_title

    return brand[:30]

# ---------- SHORT DESCRIPTION (max 80 chars) ----------
def generate_short(primary, secondary, app_name):
    brand = app_name.split(":")[0].split("-")[0].strip()

    templates = []
    if primary and secondary:
        templates.append(f"{primary[0].title()} & {secondary[0]} — fast, free & easy.")
        templates.append(f"Best {primary[0]} app. Try {secondary[0]} today!")
        templates.append(f"Boost your {primary[0]} with smart {secondary[0]} tools.")
    if primary:
        templates.append(f"Powerful {primary[0]} tool you'll actually love using.")
        templates.append(f"Simple, fast {primary[0]} — no ads, no hassle.")
    templates.append(f"The smartest {brand} app on the Play Store.")

    for t in templates:
        if len(t) <= 80:
            return t

    return (templates[0] if templates else "Fast, powerful & easy to use.")[:80]

# ---------- LONG DESCRIPTION (max 4000 chars) ----------
def generate_long_desc(app_name, primary, secondary, long_tail, core_features):
    p1 = primary[0] if len(primary) > 0 else "features"
    p2 = primary[1] if len(primary) > 1 else "performance"
    p3 = primary[2] if len(primary) > 2 else "tools"
    s1 = secondary[0] if len(secondary) > 0 else "ease of use"
    s2 = secondary[1] if len(secondary) > 1 else "speed"
    lt1 = long_tail[0] if len(long_tail) > 0 else ""
    lt2 = long_tail[1] if len(long_tail) > 1 else ""

    brand = app_name.split(":")[0].split("-")[0].strip()

    # Feature bullets: use user input if provided, else generate from keywords
    if core_features and core_features.strip():
        feature_lines = [f.strip() for f in core_features.strip().splitlines() if f.strip()]
        feature_bullets = "\n".join([f"✅ {f}" for f in feature_lines[:8]])
    else:
        feature_bullets = (
            f"✅ Lightning-fast {p1} with zero lag\n"
            f"✅ Smart {p2} engine built for real users\n"
            f"✅ Clean, intuitive interface — no learning curve\n"
            f"✅ Works offline and optimised for all screen sizes\n"
            f"✅ Regular updates with new {p3} improvements\n"
            f"✅ Battery-friendly with low memory usage"
        )

    # Use case sentence (from long-tail keywords)
    use_case_parts = []
    if lt1:
        use_case_parts.append(lt1)
    if lt2:
        use_case_parts.append(lt2)
    use_case_str = ""
    if use_case_parts:
        use_case_str = f"Whether you need {' or '.join(use_case_parts)}, {brand} has you covered.\n\n"

    desc = f"""{brand} is built for people who want real {p1} without the clutter.

Most apps in this space are bloated, full of ads, or simply don't deliver. {brand} is different — designed from the ground up to give you smooth {p2}, reliable {s1}, and everything you need to get things done faster.

{use_case_str}━━━━━━━━━━━━━━━━━━━━
WHAT YOU GET
━━━━━━━━━━━━━━━━━━━━
{feature_bullets}

━━━━━━━━━━━━━━━━━━━━
WHY USERS CHOOSE {brand.upper()}
━━━━━━━━━━━━━━━━━━━━
Users stick with {brand} because it combines powerful {p3} with a clean experience that doesn't get in the way. The {s1} is seamless, the {s2} speaks for itself, and you'll feel the difference from your very first use.

This isn't just another utility app — it's a tool built with care, tested by real users, and refined to handle the things that matter most.

━━━━━━━━━━━━━━━━━━━━
PERFECT FOR
━━━━━━━━━━━━━━━━━━━━
📱 Everyday users who want something reliable
⚡ Power users who need fast {p2} without compromise
🎯 Anyone tired of slow, ad-heavy alternatives

Download {brand} today and discover why it's the go-to {p1} app for thousands of users worldwide."""

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
st.caption("Keyword research & store listing optimizer for Google Play — no API key required")

# ---------- INPUT FORM ----------
with st.form("aso_form"):
    input_type = st.radio("Input type", ["Play Store URL", "App Title / Name"], horizontal=True)
    app_input = st.text_input(
        "Play Store URL" if input_type == "Play Store URL" else "App Title or Name",
        placeholder=(
            "https://play.google.com/store/apps/details?id=com.example.app"
            if input_type == "Play Store URL"
            else "e.g. Volume Booster Increase Sound"
        )
    )
    # ── Country tiers ──────────────────────────────────────────────────────────
    TIER1 = ["us", "gb", "ca", "au", "de", "fr", "jp", "kr", "nl", "se",
             "no", "dk", "fi", "ch", "ie", "nz", "sg", "il", "at", "be"]

    TIER2 = ["in", "br", "mx", "ru", "es", "it", "pl", "tr", "ar", "za",
             "ae", "sa", "ng", "eg", "id", "th", "vn", "ph", "pk", "my",
             "co", "cl", "pe", "cz", "ro", "hu", "pt", "gr", "ua", "kw"]

    TIER3 = ["bd", "np", "lk", "mm", "kh", "la", "mn", "uz", "kz", "az",
             "ge", "am", "by", "md", "rs", "bg", "hr", "si", "sk", "lt",
             "lv", "ee", "is", "mt", "cy", "lu", "tn", "ma", "dz", "ly",
             "gh", "ke", "tz", "et", "ug", "ci", "cm", "sn", "ao", "mz",
             "zw", "zm", "bw", "rw", "sd", "so", "ye", "iq", "ir", "sy",
             "jo", "lb", "qa", "bh", "om", "bo", "ec", "py", "uy", "ve",
             "cr", "gt", "hn", "sv", "ni", "pa", "do", "cu", "jm", "tt",
             "ht", "bb", "bs", "gy", "sr", "fj", "pg", "ws", "to", "vu"]

    ALL_COUNTRIES = TIER1 + TIER2 + TIER3

    TIER_MAP = {
        "🌍 All Countries": ALL_COUNTRIES,
        "⭐ Tier 1 — High Income (US, UK, DE, JP...)": TIER1,
        "🌐 Tier 2 — Emerging Markets (IN, BR, MX, TR...)": TIER2,
        "🌏 Tier 3 — Developing Markets (BD, MM, GH, BO...)": TIER3,
    }

    tier_choice = st.selectbox(
        "Quick-select by Tier",
        options=list(TIER_MAP.keys()),
        index=0,
        help=(
            "Tier 1: High-income English/Western markets with strong monetisation.\n"
            "Tier 2: Large emerging markets with fast-growing Play Store user bases.\n"
            "Tier 3: Developing markets — high volume, lower ARPU."
        )
    )

    selected_countries = st.multiselect(
        "Countries to analyze (edit after tier selection)",
        options=ALL_COUNTRIES,
        default=TIER_MAP[tier_choice],
        help="Tip: start with Tier 1 or Tier 2 for faster results. All countries = more data but slower."
    )
    core_features = st.text_area(
        "Core Features of Your App (one per line — optional but recommended)",
        placeholder="e.g.\nBoost volume up to 200%\nEqualizer with bass & treble control\nWorks without internet",
        height=120
    )
    submitted = st.form_submit_button("🔍 Analyze")

if submitted:
    if not app_input.strip():
        st.error("Please enter a Play Store URL or app title.")
        st.stop()

    country_list = selected_countries if selected_countries else ["us"]

    # Progress bar
    progress = st.progress(0)
    status = st.empty()

    status.info("🔍 Fetching your app data...")
    progress.progress(10)

    # Resolve own app
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

    progress.progress(25)
    status.info("🌱 Extracting seed keywords from your app...")

    desc = own.get("desc") or ""
    title_text = own.get("title") or ""
    seed = extract_seed_keywords(title_text + " " + desc)
    if not seed:
        seed = [title_text]

    # Collect competitors per country
    all_apps_by_country = {c: [] for c in country_list}
    all_apps = [own]

    total = len(country_list)
    for idx, c in enumerate(country_list):
        status.info(f"🌍 Scraping competitors — **{c.upper()}** ({idx+1}/{total})...")
        ids = get_competitors(seed, c)
        for aid in ids:
            data = get_app_data(aid, c)
            if data:
                all_apps_by_country[c].append(data)
                all_apps.append(data)
            time.sleep(0.15)
        pct = 25 + int(50 * (idx + 1) / total)
        progress.progress(pct)

    if len(all_apps) < 2:
        st.error("Not enough competitor data. Try again or add more countries.")
        st.stop()

    progress.progress(80)
    status.info("🧠 Analysing keywords & writing ASO copy...")

    kw_df = extract_keywords(all_apps)

    if kw_df.empty:
        st.error("Keyword extraction failed.")
        st.stop()

    keywords = kw_df["keyword"].tolist()
    if len(keywords) < 3:
        st.error("Not enough keywords extracted.")
        st.stop()

    primary   = keywords[:3]
    secondary = keywords[3:6]
    long_tail = keywords[6:15] if len(keywords) > 6 else []
    app_name  = own.get("title", "My App")

    title = generate_title(app_name, primary)
    short = generate_short(primary, secondary, app_name)
    long  = generate_long_desc(app_name, primary, secondary, long_tail, core_features)

    progress.progress(100)
    status.success("✅ Analysis complete!")
    time.sleep(0.5)
    status.empty()
    progress.empty()

    # ============================================================
    # RESULTS TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔑 Keywords", "✍️ ASO Copy"])

    # ---- TAB 1: Overview ----
    with tab1:
        st.subheader("Your App")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Title", own.get("title", "—"))
        c2.metric("Rating", f"{own.get('rating', 0):.1f} ⭐")
        c3.metric("Reviews", f"{own.get('reviews', 0):,}")
        c4.metric("Installs", own.get("installs", "—"))

        if own.get("short_desc"):
            st.caption(f"**Short description:** {own.get('short_desc')}")

        st.divider()

        # Country filter
        selected_country = st.selectbox("Filter competitors by country", ["All"] + country_list)

        st.subheader("Competitor Apps")
        comp_rows = []
        for c, apps_in_c in all_apps_by_country.items():
            if selected_country != "All" and c != selected_country:
                continue
            for a in apps_in_c:
                comp_rows.append({
                    "Country": c.upper(),
                    "Title": a.get("title", ""),
                    "Short Description": a.get("short_desc", ""),
                    "Rating": round(a.get("rating", 0), 1),
                    "Reviews": a.get("reviews", 0),
                    "Installs": a.get("installs", "")
                })

        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
        else:
            st.info("No competitor data for selected country.")

    # ---- TAB 2: Keywords ----
    with tab2:
        st.subheader("Keyword Opportunities")
        kw_country = st.selectbox("Analyse keywords from country", ["All"] + country_list, key="kw_country")

        if kw_country == "All":
            display_apps = all_apps
        else:
            display_apps = [a for a in all_apps if a.get("country") == kw_country]
            if own not in display_apps:
                display_apps = [own] + display_apps

        filtered_kw_df = extract_keywords(display_apps)

        if not filtered_kw_df.empty:
            st.dataframe(filtered_kw_df, use_container_width=True)
            st.bar_chart(filtered_kw_df.set_index("keyword")["opportunity"].head(15))
        else:
            st.info("Not enough data to extract keywords for this country.")

    # ---- TAB 3: ASO Copy ----
    with tab3:
        st.subheader("Generated ASO Listing")

        st.markdown("#### 📌 Title")
        st.caption("Google Play limit: **30 characters**")
        title_col, char_col = st.columns([5, 1])
        with title_col:
            title_edit = st.text_input("Title", value=title, max_chars=30, label_visibility="collapsed")
        with char_col:
            char_color = "🟢" if len(title_edit) <= 30 else "🔴"
            st.markdown(f"{char_color} `{len(title_edit)}/30`")

        st.divider()

        st.markdown("#### 📝 Short Description")
        st.caption("Google Play limit: **80 characters**")
        short_col, sc_col = st.columns([5, 1])
        with short_col:
            short_edit = st.text_area("Short", value=short, max_chars=80, height=80, label_visibility="collapsed")
        with sc_col:
            sc_color = "🟢" if len(short_edit) <= 80 else "🔴"
            st.markdown(f"{sc_color} `{len(short_edit)}/80`")

        st.divider()

        st.markdown("#### 📄 Long Description")
        st.caption("Google Play limit: **4,000 characters**")
        long_col, lc_col = st.columns([5, 1])
        with long_col:
            long_edit = st.text_area("Long", value=long, max_chars=4000, height=520, label_visibility="collapsed")
        with lc_col:
            lc_color = "🟢" if len(long_edit) <= 4000 else "🔴"
            st.markdown(f"{lc_color} `{len(long_edit)}/4000`")

        st.divider()
        st.markdown("#### 🔑 Keywords Embedded in Description")
        st.caption("Top opportunity keywords woven into your listing:")
        all_used = primary + secondary + (long_tail[:5] if long_tail else [])
        kw_cols = st.columns(3)
        for i, kw in enumerate(all_used):
            kw_cols[i % 3].markdown(f"- `{kw}`")

# app_fk_bijwerkingen.py
# 🇳🇱 Farmacotherapeutisch Kompas — Bijwerkingen checker (narratief, met frequentiecategorieën)
# - Zoekt per geneesmiddel de FK-preparaatpagina (fuzzy mogelijk)
# - Leest uitsluitend de <p>-alinea's in <section id="bijwerkingen">
# - Detecteert frequentiecategorie per alinea
# - Matrix toont: ✅ <Categorie> (optioneel aantal treffers)
# - Evidentie per middel met categorie & highlights

import html
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optioneel: RapidFuzz voor fuzzy matching
try:
    from rapidfuzz import fuzz  # type: ignore
    RF_AVAILABLE = True
except Exception:
    fuzz = None  # type: ignore
    RF_AVAILABLE = False

BASE = "https://www.farmacotherapeutischkompas.nl"
SESSION = requests.Session()
DEFAULT_HEADERS = {
    "User-Agent": "FK-bijwerkingen-checker/1.3 (+contact: your-email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_TIMEOUT = 25
REQUEST_PAUSE_S = 0.6  # beleefdheidspauze tussen requests

# ---------------- UI ----------------
st.set_page_config(page_title="FK — Bijwerkingen checker", page_icon="🇳🇱", layout="wide")
st.title("🇳🇱 Farmacotherapeutisch Kompas — Bijwerkingen checker (met frequentiecategorieën)")

with st.sidebar:
    st.header("Instellingen")
    enable_fuzzy_meds = st.checkbox(
        "Fuzzy zoeken voor geneesmiddelnaam (corrigeer spelfouten)",
        value=True,
        help="Bijv. ‘bleomycinee’ → ‘bleomycine’."
    )
    use_fuzzy_effects = st.checkbox(
        "Fuzzy matching voor bijwerkingen",
        value=RF_AVAILABLE,
        help="Zoekt lichte varianten/typo’s in de tekst."
    )
    show_counts = st.checkbox(
        "Toon aantal treffers per cel",
        value=True
    )
    st.caption("Tip: pas de User-Agent in de code aan als je dit intensief gebruikt.")
    st.markdown("---")
    st.markdown("Bron: **Farmacotherapeutisch Kompas** — sectie **Bijwerkingen** per preparaatpagina.")

st.subheader("1) Patiëntgegevens invoeren")
c1, c2 = st.columns(2)
with c1:
    meds_raw = st.text_area(
        "Geneesmiddelen (één per regel of komma/semicolon-gescheiden)",
        placeholder="bijv.\nomeprazol\nmacrogol\nmorfine\noxycodon\nlosartan\nbleomycine",
        height=170,
    )
with c2:
    effects_raw = st.text_area(
        "Geobserveerde bijwerkingen/klachten (één per regel of komma/semicolon-gescheiden)",
        placeholder="bijv.\nhoofdpijn\nmisselijkheid\nobstipatie\nhuiduitslag",
        height=170,
    )

run = st.button("Matrix opbouwen", type="primary")

# ---------------- Helpers ----------------

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_ascii(s: str) -> str:
    """Verwijder diacritics voor robuuste vergelijkingen."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def split_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\n;]+", s)
    seen = set()
    out: List[str] = []
    for p in (t.strip() for t in parts):
        if not p:
            continue
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def http_get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    try:
        r = SESSION.get(url, params=params, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r
    except Exception:
        return None

# ---- FK zoeken ----
def snelzoeken_fk(term: str) -> List[Tuple[str, str]]:
    """Gebruik FK-snelzoeken binnen 'geneesmiddelen'. Retourneert (titel, absolute_url)."""
    url = f"{BASE}/snelzoeken"
    params = {"domein": "geneesmiddelen", "zoekterm": term}
    r = http_get(url, params=params)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    links: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = normalize_spaces(a.get_text(" "))
        if "/bladeren/preparaatteksten/" in href:
            abs_url = href if href.startswith("http") else (BASE + href)
            links.append((text, abs_url))
    # uniek, volgorde behouden
    seen, uniq = set(), []
    for t, u in links:
        if u not in seen:
            seen.add(u)
            uniq.append((t, u))
    return uniq

def best_match(term: str, candidates: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """Kies beste kandidaat o.b.v. titel/slug + (optioneel) fuzzy score."""
    if not candidates:
        return None
    term_norm = normalize_ascii(term).lower()
    scored: List[Tuple[int, str, str]] = []
    for title, url in candidates:
        title_norm = normalize_ascii(title).lower()
        slug = normalize_ascii(url.split("/")[-1]).lower()
        base = 0
        if term_norm == title_norm or term_norm == slug:
            base = 100
        elif term_norm in title_norm or term_norm in slug:
            base = 92
        if RF_AVAILABLE and fuzz is not None:
            base = max(base, fuzz.partial_ratio(term_norm, title_norm), fuzz.partial_ratio(term_norm, slug))
        scored.append((int(base), title, url))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    return (best[1], best[2]) if best[0] >= 70 else None

def resolve_med_to_fk_url(user_term: str, enable_fuzzy: bool = True) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Zoek de beste FK-preparaatpagina bij user_term. Geeft (titel, url, notitie) terug."""
    hits = snelzoeken_fk(user_term)
    note = None
    if hits:
        best = best_match(user_term, hits) if enable_fuzzy else hits[0]
        if best:
            title, url = best
            if enable_fuzzy and normalize_ascii(user_term).lower() not in normalize_ascii(title).lower():
                note = f"Automatisch gecorrigeerd: “{user_term}” → “{title}”"
            return title, url, note

    # Fallback: eenvoudige slug-gok
    slug_guess = normalize_ascii(user_term).lower().replace(" ", "_")
    if slug_guess:
        for prefix in [slug_guess, f"{slug_guess}__systemisch_", f"{slug_guess}__parenteraal_", f"{slug_guess}__intraveneus_", f"{slug_guess}__oraal_"]:
            first = prefix[0]
            url = f"{BASE}/bladeren/preparaatteksten/{first}/{prefix}"
            r = http_get(url)
            if r and r.status_code == 200:
                soup = BeautifulSoup(r.text, "lxml")
                h1 = soup.find("h1")
                title = normalize_spaces(h1.get_text(" ")) if h1 else user_term
                note = f"Pagina gevonden via slug-schatting ({prefix})"
                return title, url, note
    return None, None, None

# ---- Robuuste weergavenaam uit pagina ----
def extract_display_title_from_page(url: str, html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    candidates = []
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        candidates.append(og["content"])
    if soup.title and soup.title.string:
        candidates.append(soup.title.string)
    for tag in soup.find_all(["h1", "h2"], limit=3):
        txt = normalize_spaces(tag.get_text(" "))
        if txt and "Farmacotherapeutisch Kompas" not in txt:
            candidates.append(txt)
            break
    slug = url.rstrip("/").split("/")[-1]
    candidates.append(slug.replace("__", " ").replace("_", " "))

    for cand in candidates:
        if not cand:
            continue
        c = re.split(r"\s*[|–—-]\s*Farmacotherapeutisch Kompas", cand, maxsplit=1)[0]
        c = re.sub(r"\s{2,}", " ", c).strip()
        if c:
            return c
    return "Onbekend middel"

# ---- Frequentiecategorieën uit FK-tekst ----
FREQ_CANON = {
    "zeer vaak": "Zeer vaak",      # >10%
    "vaak": "Vaak",                # 1–10%
    "soms": "Soms",                # 0,1–1%
    "zelden": "Zelden",            # 0,01–0,1%
    "zeer zelden": "Zeer zelden",  # <0,01%
}
# Prioriteit bij samenvoegen (hoogst eerst)
FREQ_RANK = {"Zeer vaak": 5, "Vaak": 4, "Soms": 3, "Zelden": 2, "Zeer zelden": 1, "Overig": 0}

def detect_freq_category(paragraph_text: str) -> str:
    """
    Detecteer frequentie aan het begin van de alinea:
    'Zeer vaak (...):', 'Vaak (...):', 'Soms (...):', 'Zelden (...):', 'Zeer zelden (...):'
    Alles daarbuiten -> 'Overig' (bv. 'Verder zijn gemeld ...').
    """
    if not paragraph_text:
        return "Overig"
    head = paragraph_text.strip().lower()
    m = re.match(r"^\s*(zeer vaak|vaak|soms|zelden|zeer zelden)\b.*?:", head, flags=re.I)
    if m:
        return FREQ_CANON.get(m.group(1).lower(), "Overig")
    if head.startswith("verder zijn gemeld") or head.startswith("verder kan voorkomen"):
        return "Overig"
    return "Overig"

# ---- Extractie: uitsluitend <p>-alinea's in section#bijwerkingen ----
def extract_bijwerkingen_paragraphs(html_text: str) -> List[Dict[str, str]]:
    """
    Vind <section id="bijwerkingen"> en retourneer [{"category": "...", "text": "..."}].
    Negeert linklijsten (Lareb etc.). Valt terug op h2 'Bijwerkingen' als section ontbreekt.
    """
    soup = BeautifulSoup(html_text, "lxml")

    def collect_from_container(container) -> List[Dict[str, str]]:
        for ul in container.select("ul.link-list, ul[class*='link']"):
            ul.decompose()
        out: List[Dict[str, str]] = []
        for p in container.find_all("p"):
            txt = p.get_text("\n", strip=True)
            if not txt:
                continue
            out.append({"category": detect_freq_category(txt), "text": txt})
        return out

    sec = soup.find("section", id="bijwerkingen")
    if sec:
        return collect_from_container(sec)

    # Fallback: h2/h3 'Bijwerkingen' → siblings tot volgende kop
    h2 = None
    toc = soup.find("a", string=lambda s: s and s.strip().lower() == "bijwerkingen")
    if toc and toc.get("href", "").startswith("#"):
        h2 = soup.find(id=toc.get("href", "")[1:])
    if not h2:
        for tag in soup.find_all(re.compile("^h[2-3]$")):
            if tag.get_text(strip=True).lower() == "bijwerkingen":
                h2 = tag
                break
    if not h2:
        return []

    nodes = []
    for sib in h2.next_siblings:
        if getattr(sib, "name", None) and re.match(r"^h[2-3]$", sib.name, re.I):
            break
        nodes.append(sib)
    soup2 = BeautifulSoup("".join(str(n) for n in nodes), "lxml")
    return collect_from_container(soup2)

# ---- Matching & highlights ----
def match_term_in_text(term: str, text: str, allow_fuzzy: bool = True) -> Tuple[bool, int]:
    """(is_match, approx_count) — eerst woordgrenzen, dan optioneel fuzzy."""
    t = term.strip().lower()
    if not t or not text:
        return False, 0
    blob = re.sub(r"\s+", " ", text).lower()

    # Exact-ish
    pattern = r"(?<!\w)" + re.escape(t) + r"(?!\w)"
    hits = re.findall(pattern, blob)
    if hits:
        return True, len(hits)

    # Fuzzy
    if allow_fuzzy and RF_AVAILABLE and fuzz is not None:
        tokens = blob.split()
        window = max(1, min(6, len(t.split()) + 1))
        count = 0
        threshold = 85
        for i in range(len(tokens) - window + 1):
            seg = " ".join(tokens[i : i + window])
            if fuzz.partial_ratio(t, seg) >= threshold:
                count += 1
        if count:
            return True, count

    return False, 0

def highlight_terms_safe(text: str, terms: List[str]) -> str:
    """
    Veilig highlighten in plain text:
    1) markeer matches met placeholders; 2) HTML-escape; 3) placeholders → <mark>; 4) \n → <br>
    (Highlighting is exact-ish; fuzzy-treffers worden niet gemarkeerd.)
    """
    if not text:
        return ""
    if not terms:
        return html.escape(text).replace("\n", "<br>")

    PRE, POST = "\u0001HL\u0001", "\u0001/HL\u0001"
    terms_sorted = sorted({t for t in terms if t.strip()}, key=lambda x: -len(x))

    marked = text
    for t in terms_sorted:
        pattern = re.compile(rf"(?i)(?<!\w)({re.escape(t)})(?!\w)")
        marked = pattern.sub(PRE + r"\1" + POST, marked)

    escaped = html.escape(marked).replace("\n", "<br>")
    escaped = escaped.replace(html.escape(PRE), "<mark>").replace(html.escape(POST), "</mark>")
    return escaped

# --------------- Datamodel ---------------
@dataclass
class DrugRecord:
    input_term: str
    display_title: str
    url: str
    bijwerkingen_text: str
    bijwerkingen_html: str
    bijwerkingen_paragraphs: List[Dict[str, str]]   # [{"category","text"}]
    correction_note: Optional[str] = None

# --------------- Pipeline ---------------
def fetch_fk_bijwerkingen_for_med(med: str, allow_fuzzy: bool = True) -> Optional[DrugRecord]:
    title_guess, url, note = resolve_med_to_fk_url(med, enable_fuzzy=allow_fuzzy)
    if not url:
        return None

    time.sleep(REQUEST_PAUSE_S)
    r = http_get(url)
    if not r:
        return None

    html_page = r.text
    display_title = extract_display_title_from_page(url, html_page)

    paras = extract_bijwerkingen_paragraphs(html_page)
    bijw_text = "\n\n".join(p["text"] for p in paras)
    bijw_html = "".join(f"<p>{html.escape(p['text'])}</p>" for p in paras)

    return DrugRecord(
        input_term=med,
        display_title=display_title,
        url=url,
        bijwerkingen_text=bijw_text,
        bijwerkingen_html=bijw_html,
        bijwerkingen_paragraphs=paras,
        correction_note=note
    )

def build_matrix(drugs: List[DrugRecord], effects: List[str], fuzzy_effects: bool, show_counts: bool = True) -> pd.DataFrame:
    """
    Voor elke Geneesmiddel × Bijwerking: toon een checkmark + hoogste frequentiecategorie.
    Voorbeeld cel: '✅ Vaak (3)' of '✅ Zeer zelden' — leeg als geen match.
    """
    cols = ["Geneesmiddel"] + effects
    rows = []
    for rec in drugs:
        row = {"Geneesmiddel": rec.display_title}
        for eff in effects:
            best_cat, best_rank, total_hits = None, -1, 0
            for para in rec.bijwerkingen_paragraphs:
                matched, count = match_term_in_text(eff, para["text"], allow_fuzzy=fuzzy_effects)
                if matched:
                    total_hits += count
                    rank = FREQ_RANK.get(para["category"], 0)
                    if rank > best_rank:
                        best_rank, best_cat = rank, para["category"]

            if best_cat:
                label = f"✅ {best_cat}"
                if show_counts and total_hits > 1:
                    label += f" ({total_hits})"
                row[eff] = label
            else:
                row[eff] = ""
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

# ---------------- RUN ----------------
if run:
    meds = split_list(meds_raw)
    effects = split_list(effects_raw)

    if not meds:
        st.warning("Voer minimaal één geneesmiddel in.")
        st.stop()
    if not effects:
        st.warning("Voer minimaal één bijwerking/klacht in.")
        st.stop()

    st.info(
        f"Zoeken in het **Farmacotherapeutisch Kompas** voor **{len(meds)}** middel(en) "
        f"en kruisen met **{len(effects)}** bijwerking(en) — uitsluitend de sectie **Bijwerkingen**."
    )

    with st.spinner("Zoeken en ophalen…"):
        records: List[DrugRecord] = []
        status_lines: List[str] = []
        for med in meds:
            rec = fetch_fk_bijwerkingen_for_med(med, allow_fuzzy=enable_fuzzy_meds)
            if rec:
                note = f"  _({rec.correction_note})_" if rec.correction_note else ""
                lens = len(rec.bijwerkingen_text)
                status_lines.append(
                    f"• **{rec.display_title}** — bijwerkingen-tekst: {lens} tekens{note} — [FK-pagina]({rec.url})"
                )
                records.append(rec)
            else:
                status_lines.append(f"• **{med}** — geen FK-preparaatpagina gevonden")

    st.subheader("Zoeksamenvatting")
    for line in status_lines:
        st.markdown(line)

    if not records:
        st.warning("Geen bruikbare middelen gevonden met een ‘Bijwerkingen’-sectie.")
        st.stop()

    # Matrix
    st.subheader("2) Kruistabel (Geneesmiddel × Bijwerking) — met checkmark + categorie")
    df = build_matrix(records, effects, fuzzy_effects=use_fuzzy_effects, show_counts=show_counts)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download matrix (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="fk_bijwerkingen_matrix.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download matrix (JSON)",
            data=df.to_json(orient="records").encode("utf-8"),
            file_name="fk_bijwerkingen_matrix.json",
            mime="application/json",
        )

    # Evidentie
    st.subheader("3) Bijwerkingen per middel (met categorie & highlights)")
    for rec in records:
        with st.container():
            exp1 = st.expander(f"Bijwerkingen — {rec.display_title}", expanded=False)
            with exp1:
                st.markdown(f"Bron: [{rec.url}]({rec.url})")
                if not rec.bijwerkingen_paragraphs:
                    st.info("Geen afzonderlijke ‘Bijwerkingen’-sectie gevonden.")
                else:
                    for para in rec.bijwerkingen_paragraphs:
                        st.markdown(f"**Categorie:** {para['category']}")
                        st.markdown(highlight_terms_safe(para["text"], effects), unsafe_allow_html=True)
                        st.markdown("---")

            #exp2 = st.expander(f"Ruwe HTML — {rec.display_title}", expanded=False)
            #with exp2:
            #    if rec.bijwerkingen_html:
            #        st.code(rec.bijwerkingen_html, language="html")
            #    else:
            #        st.write("—")

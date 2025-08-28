# streamlit_app.py
import html
import math
import re
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# Optional fuzzy matching (for both med correction and effect matching)
try:
    from rapidfuzz import fuzz  # type: ignore
    RF_AVAILABLE = True
except Exception:
    fuzz = None  # type: ignore
    RF_AVAILABLE = False

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
MAX_LIMIT = 100  # OpenFDA per-request cap

st.set_page_config(page_title="OpenFDA — Adverse Reactions (Narrative) Cross-check", page_icon="💊", layout="wide")
st.title("💊 OpenFDA — Adverse Reactions (Narrative) Cross-check Matrix")

# =========================
# Sidebar / controls
# =========================
with st.sidebar:
    st.header("Search settings")
    field = st.selectbox(
        "Medication search field",
        options=[
            ("openfda.generic_name", "Generic name (recommended)"),
            ("openfda.brand_name", "Brand name"),
            ("openfda.substance_name", "Substance/active ingredient"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]
    sort = st.selectbox("Sort labels", ["effective_time:desc", "effective_time:asc"], index=0)
    per_med_limit = st.number_input("Max labels per medication", min_value=1, max_value=200, value=50, step=5)
    api_key = st.text_input("OpenFDA API key (optional)", type="password")

    st.markdown("---")
    enable_fuzzy_meds = st.checkbox("Enable fuzzy medication lookup (suggest corrections)", value=True)
    use_fuzzy_effects = st.checkbox("Enable fuzzy side-effect matching", value=RF_AVAILABLE, help="Uses RapidFuzz if available.")

st.subheader("1) Enter patient data")

col1, col2 = st.columns(2)
with col1:
    meds_raw = st.text_area(
        "Medication list (one per line or comma/semicolon-separated)",
        placeholder="e.g.\nibuprofen\nomeprazole\nbleomycine",
        height=180,
    )
with col2:
    effects_raw = st.text_area(
        "Observed side effects (one per line or comma/semicolon-separated)",
        placeholder="e.g.\nnausea\nabdominal pain\nrash",
        height=180,
    )

run = st.button("Build cross-check matrix", type="primary")

# =========================
# Helpers
# =========================
def split_list(s: str) -> List[str]:
    """Split by comma/semicolon/newline, clean, de-dup (case-insensitive) preserving order."""
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
            out.append(p)  # keep original casing for display
    return out

@st.cache_data(show_spinner=False)
def fetch_openfda_labels(
    search_value: str,
    search_field: str,
    sort: str,
    total_limit: int,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Fetch up to total_limit label results (newest first). Returns (results, errors)."""
    results: List[Dict] = []
    errors: List[Dict] = []

    fetched = 0
    pages = math.ceil(total_limit / MAX_LIMIT)
    for page in range(pages):
        limit = min(MAX_LIMIT, total_limit - fetched)
        if limit <= 0:
            break

        params = {
            "search": f'{search_field}:"{search_value}"',
            "sort": sort,
            "limit": limit,
            "skip": fetched,
        }
        if api_key:
            params["api_key"] = api_key

        try:
            r = requests.get(OPENFDA_BASE, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            batch = data.get("results", [])
            if not batch:
                break
            results.extend(batch)
            fetched += len(batch)
            if len(batch) < limit:
                break
        except requests.HTTPError as e:
            try:
                payload = r.json()
            except Exception:
                payload = None
            errors.append({"page": page, "http_error": str(e), "payload": payload})
            break
        except Exception as e:
            errors.append({"page": page, "error": str(e)})
            break

    return results, errors

@st.cache_data(show_spinner=False)
def fetch_value_counts(
    field: str,
    pattern: str,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Use OpenFDA's aggregation to list candidate values for a field that match a wildcard pattern.
    Example: search=openfda.generic_name:bleo*&count=openfda.generic_name.exact
    Returns list of {"term": "...", "count": n}
    """
    params = {
        "search": f"{field}:{pattern}",
        "count": f"{field}.exact"
    }
    if api_key:
        params["api_key"] = api_key

    r = requests.get(OPENFDA_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []

def best_string_match(query: str, candidates: List[str]) -> Optional[str]:
    """Pick the candidate closest to query using RapidFuzz (if available) or difflib."""
    if not candidates:
        return None
    if RF_AVAILABLE:
        # partial_ratio handles extra tokens like "bleomycin sulfate"
        scored = [(c, fuzz.partial_ratio(query.lower(), c.lower())) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, score = scored[0]
        # threshold tuned to catch common suffix/prefix differences
        return best if score >= 75 else None
    # fallback to difflib
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=0.7)
    return matches[0] if matches else None

def stem_prefixes(s: str) -> List[str]:
    """
    Heuristic: build 2-3 trailing-wildcard prefixes for suggestion queries.
    e.g., 'bleomycine' -> ['bleomy*', 'bleom*', 'bleo*']
    """
    s = re.sub(r"[^A-Za-z0-9\s\-]+", "", s).strip()
    s = s.replace("  ", " ")
    s = s.split()[0] if s else s  # first token usually enough
    if not s:
        return []
    L = len(s)
    picks = []
    for n in (6, 5, 4):
        if L >= n:
            picks.append(s[:n] + "*")
    if not picks:
        picks.append(s + "*")
    # de-dup preserve order
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def distinct_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def try_resolve_medication(
    user_term: str,
    primary_field: str,
    api_key: Optional[str],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Returns (resolved_field, resolved_value, tried_patterns)
    Strategy:
      1) Query value counts with trailing-wildcard prefixes on primary field.
      2) If none, try the two alternate fields.
      3) Choose best candidate by fuzzy similarity.
    """
    fields_order = distinct_preserve_order([primary_field, "openfda.generic_name", "openfda.brand_name", "openfda.substance_name"])
    tried_patterns: List[str] = []
    for f in fields_order:
        prefixes = stem_prefixes(user_term)
        candidates: List[str] = []
        for pat in prefixes:
            tried_patterns.append(f"{f}:{pat}")
            try:
                rows = fetch_value_counts(f, pat, api_key)
                candidates.extend([row["term"] for row in rows if "term" in row])
            except Exception:
                # ignore and continue
                pass
        candidates = distinct_preserve_order(candidates)
        best = best_string_match(user_term, candidates)
        if best:
            return f, best, tried_patterns
    return None, None, tried_patterns

def normalize_narrative(item: Dict) -> List[str]:
    """Return list of narrative paragraphs from adverse_reactions."""
    adv = item.get("adverse_reactions") or []
    out: List[str] = []
    for t in adv:
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out

def narrative_blocks_to_text(blocks: List[str]) -> str:
    """Lowercased, whitespace-normalized blob for matching."""
    text = "\n".join(blocks)
    text = re.sub(r"\s+", " ", text).lower()
    return text

def match_term_in_text(term: str, text: str, allow_fuzzy: bool = True) -> Tuple[bool, int]:
    """
    Returns (is_match, approx_count) against the narrative text blob.
    - Exact-ish: word-boundary substring count
    - Fuzzy (RapidFuzz) for small variations
    """
    t = term.strip().lower()
    if not t or not text:
        return False, 0

    # exact-ish first
    pattern = r"(?<!\w)" + re.escape(t) + r"(?!\w)"
    hits = re.findall(pattern, text)
    if hits:
        return True, len(hits)

    # fuzzy fallback
    if allow_fuzzy and RF_AVAILABLE:
        tokens = text.split()
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
    Safely highlight terms in plain text:
    1) Insert markers in the ORIGINAL text using regex (case-insensitive, word boundaries).
    2) HTML-escape the entire text.
    3) Replace markers with <mark> tags.
    Also converts newlines to <br> so paragraphs render properly.
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

@dataclass
class DrugRecord:
    display_name: str
    labels: List[Dict]
    text_narr: str
    narr_blocks: List[str]
    original_term: str
    resolved_field: str
    resolved_value: str
    corrected: bool

def build_drug_record(
    drug: str,
    search_field: str,
    sort: str,
    per_med_limit: int,
    api_key: Optional[str],
    enable_fuzzy_meds: bool,
) -> Tuple[Optional[DrugRecord], List[Dict], Optional[str]]:
    """
    Fetch labels for a single drug and compile narrative text corpus only.
    Returns (DrugRecord or None, errors, status_message)
    """
    # First, try exactly as entered
    results, errors = fetch_openfda_labels(drug, search_field, sort, per_med_limit, api_key)

    used_field = search_field
    used_value = drug
    corrected_note = None
    corrected = False

    # If nothing, attempt fuzzy resolution to a canonical value (e.g., bleomycine -> bleomycin)
    if not results and enable_fuzzy_meds:
        rf, rv, tried = try_resolve_medication(drug, search_field, api_key)
        if rf and rv:
            used_field, used_value = rf, rv
            corrected = (rv.lower() != drug.lower()) or (rf != search_field)
            corrected_note = f"Corrected “{drug}” → “{rv}” via {rf.split('.')[-1]}"
            # Re-query with corrected value
            results, errors2 = fetch_openfda_labels(used_value, used_field, sort, per_med_limit, api_key)
            if errors2:
                errors.extend(errors2)

    # Keep only labels that actually have narrative adverse_reactions
    results = [r for r in results if normalize_narrative(r)]

    if not results:
        return None, errors, corrected_note  # surface errors but no record

    narr_blocks: List[str] = []
    for r in results:
        narr_blocks.extend(normalize_narrative(r))

    text_narr = narrative_blocks_to_text(narr_blocks) if narr_blocks else ""

    # Friendly display name from first label (fall back to corrected value)
    fda = results[0].get("openfda", {}) or {}
    g = ", ".join(fda.get("generic_name", []) or [])
    b = ", ".join(fda.get("brand_name", []) or [])
    display = g or b or used_value

    rec = DrugRecord(
        display_name=display,
        labels=results,
        text_narr=text_narr,
        narr_blocks=narr_blocks,
        original_term=drug,
        resolved_field=used_field,
        resolved_value=used_value,
        corrected=corrected,
    )
    return rec, errors, corrected_note

def crosscheck_matrix(
    drugs: List[DrugRecord],
    effects: List[str],
    use_fuzzy: bool = True
) -> pd.DataFrame:
    """Build Drug × Side-effect matrix with ✅ and small match counts (narrative only)."""
    cols = ["Drug"] + effects
    rows = []
    for d in drugs:
        row = {"Drug": d.display_name}
        for eff in effects:
            matched, count = match_term_in_text(eff, d.text_narr, allow_fuzzy=use_fuzzy)
            row[eff] = "✅" + (f" ({count})" if count > 1 else "") if matched else ""
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

def meta_line(item: Dict) -> str:
    fda = item.get("openfda", {}) or {}
    g = ", ".join(fda.get("generic_name", []) or [])
    b = ", ".join(fda.get("brand_name", []) or [])
    s = ", ".join(fda.get("substance_name", []) or [])
    et = item.get("effective_time", "")
    return f"**Effective:** {et} | **Generic:** {g or '—'} | **Brand:** {b or '—'} | **Substance:** {s or '—'}"

# =========================
# RUN
# =========================
if run:
    meds = split_list(meds_raw)
    effects = split_list(effects_raw)

    if not meds:
        st.warning("Please enter at least one medication.")
        st.stop()
    if not effects:
        st.warning("Please enter at least one observed side effect.")
        st.stop()

    st.info(f"Searching labels for **{len(meds)}** medication(s) and cross-checking **{len(effects)}** side effect(s) in the narrative adverse reactions.")

    with st.spinner("Fetching from OpenFDA and building datasets…"):
        drug_records: List[DrugRecord] = []
        fetch_errors: List[Dict] = []
        per_drug_status: List[str] = []

        for med in meds:
            rec, errs, note = build_drug_record(
                drug=med,
                search_field=field,
                sort=sort,
                per_med_limit=int(per_med_limit),
                api_key=api_key.strip() or None,
                enable_fuzzy_meds=enable_fuzzy_meds,
            )
            if errs:
                fetch_errors.extend([{"medication": med, **e} for e in errs])
            if rec:
                drug_records.append(rec)
                status = f"• **{rec.display_name}** — labels: {len(rec.labels)} | narrative length: {len(rec.text_narr)} chars"
                if rec.corrected and note:
                    status += f"  _(auto-corrected: {note})_"
                per_drug_status.append(status)
            else:
                msg = f"• **{med}** — no labels with narrative 'adverse_reactions' found"
                if note:
                    msg += f"  _(attempted correction: {note})_"
                per_drug_status.append(msg)

    # Fetch summary
    st.subheader("Fetch summary")
    for line in per_drug_status:
        st.write(line)

    if fetch_errors:
        with st.expander("Show fetch errors"):
            for err in fetch_errors:
                st.error(err)

    if not drug_records:
        st.warning("No usable labels with narrative 'adverse_reactions' found. Try a different search field.")
        st.stop()

    # ----- MATRIX -----
    st.subheader("2) Cross-check Matrix (Narrative only)")
    matrix_df = crosscheck_matrix(drug_records, effects, use_fuzzy=use_fuzzy_effects)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download matrix (CSV)",
            data=matrix_df.to_csv(index=False).encode("utf-8"),
            file_name="crosscheck_matrix_narrative.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download matrix (JSON)",
            data=matrix_df.to_json(orient="records").encode("utf-8"),
            file_name="crosscheck_matrix_narrative.json",
            mime="application/json",
        )

    # ----- EVIDENCE PER DRUG -----
    st.subheader("3) Narrative evidence per medication (with highlights)")
    for rec in drug_records:
        with st.container():
            exp_narr = st.expander(f"Narrative for: {rec.display_name}", expanded=False)
            with exp_narr:
                # show each label's meta + narrative blocks with highlights
                for item in rec.labels:
                    blocks = normalize_narrative(item)
                    if not blocks:
                        continue
                    st.markdown(meta_line(item))
                    for para in blocks:
                        highlighted = highlight_terms_safe(para, effects)
                        st.markdown(highlighted, unsafe_allow_html=True)

            #exp_json = st.expander(f"Raw label JSON — {rec.display_name}", expanded=False)
            #with exp_json:
            #    for item in rec.labels:
            #        st.json(item, expanded=False)

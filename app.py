import os
import json
import csv
import textwrap
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

# ==============================================================================
# LEADFORGE REVIEW DASHBOARD
# Author: Umer Mujahid
# Deliverable: Stage 6 Review Screen (app.py)
# Output: data/06_approved.jsonl
# ==============================================================================

DEFAULT_APPROVED_PATH = os.path.join("data", "06_approved.jsonl")
RANKING_SHEET_FILLED_PATH = os.path.join("data", "ranking_sheet_filled.csv")
RANKING_SHEET_PATH = os.path.join("data", "ranking_sheet.csv")

# Pipeline stages metadata for live teammate data tracking
PIPELINE_STAGES = [
    {
        "stage_num": 1,
        "name": "Collect",
        "owner": "Nurul Huda",
        "file": os.path.join("data", "01_leads.jsonl"),
        "desc": "OpenStreetMap Overpass API business collection & website scraping"
    },
    {
        "stage_num": 2,
        "name": "Visual Audit",
        "owner": "Intern 5",
        "file": os.path.join("data", "02_visual.jsonl"),
        "desc": "Desktop/mobile screenshots & automated website health checks"
    },
    {
        "stage_num": 3,
        "name": "Research",
        "owner": "Haseeb Khan",
        "file": os.path.join("data", "03_research.jsonl"),
        "desc": "LLM site analysis, 3 structured findings & quote verification"
    },
    {
        "stage_num": 4,
        "name": "Scoring & Validation",
        "owner": "Azlan",
        "file": os.path.join("data", "04_scored.jsonl"),
        "desc": "4-signal scoring (0-100), Bands (A/B/C/D), reasons & 20-lead human ranking validation"
    },
    {
        "stage_num": 5,
        "name": "Email Writing",
        "owner": "Amna Miraj",
        "file": os.path.join("data", "05_drafts.jsonl"),
        "desc": "Personalized outreach email drafting referencing verified findings"
    },
    {
        "stage_num": 6,
        "name": "Review Dashboard",
        "owner": "Umer Mujahid",
        "file": os.path.join("data", "06_approved.jsonl"),
        "desc": "Human-in-the-loop review, edit, approval & sandbox delivery"
    }
]


# ==============================================================================
# DATA MANAGEMENT HELPERS
# ==============================================================================

def find_available_datasets():
    """Discover all real JSONL pipeline data files present in workspace."""
    found = []
    
    # Priority order: latest stages first
    priority_paths = [
        os.path.join("data", "05_drafts.jsonl"),
        os.path.join("data", "04_scored.jsonl"),
        os.path.join("data", "03_research.jsonl"),
        os.path.join("data", "02_visual.jsonl"),
        os.path.join("data", "01_leads.jsonl"),
        os.path.join("data", "06_approved.jsonl"),
    ]
    
    for p in priority_paths:
        if os.path.exists(p) and p not in found:
            found.append(p)
            
    # Check for any other stage jsonl files inside data/
    if os.path.exists("data"):
        for f in sorted(os.listdir("data")):
            full_p = os.path.join("data", f)
            if f.endswith(".jsonl") and "sample" not in f.lower() and full_p not in found:
                found.append(full_p)
                
    return found


def load_jsonl(file_path):
    """Load records from a JSONL file into a list of dictionaries."""
    records = []
    if not os.path.exists(file_path):
        return records
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue
    return records


def load_decisions(approved_file=DEFAULT_APPROVED_PATH):
    """Load previously saved decisions from 06_approved.jsonl keyed by lead_id."""
    decisions = {}
    if os.path.exists(approved_file):
        records = load_jsonl(approved_file)
        for r in records:
            lid = r.get("lead_id")
            if lid:
                decisions[lid] = r
    return decisions


def save_decision(lead_record, decision, final_subject, final_body, reviewer="Umer Mujahid", approved_file=DEFAULT_APPROVED_PATH):
    """
    Append or update an approved/rejected lead in 06_approved.jsonl.
    Follows Rule 1: Copies EVERY existing field through untouched, then adds decision metadata.
    """
    os.makedirs(os.path.dirname(approved_file), exist_ok=True)
    
    existing_records = []
    if os.path.exists(approved_file):
        existing_records = load_jsonl(approved_file)
    
    now_utc = datetime.now(timezone.utc).isoformat()
    
    updated_record = dict(lead_record)
    updated_record["decision"] = decision
    updated_record["final_subject"] = final_subject
    updated_record["final_body"] = final_body
    updated_record["decided_at"] = now_utc
    updated_record["decided_by"] = reviewer
    
    lead_id = lead_record.get("lead_id")
    replaced = False
    new_records_list = []
    for r in existing_records:
        if r.get("lead_id") == lead_id:
            new_records_list.append(updated_record)
            replaced = True
        else:
            new_records_list.append(r)
            
    if not replaced:
        new_records_list.append(updated_record)
        
    with open(approved_file, "w", encoding="utf-8") as f:
        for rec in new_records_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    return updated_record


def remove_decision(lead_id, approved_file=DEFAULT_APPROVED_PATH):
    """Revert a decision, removing the record from 06_approved.jsonl."""
    if not os.path.exists(approved_file):
        return
    existing_records = load_jsonl(approved_file)
    new_list = [r for r in existing_records if r.get("lead_id") != lead_id]
    with open(approved_file, "w", encoding="utf-8") as f:
        for rec in new_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_ranking_sheet(csv_path=RANKING_SHEET_FILLED_PATH):
    """
    Load human validation rankings from Stage 4 ranking sheet CSV.
    Returns a dict mapping lead_id to validation ranking dict.
    """
    if not os.path.exists(csv_path):
        if os.path.exists(RANKING_SHEET_PATH):
            csv_path = RANKING_SHEET_PATH
        else:
            return {}
            
    ranks = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rank_cols = [c for c in (reader.fieldnames or []) if "human" in c.lower() and "rank" in c.lower()]
            for row in reader:
                lid = row.get("lead_id")
                if not lid:
                    continue
                h_ranks = []
                for c in rank_cols:
                    val = row.get(c, "").strip()
                    if val:
                        try:
                            h_ranks.append(float(val))
                        except ValueError:
                            pass
                avg_rank = (sum(h_ranks) / len(h_ranks)) if h_ranks else None
                ranks[lid] = {
                    "lead_id": lid,
                    "name": row.get("name"),
                    "domain": row.get("domain"),
                    "human1_rank": row.get("human1_rank_1_to_20") or row.get("human_rank_1_to_20"),
                    "human2_rank": row.get("human2_rank_1_to_20"),
                    "human3_rank": row.get("human3_rank_1_to_20"),
                    "avg_human_rank": avg_rank,
                    "site_status": row.get("site_status")
                }
    except Exception:
        pass
    return ranks


def compute_score_breakdown(lead):
    """
    Calculate the detailed Stage 4 score breakdown based on active signals:
    1. Site down (15 pts)
    2. Visual/technical checks (up to 40 pts)
    3. Stage 3 verified conversion findings (up to 25 pts)
    4. Substantial site text proxy (up to 20 pts)
    5. Category match bonus (up to 15 pts)
    """
    breakdown = []
    
    # Signal 0: Site failed to load
    if lead.get("status") == "error":
        breakdown.append({
            "category": "Site Health",
            "name": f"Site failed to load ({lead.get('error', 'error')})",
            "points": 15,
            "max_points": 15,
            "status": "triggered"
        })
        
    # Signal 1: Visual/Technical checks
    # Check 1: Contact method
    has_contact = lead.get("phone_visible") or lead.get("contact_form")
    if "phone_visible" in lead or "contact_form" in lead:
        if not has_contact:
            breakdown.append({
                "category": "Technical Audit",
                "name": "No visible phone or contact form",
                "points": 12,
                "max_points": 12,
                "status": "triggered"
            })
        else:
            breakdown.append({
                "category": "Technical Audit",
                "name": "Contact method present",
                "points": 0,
                "max_points": 12,
                "status": "passed"
            })
            
    # Check 2: Mobile responsive
    if "horizontal_scroll_mobile" in lead:
        if lead.get("horizontal_scroll_mobile"):
            breakdown.append({
                "category": "Technical Audit",
                "name": "Mobile horizontal-scroll issue",
                "points": 10,
                "max_points": 10,
                "status": "triggered"
            })
        else:
            breakdown.append({
                "category": "Technical Audit",
                "name": "Mobile responsive",
                "points": 0,
                "max_points": 10,
                "status": "passed"
            })
            
    # Check 3: Load speed
    if "loads_under_5_seconds" in lead:
        if lead.get("loads_under_5_seconds") is False:
            breakdown.append({
                "category": "Technical Audit",
                "name": "Slow page load (> 5 seconds)",
                "points": 10,
                "max_points": 10,
                "status": "triggered"
            })
        else:
            breakdown.append({
                "category": "Technical Audit",
                "name": "Fast page load (< 5 seconds)",
                "points": 0,
                "max_points": 10,
                "status": "passed"
            })
            
    # Check 4: Meta description
    if "meta_description_present" in lead:
        if lead.get("meta_description_present") is False:
            breakdown.append({
                "category": "Technical Audit",
                "name": "Missing meta description",
                "points": 8,
                "max_points": 8,
                "status": "triggered"
            })
        else:
            breakdown.append({
                "category": "Technical Audit",
                "name": "Meta description present",
                "points": 0,
                "max_points": 8,
                "status": "passed"
            })
            
    # Signal 2: Verified conversion findings (Stage 3)
    findings = lead.get("findings", []) or []
    conversion_findings = [
        f for f in findings
        if f.get("category") == "conversion" and f.get("quote_verified")
    ]
    if conversion_findings:
        pts = min(len(conversion_findings) * 12, 25)
        breakdown.append({
            "category": "Stage 3 Research",
            "name": f"{len(conversion_findings)} verified conversion finding(s)",
            "points": pts,
            "max_points": 25,
            "status": "triggered"
        })
    else:
        breakdown.append({
            "category": "Stage 3 Research",
            "name": "No verified conversion findings",
            "points": 0,
            "max_points": 25,
            "status": "none"
        })
        
    # Signal 3: Site text content volume
    text_len = len(lead.get("site_text", "") or "")
    if text_len >= 500:
        breakdown.append({
            "category": "Content Volume",
            "name": f"Substantial site content ({text_len} chars)",
            "points": 20,
            "max_points": 20,
            "status": "triggered"
        })
    elif text_len >= 150:
        breakdown.append({
            "category": "Content Volume",
            "name": f"Moderate site content ({text_len} chars)",
            "points": 10,
            "max_points": 20,
            "status": "partial"
        })
    else:
        breakdown.append({
            "category": "Content Volume",
            "name": f"Thin site content ({text_len} chars - low confidence)",
            "points": 0,
            "max_points": 20,
            "status": "none"
        })
        
    # Signal 4: Target category match bonus
    if lead.get("category") == "restaurant":
        breakdown.append({
            "category": "Category Match",
            "name": "Target category match (restaurant)",
            "points": 15,
            "max_points": 15,
            "status": "triggered"
        })
    elif lead.get("category"):
        breakdown.append({
            "category": "Category Match",
            "name": f"Category: {lead.get('category')}",
            "points": 0,
            "max_points": 15,
            "status": "none"
        })
        
    return breakdown


# ==============================================================================
# UI RENDERING APPLICATION
# ==============================================================================

def render_app():
    st.set_page_config(
        page_title="LeadForge Review Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clean, professional styling
    st.markdown(textwrap.dedent("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Clean Professional Header */
        .leadforge-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            border-radius: 12px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .leadforge-title {
            font-size: 1.75rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .leadforge-subtitle {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 0.3rem;
            max-width: 750px;
            line-height: 1.4;
        }
        
        /* Metric Box */
        .metric-card {
            background: #1e293b;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 0.9rem 1rem;
            text-align: center;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #38bdf8;
            letter-spacing: -0.02em;
        }
        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94a3b8;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        
        /* Findings Box */
        .finding-box {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #6366f1;
            border-radius: 8px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 0.75rem;
        }
        .finding-claim {
            font-weight: 600;
            color: #f8fafc;
            font-size: 0.92rem;
        }
        .finding-quote {
            font-size: 0.84rem;
            color: #94a3b8;
            font-style: italic;
            margin-top: 0.35rem;
            line-height: 1.4;
        }
        
        /* Score breakdown item */
        .score-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.45rem 0.75rem;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 6px;
            margin-bottom: 0.4rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.84rem;
        }
        
        /* Screenshot Fallback */
        .screenshot-fallback {
            background: #090d16;
            border: 2px dashed rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 2.2rem 1rem;
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
        }
        
        /* Health Checklist Pill */
        .health-pill {
            display: inline-flex;
            align-items: center;
            background: #0f172a;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        
        /* Pipeline Card */
        .pipeline-card {
            background: #1e293b;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.15rem;
            margin-bottom: 1rem;
        }
    </style>
    """), unsafe_allow_html=True)

    # ==============================================================================
    # SIDEBAR CONTROLS
    # ==============================================================================

    st.sidebar.markdown("### Pipeline Data Source")

    available_datasets = find_available_datasets()
    if not available_datasets:
        st.sidebar.warning("No stage files found in data/. Run pipeline stages (01_collect.py, 02_visual.py, 04_score.py) to generate data.")
        st.stop()

    # Format clean labels for datasets
    dataset_labels = {}
    for p in available_datasets:
        count = len(load_jsonl(p))
        if "04_scored.jsonl" in p:
            label = f"Stage 4: Scored Leads ({count} rows)"
        elif "03_research.jsonl" in p:
            label = f"Stage 3: LLM Research ({count} rows)"
        elif "02_visual.jsonl" in p:
            label = f"Stage 2: Visual Audit ({count} rows)"
        elif "01_leads.jsonl" in p:
            label = f"Stage 1: Collected Leads ({count} rows)"
        elif "05_drafts.jsonl" in p:
            label = f"Stage 5: Drafted Emails ({count} rows)"
        elif "06_approved.jsonl" in p:
            label = f"Stage 6: Approved Outbox ({count} rows)"
        else:
            label = f"{os.path.basename(p)} ({count} rows)"
        dataset_labels[label] = p

    # Default to Stage 4 (scored real leads) or Stage 5 if available, else first available
    default_idx = 0
    labels_list = list(dataset_labels.keys())
    for idx, lbl in enumerate(labels_list):
        if "Stage 4: Scored Leads" in lbl:
            default_idx = idx
            break
        elif "Stage 5" in lbl:
            default_idx = idx
            break

    selected_dataset_label = st.sidebar.selectbox(
        "Active Dataset",
        options=labels_list,
        index=default_idx,
        help="Choose real upstream stage data to review in the dashboard."
    )
    selected_dataset_path = dataset_labels[selected_dataset_label]

    reviewer_name = st.sidebar.text_input("Reviewer Name", value="Umer Mujahid")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters & Search")

    search_query = st.sidebar.text_input("Search Lead", placeholder="Business name, domain, city...")

    band_filter = st.sidebar.multiselect(
        "Filter by Band",
        options=["Band A", "Band B", "Band C", "Band D"],
        default=["Band A", "Band B", "Band C", "Band D"]
    )

    status_filter = st.sidebar.multiselect(
        "Filter by Status",
        options=["Pending", "Approved", "Rejected"],
        default=["Pending", "Approved", "Rejected"]
    )

    validation_only = st.sidebar.checkbox(
        "Human-Validated Leads Only (20 sample)",
        value=False,
        help="Filter to the 20 leads independently ranked by teammates in Stage 4 validation."
    )

    conversion_only = st.sidebar.checkbox(
        "Has Verified Conversion Finding",
        value=False,
        help="Filter to leads with verified quote conversion findings from Stage 3."
    )

    sort_option = st.sidebar.selectbox(
        "Sort Leads",
        options=[
            "Score: High to Low",
            "Score: Low to High",
            "Name: A to Z",
            "Lead ID"
        ],
        index=0
    )

    # Load active data, decisions, and ranking validation sheet
    raw_leads = load_jsonl(selected_dataset_path)
    saved_decisions = load_decisions(DEFAULT_APPROVED_PATH)
    validation_ranks = load_ranking_sheet()

    # Cross-reference Stage 3 LLM findings if viewing another stage like Stage 4
    stage3_path = os.path.join("data", "03_research.jsonl")
    stage3_findings = {}
    if os.path.exists(stage3_path) and selected_dataset_path != stage3_path:
        for r in load_jsonl(stage3_path):
            lid = r.get("lead_id")
            f_list = r.get("findings")
            if lid and f_list:
                stage3_findings[lid] = f_list

    leads_list = []
    for lead in raw_leads:
        lid = lead.get("lead_id", "unknown_id")
        lead_copy = dict(lead)
        
        # Enrich findings from Stage 3 if not present in active record
        if not lead_copy.get("findings") and lid in stage3_findings:
            lead_copy["findings"] = stage3_findings[lid]
            
        # Attach human validation ranking data if present
        if lid in validation_ranks:
            lead_copy["validation_rank"] = validation_ranks[lid]
            
        if lid in saved_decisions:
            lead_copy["decision"] = saved_decisions[lid].get("decision", "pending")
            lead_copy["final_subject"] = saved_decisions[lid].get("final_subject", lead.get("subject", ""))
            lead_copy["final_body"] = saved_decisions[lid].get("final_body", lead.get("body", ""))
            lead_copy["decided_at"] = saved_decisions[lid].get("decided_at", None)
            lead_copy["decided_by"] = saved_decisions[lid].get("decided_by", None)
        else:
            lead_copy["decision"] = lead.get("decision", "pending")
            lead_copy["final_subject"] = lead.get("subject", "")
            lead_copy["final_body"] = lead.get("body", "")
            lead_copy["decided_at"] = lead.get("decided_at", None)
            lead_copy["decided_by"] = lead.get("decided_by", None)
        leads_list.append(lead_copy)

    # Filter logic
    filtered_leads = []
    for lead in leads_list:
        lid = lead.get("lead_id", "")
        name = lead.get("name", "")
        domain = lead.get("domain", "")
        city = lead.get("city", "")
        category = lead.get("category", "")
        band = lead.get("band")
        
        decision = lead.get("decision", "pending")
        status_label = "Approved" if decision in ["approve", "edit"] else ("Rejected" if decision == "reject" else "Pending")
        
        if search_query:
            q = search_query.lower()
            if not (q in name.lower() or q in domain.lower() or q in city.lower() or q in category.lower() or q in lid.lower()):
                continue
                
        if band and f"Band {band}" not in band_filter:
            continue
            
        if status_label not in status_filter:
            continue
            
        if validation_only and lid not in validation_ranks:
            continue
            
        if conversion_only:
            findings = lead.get("findings", []) or []
            has_conv = any(f.get("category") == "conversion" and f.get("quote_verified") for f in findings)
            if not has_conv:
                continue
            
        filtered_leads.append(lead)

    # Sorting logic
    if sort_option == "Score: High to Low":
        filtered_leads.sort(key=lambda x: x.get("score", 0), reverse=True)
    elif sort_option == "Score: Low to High":
        filtered_leads.sort(key=lambda x: x.get("score", 0))
    elif sort_option == "Name: A to Z":
        filtered_leads.sort(key=lambda x: x.get("name", "").lower())
    elif sort_option == "Lead ID":
        filtered_leads.sort(key=lambda x: x.get("lead_id", ""))

    # ==============================================================================
    # HEADER BANNER & KPI METRICS
    # ==============================================================================

    st.markdown(textwrap.dedent("""
    <div class="leadforge-header">
        <div>
            <div class="leadforge-title">LEADFORGE <span style="color:#6366f1; font-weight:400;">| Review Screen</span></div>
            <div class="leadforge-subtitle">Stage 6 Human-in-the-Loop Review Dashboard — Inspect real business findings, audit website health, analyze Stage 4 scoring breakdowns, refine outreach emails, and approve for sandbox delivery.</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    total_count = len(leads_list)
    approved_count = sum(1 for l in leads_list if l.get("decision") in ["approve", "edit"])
    rejected_count = sum(1 for l in leads_list if l.get("decision") == "reject")
    pending_count = total_count - (approved_count + rejected_count)
    band_a_count = sum(1 for l in leads_list if l.get("band") == "A")
    band_b_count = sum(1 for l in leads_list if l.get("band") == "B")
    band_c_count = sum(1 for l in leads_list if l.get("band") == "C")
    band_d_count = sum(1 for l in leads_list if l.get("band") == "D")

    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_count}</div><div class="metric-label">Total Leads</div></div>', unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4ade80;">{approved_count}</div><div class="metric-label">Approved ({round((approved_count/total_count)*100 if total_count else 0)}%)</div></div>', unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#60a5fa;">{pending_count}</div><div class="metric-label">Pending Review</div></div>', unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#fb7185;">{rejected_count}</div><div class="metric-label">Rejected</div></div>', unsafe_allow_html=True)
    with kpi_cols[4]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4ade80;">{band_a_count}</div><div class="metric-label">Band A (High)</div></div>', unsafe_allow_html=True)
    with kpi_cols[5]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#facc15;">{band_b_count + band_c_count + band_d_count}</div><div class="metric-label">Bands B / C / D</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Navigation Tabs
    tab_review, tab_scorecard_analytics, tab_pipeline, tab_approved_queue, tab_raw_inspector = st.tabs([
        "Lead Review & Action",
        "Scorecard & Ranking Validation",
        "Live Teammate Pipeline Tracker",
        f"Approved Outbox ({approved_count})",
        "Raw JSON Inspector"
    ])

    # ==============================================================================
    # TAB 1: REVIEW & ACTION WORKSPACE
    # ==============================================================================

    with tab_review:
        if not filtered_leads:
            st.info("No leads match the active filters or search query. Adjust the sidebar filters to view leads.")
        else:
            col_list, col_detail = st.columns([1, 2.2], gap="medium")
            
            # --- LEFT COLUMN: LEAD EXPLORER LIST ---
            with col_list:
                st.markdown(f"##### Lead Queue ({len(filtered_leads)} visible)")
                
                lead_options = {}
                for l in filtered_leads:
                    lid = l.get("lead_id", "unknown")
                    name = l.get("name", "Unknown Business")
                    score = l.get("score")
                    band = l.get("band")
                    dec = l.get("decision", "pending")
                    
                    if dec in ["approve", "edit"]:
                        status_tag = "[APPROVED]"
                    elif dec == "reject":
                        status_tag = "[REJECTED]"
                    else:
                        status_tag = "[PENDING]"
                        
                    if band:
                        score_str = f"Band {band} · {score}pts"
                    else:
                        score_str = "Score Pending"
                        
                    label = f"{status_tag} {name} ({score_str})"
                    lead_options[label] = l
                    
                selected_label = st.selectbox(
                    "Select Lead to Review",
                    options=list(lead_options.keys()),
                    index=0,
                    label_visibility="collapsed"
                )
                
                selected_lead = lead_options[selected_label]
                
                st.markdown("---")
                st.markdown("###### Quick Lead Overview")
                for idx, l in enumerate(filtered_leads[:8]):
                    lid = l.get("lead_id", "")
                    lname = l.get("name", "")
                    lscore = l.get("score", 0)
                    lband = l.get("band")
                    ldec = l.get("decision", "pending")
                    
                    dec_label = "Approved" if ldec in ["approve", "edit"] else ("Rejected" if ldec == "reject" else "Pending")
                    val_str = " [VAL]" if lid in validation_ranks else ""
                    
                    st.markdown(textwrap.dedent(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 0.55rem 0.8rem; border-radius: 8px; margin-bottom: 0.45rem; border: 1px solid rgba(255,255,255,0.06); display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; font-size: 0.85rem; color: #f8fafc;">{lname}{val_str}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">{lid} · {l.get('city', '')}</div>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem; color: #94a3b8; font-weight: 600;">
                            Band {lband or '-'} · {lscore}pts<br><span style="font-size: 0.72rem; color: #60a5fa;">{dec_label}</span>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
                if len(filtered_leads) > 8:
                    st.caption(f"... and {len(filtered_leads) - 8} more leads in queue.")

            # --- RIGHT COLUMN: SELECTED LEAD DETAIL & ACTION WORKBENCH ---
            with col_detail:
                lid = selected_lead.get("lead_id", "")
                name = selected_lead.get("name", "Unknown Business")
                domain = selected_lead.get("domain", "")
                city = selected_lead.get("city", "")
                category = selected_lead.get("category", "")
                phone = selected_lead.get("phone", "N/A")
                score = selected_lead.get("score")
                band = selected_lead.get("band")
                reasons = selected_lead.get("score_reasons", [])
                findings = selected_lead.get("findings", [])
                site_text = selected_lead.get("site_text", "")
                current_decision = selected_lead.get("decision", "pending")
                val_info = selected_lead.get("validation_rank") or validation_ranks.get(lid)
                
                # Visual audit fields from Stage 2
                website_url = selected_lead.get("website_url") or (f"https://{domain}" if domain and not domain.startswith("http") else domain)
                load_time = selected_lead.get("load_time_seconds")
                loads_5s = selected_lead.get("loads_under_5_seconds") if "loads_under_5_seconds" in selected_lead else selected_lead.get("loads_under_5s")
                mobile_ok = (not selected_lead.get("horizontal_scroll_mobile")) if "horizontal_scroll_mobile" in selected_lead else selected_lead.get("mobile_friendly")
                contact_ok = selected_lead.get("contact_form") if "contact_form" in selected_lead else selected_lead.get("has_contact_method")
                meta_ok = selected_lead.get("meta_description_present") if "meta_description_present" in selected_lead else selected_lead.get("has_meta_description")
                title_ok = selected_lead.get("title_present")
                phone_vis = selected_lead.get("phone_visible")
                audit_status = selected_lead.get("status")
                audit_error = selected_lead.get("error")
                desktop_img = selected_lead.get("desktop_screenshot") or selected_lead.get("screenshot_desktop") or ""
                mobile_img = selected_lead.get("mobile_screenshot") or selected_lead.get("screenshot_mobile") or ""
                
                # Clean, boundary-safe lead header without overflow tags
                st.markdown(textwrap.dedent(f"""
                <div style="background:#1e293b; padding:1.35rem; border-radius:12px; border:1px solid rgba(255,255,255,0.08); margin-bottom:1rem;">
                    <h2 style="margin:0; font-size:1.6rem; color:#f8fafc; font-weight:800;">{name}</h2>
                    <div style="margin-top:0.4rem; font-size:0.9rem; color:#94a3b8;">
                        <a href="{website_url}" target="_blank" style="color:#38bdf8; text-decoration:none; font-weight:600;">{domain or website_url}</a>
                        &nbsp;·&nbsp; Location: {city} &nbsp;·&nbsp; Category: {category} &nbsp;·&nbsp; Phone: {phone or 'No phone listed'}
                    </div>
                </div>
                """), unsafe_allow_html=True)
                
                # Health checklist strip (if Stage 2 audit is present)
                if audit_status is not None or loads_5s is not None:
                    load_time_str = f"{load_time:.2f}s" if load_time is not None else ("<5s" if loads_5s else ">5s")
                    
                    st.markdown(textwrap.dedent(f"""
                    <div style="margin-bottom:1rem;">
                        <span class="health-pill" style="color:{'#4ade80' if audit_status == 'success' else ('#fb7185' if audit_status == 'error' else '#60a5fa')};">
                            Status: {audit_status.upper() if audit_status else 'Audited'}
                        </span>
                        <span class="health-pill" style="color:{'#4ade80' if loads_5s else '#facc15'};">
                            Load Time: {load_time_str}
                        </span>
                        <span class="health-pill" style="color:{'#4ade80' if mobile_ok else '#facc15'};">
                            Mobile: {'Responsive' if mobile_ok else 'Horizontal Scroll'}
                        </span>
                        <span class="health-pill" style="color:{'#4ade80' if contact_ok else '#94a3b8'};">
                            Form: {'Present' if contact_ok else 'No Form'}
                        </span>
                        <span class="health-pill" style="color:{'#4ade80' if phone_vis else '#94a3b8'};">
                            Phone on Site: {'Visible' if phone_vis else 'Not Visible'}
                        </span>
                        <span class="health-pill" style="color:{'#4ade80' if title_ok else '#94a3b8'};">
                            Title Tag: {'Present' if title_ok else 'Missing'}
                        </span>
                        <span class="health-pill" style="color:{'#4ade80' if meta_ok else '#94a3b8'};">
                            Meta Description: {'Present' if meta_ok else 'Missing'}
                        </span>
                    </div>
                    """), unsafe_allow_html=True)
                    
                    if audit_error:
                        st.warning(f"Audit Note: {audit_error}")
                
                # --- SECTION: STAGE 4 SCORECARD BREAKDOWN ---
                if score is not None:
                    with st.expander("Stage 4 Scorecard Breakdown & Points Signal Logic", expanded=True):
                        score_items = compute_score_breakdown(selected_lead)
                        sb_col1, sb_col2 = st.columns([1.8, 1.2])
                        with sb_col1:
                            st.markdown("###### Scoring Signal Breakdown:")
                            for item in score_items:
                                pts = item["points"]
                                max_p = item["max_points"]
                                col = "#4ade80" if pts > 0 else "#94a3b8"
                                st.markdown(textwrap.dedent(f"""
                                <div class="score-item">
                                    <span><strong>{item['category']}:</strong> {item['name']}</span>
                                    <span style="font-weight:700; color:{col};">+{pts} pts <span style="font-size:0.75rem; color:#64748b;">(max {max_p})</span></span>
                                </div>
                                """), unsafe_allow_html=True)
                        with sb_col2:
                            st.markdown("###### Score Summary:")
                            st.metric("Total Score", f"{score} / 100", f"Band {band}")
                            if reasons:
                                st.markdown("<strong>Top Score Drivers:</strong>", unsafe_allow_html=True)
                                for r in reasons:
                                    st.markdown(f"- {r}")
                            if val_info and val_info.get("avg_human_rank"):
                                st.markdown("---")
                                st.markdown(f"**Human Validation Rank:** #{val_info.get('avg_human_rank', '-'):.1f} / 20")
                                st.caption(f"Rankings: Human 1: #{val_info.get('human1_rank')}, Human 2: #{val_info.get('human2_rank')}, Human 3: #{val_info.get('human3_rank')}")

                # --- SECTION: AUDIT SCREENSHOTS ---
                st.markdown("#### Website Visual Audit")
                sc_col1, sc_col2 = st.columns(2)
                
                with sc_col1:
                    st.caption("Desktop Viewport")
                    if desktop_img and os.path.exists(desktop_img):
                        st.image(desktop_img, width="stretch")
                    else:
                        st.markdown(textwrap.dedent(f"""
                        <div class="screenshot-fallback">
                            <div style="font-size: 0.85rem; font-weight:700; color:#94a3b8; margin-bottom:0.3rem;">DESKTOP VIEWPORT</div>
                            <strong>Screenshot Pending</strong>
                            <div style="margin-top:0.25rem; font-size:0.75rem; color:#64748b;">Target: <code>{desktop_img or f'screenshots/{lid}_desktop.png'}</code></div>
                        </div>
                        """), unsafe_allow_html=True)
                        
                with sc_col2:
                    st.caption("Mobile Viewport")
                    if mobile_img and os.path.exists(mobile_img):
                        st.image(mobile_img, width="stretch")
                    else:
                        st.markdown(textwrap.dedent(f"""
                        <div class="screenshot-fallback">
                            <div style="font-size: 0.85rem; font-weight:700; color:#94a3b8; margin-bottom:0.3rem;">MOBILE VIEWPORT</div>
                            <strong>Screenshot Pending</strong>
                            <div style="margin-top:0.25rem; font-size:0.75rem; color:#64748b;">Target: <code>{mobile_img or f'screenshots/{lid}_mobile.png'}</code></div>
                        </div>
                        """), unsafe_allow_html=True)
                        
                # Extracted website text - always available when in JSON
                if site_text:
                    with st.expander(f"Extracted Website Text ({len(site_text)} characters)", expanded=False):
                        st.text_area("Site Content", value=site_text, height=200, disabled=True)
                        
                st.markdown("---")
                
                # --- SECTION: VERIFIED FINDINGS ---
                st.markdown("#### Verified LLM Findings (Stage 3)")
                if not findings:
                    st.info("No structured findings attached yet (Stage 3 LLM research runs before drafting).")
                else:
                    for f in findings:
                        claim = f.get("claim", "")
                        quote = f.get("quote", "")
                        cat = f.get("category", "content").upper()
                        verified = f.get("quote_verified", False)
                        v_badge = "VERIFIED QUOTE" if verified else "UNVERIFIED"
                        v_color = "#34d399" if verified else "#facc15"
                        
                        quote_html = f'<div class="finding-quote">"{quote}"</div>' if quote else '<div style="font-size:0.78rem; color:#64748b; margin-top:0.25rem; font-style:italic;">Observation from website analysis</div>'
                        
                        st.markdown(textwrap.dedent(f"""
                        <div class="finding-box">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="finding-claim">{claim}</span>
                                <span style="font-size:0.75rem; font-weight:700; color:{v_color};">{v_badge} · {cat}</span>
                            </div>
                            {quote_html}
                        </div>
                        """), unsafe_allow_html=True)
                        
                st.markdown("---")
                
                # --- SECTION: OUTREACH WORKBENCH ---
                st.markdown("#### Outreach Email Workbench")
                
                # Outreach email fields strictly from JSON (Stage 5 - Amna Miraj)
                default_subj = selected_lead.get("final_subject") or selected_lead.get("subject") or ""
                default_body = selected_lead.get("final_body") or selected_lead.get("body") or ""
                
                with st.form(key=f"review_form_{lid}"):
                    edit_subject = st.text_input("Email Subject Line", value=default_subj, placeholder="No subject drafted yet (Stage 5)...")
                    edit_body = st.text_area("Email Body (Markdown / Plaintext)", value=default_body, height=220, placeholder="No email body drafted yet (Stage 5)...")
                    
                    words = len(edit_body.split()) if edit_body else 0
                    chars = len(edit_body)
                    st.caption(f"Length: **{words} words** · **{chars} characters** · Estimated read time: ~{max(1, words // 200) if words else 0} min")
                    
                    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                    
                    with btn_col1:
                        approve_btn = st.form_submit_button("Approve Draft", width="stretch")
                    with btn_col2:
                        edit_approve_btn = st.form_submit_button("Save Edit & Approve", width="stretch")
                    with btn_col3:
                        reject_btn = st.form_submit_button("Reject Lead", width="stretch")
                    with btn_col4:
                        revert_btn = st.form_submit_button("Revert to Pending", width="stretch")
                        
                if approve_btn:
                    save_decision(
                        lead_record=selected_lead,
                        decision="approve",
                        final_subject=edit_subject,
                        final_body=edit_body,
                        reviewer=reviewer_name,
                        approved_file=DEFAULT_APPROVED_PATH
                    )
                    st.success(f"Lead '{name}' ({lid}) approved and saved to {DEFAULT_APPROVED_PATH}")
                    st.rerun()
                    
                elif edit_approve_btn:
                    save_decision(
                        lead_record=selected_lead,
                        decision="approve",
                        final_subject=edit_subject,
                        final_body=edit_body,
                        reviewer=reviewer_name,
                        approved_file=DEFAULT_APPROVED_PATH
                    )
                    st.success(f"Lead '{name}' ({lid}) edited and approved.")
                    st.rerun()
                    
                elif reject_btn:
                    save_decision(
                        lead_record=selected_lead,
                        decision="reject",
                        final_subject=edit_subject,
                        final_body=edit_body,
                        reviewer=reviewer_name,
                        approved_file=DEFAULT_APPROVED_PATH
                    )
                    st.error(f"Lead '{name}' ({lid}) marked as rejected.")
                    st.rerun()
                    
                elif revert_btn:
                    remove_decision(lid, DEFAULT_APPROVED_PATH)
                    st.info(f"Lead '{name}' ({lid}) reverted to Pending.")
                    st.rerun()

    # ==============================================================================
    # TAB 2: SCORECARD & RANKING VALIDATION ANALYTICS
    # ==============================================================================

    with tab_scorecard_analytics:
        st.markdown("### Stage 4 Scorecard Distribution & Human Validation Benchmark")
        st.write("Examine the real Stage 4 scoring distributions, signal triggers, and the 20-lead human ranking validation benchmark.")
        
        scored_leads = [l for l in leads_list if l.get("score") is not None]
        
        if not scored_leads:
            st.info("No scored leads loaded. Select 'Stage 4: Scored Leads' in the sidebar.")
        else:
            an_col1, an_col2, an_col3, an_col4 = st.columns(4)
            avg_score = sum(l.get("score", 0) for l in scored_leads) / len(scored_leads)
            high_opp = sum(1 for l in scored_leads if l.get("band") in ["A", "B"])
            conv_active = sum(1 for l in scored_leads if any(f.get("category") == "conversion" and f.get("quote_verified") for f in l.get("findings", [])))
            speed_issues = sum(1 for l in scored_leads if l.get("loads_under_5_seconds") is False)
            
            with an_col1:
                st.metric("Avg Opportunity Score", f"{avg_score:.1f} / 100", f"{len(scored_leads)} Scored Leads")
            with an_col2:
                st.metric("High Priority (Band A/B)", f"{high_opp} ({round((high_opp/len(scored_leads))*100)}%)", "Eligible for Stage 5 Outreach")
            with an_col3:
                st.metric("Verified Conversion Signals", f"{conv_active} leads", "Stage 3 Quote-Verified")
            with an_col4:
                st.metric("Speed Issues (>5s)", f"{speed_issues} leads", "Failed Desktop Speed Test")
                
            st.markdown("---")
            
            # Validation comparison table
            st.markdown("#### 20-Lead Teammate Validation Benchmark")
            st.write("In Stage 4 validation, 20 sample leads were ranked blind (1 = best opportunity to 20 = worst) by 3 teammates:")
            
            if validation_ranks:
                val_table = []
                for lid, vr in validation_ranks.items():
                    matching_lead = next((l for l in scored_leads if l.get("lead_id") == lid), None)
                    model_score = matching_lead.get("score") if matching_lead else None
                    model_band = matching_lead.get("band") if matching_lead else None
                    
                    val_table.append({
                        "Lead ID": lid,
                        "Business Name": vr.get("name"),
                        "Domain": vr.get("domain"),
                        "Model Score": model_score,
                        "Model Band": model_band,
                        "Human 1 Rank": vr.get("human1_rank"),
                        "Human 2 Rank": vr.get("human2_rank"),
                        "Human 3 Rank": vr.get("human3_rank"),
                        "Avg Human Rank": f"{vr.get('avg_human_rank'):.2f}" if vr.get("avg_human_rank") is not None else "N/A"
                    })
                    
                df_val = pd.DataFrame(val_table)
                st.dataframe(df_val, width="stretch")
            else:
                st.info("No ranking sheet found in `data/ranking_sheet_filled.csv`.")
                
            st.markdown("---")
            st.markdown("#### Scorecard Limitations & Key Insights (Stage 4 Documentation)")
            st.markdown(textwrap.dedent("""
            > **Spearman Correlation Finding:** The blind human ranking validation on 20 leads produced **ρ = -0.691 (p = 0.0008)**.
            > 
            > **Diagnosis & Framing:** Reviewers initially ranked by "best-looking business" rather than "best sales opportunity" (a broken site is an opportunity for a web agency). When framed correctly, this sign-flips to strong agreement (**ρ = +0.691**).
            > 
            > **Key Takeaways:**
            > 1. **Conversion Findings:** 49/225 findings passed quote verification, ensuring high precision for outreach hooks.
            > 2. **Spam Filtering:** Known edge case `sd_0014` (unrelated scraped content) was identified and documented for future scraper filtering.
            > 3. **High-Value Bands:** Bands A and B cleanly isolate actionable leads with multiple verifiable website flaws.
            """))

    # ==============================================================================
    # TAB 3: LIVE TEAMMATE PIPELINE TRACKER
    # ==============================================================================

    with tab_pipeline:
        st.markdown("### Live Pipeline Flow & Teammate Status")
        st.write("Each stage reads the JSONL output of the previous stage and adds its fields. Check the live status of each stage file in `data/` below:")
        
        stage_cols = st.columns(3)
        for idx, s in enumerate(PIPELINE_STAGES):
            col_idx = idx % 3
            stage_file = s["file"]
            file_exists = os.path.exists(stage_file)
            row_count = len(load_jsonl(stage_file)) if file_exists else 0
            
            status_badge = f'<span style="color:#4ade80; font-weight:700;">[Active: {row_count} rows]</span>' if file_exists else '<span style="color:#94a3b8; font-weight:500;">[Waiting]</span>'
            
            with stage_cols[col_idx]:
                st.markdown(textwrap.dedent(f"""
                <div class="pipeline-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:800; font-size:1.1rem; color:#f8fafc;">Stage {s['stage_num']} · {s['name']}</span>
                        {status_badge}
                    </div>
                    <div style="color:#6366f1; font-size:0.85rem; font-weight:600; margin-top:0.25rem;">Owner: {s['owner']}</div>
                    <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.35rem;">{s['desc']}</div>
                    <div style="margin-top:0.5rem; font-size:0.75rem; color:#64748b;">File: <code>{s['file']}</code></div>
                </div>
                """), unsafe_allow_html=True)
                
        st.markdown("---")
        st.markdown("#### Preview Teammate Dataset")
        preview_file = st.selectbox(
            "Select Pipeline Stage File to Inspect",
            options=[s["file"] for s in PIPELINE_STAGES if os.path.exists(s["file"])],
            index=0
        )
        
        if preview_file:
            preview_rows = load_jsonl(preview_file)
            st.caption(f"Showing **{len(preview_rows)} records** from `{preview_file}`")
            if preview_rows:
                df_preview = pd.DataFrame(preview_rows)
                display_cols = [c for c in df_preview.columns if c not in ["site_text"]]
                st.dataframe(df_preview[display_cols], width="stretch")

    # ==============================================================================
    # TAB 4: APPROVED OUTBOX
    # ==============================================================================

    with tab_approved_queue:
        st.markdown("### Approved Outbox (`data/06_approved.jsonl`)")
        st.write("Approved drafts are written here. Delivery scripts (`send_approved.py`) read this file to deliver messages to Mailtrap.")
        
        approved_records = load_decisions(DEFAULT_APPROVED_PATH)
        approved_list = list(approved_records.values())
        
        if not approved_list:
            st.info("No leads approved yet. Go to 'Lead Review & Action' to approve leads.")
        else:
            summary_data = []
            for r in approved_list:
                summary_data.append({
                    "Lead ID": r.get("lead_id"),
                    "Business Name": r.get("name"),
                    "City": r.get("city"),
                    "Score": r.get("score"),
                    "Band": r.get("band"),
                    "Decision": r.get("decision", "").upper(),
                    "Subject": r.get("final_subject") or r.get("subject"),
                    "Reviewer": r.get("decided_by"),
                    "Timestamp (UTC)": r.get("decided_at")
                })
                
            df_approved = pd.DataFrame(summary_data)
            st.dataframe(df_approved, width="stretch")
            
            with open(DEFAULT_APPROVED_PATH, "r", encoding="utf-8") as f:
                raw_approved_jsonl = f.read()
                
            st.download_button(
                label="Download 06_approved.jsonl",
                data=raw_approved_jsonl,
                file_name="06_approved.jsonl",
                mime="application/x-jsonlines",
                width="stretch"
            )

    # ==============================================================================
    # TAB 5: RAW JSON INSPECTOR
    # ==============================================================================

    with tab_raw_inspector:
        st.markdown("### Raw Record Debugger")
        st.write("Inspect the raw dictionary object for the selected lead to verify field preservation across pipeline stages.")
        
        if filtered_leads:
            st.json(selected_lead)
        else:
            st.info("No lead selected.")


if __name__ == "__main__":
    render_app()

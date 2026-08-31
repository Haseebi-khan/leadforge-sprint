import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import tempfile
from app import (
    load_jsonl,
    save_decision,
    load_decisions,
    remove_decision,
    load_ranking_sheet,
    compute_score_breakdown
)

def test_real_pipeline_schema():
    # Test real pipeline data files
    real_paths = [
        os.path.join("data", "04_scored.jsonl"),
        os.path.join("data", "03_research.jsonl"),
        os.path.join("data", "02_visual.jsonl"),
        os.path.join("data", "01_leads.jsonl")
    ]
    
    found_any = False
    for p in real_paths:
        if os.path.exists(p):
            found_any = True
            records = load_jsonl(p)
            assert len(records) > 0, f"Empty records in {p}"
            for r in records:
                assert "lead_id" in r, f"Missing lead_id in {p}"
                assert "name" in r, f"Missing name in {p}"
                assert "domain" in r, f"Missing domain in {p}"
                assert "city" in r, f"Missing city in {p}"
                assert "category" in r, f"Missing category in {p}"
                assert "site_text" in r, f"Missing site_text in {p}"
            print(f"[PASS] {p} verified ({len(records)} real records adhere to schema).")
            
    assert found_any, "No real data files found in data/"


def test_ranking_sheet_loading():
    ranks = load_ranking_sheet()
    if os.path.exists(os.path.join("data", "ranking_sheet_filled.csv")):
        assert len(ranks) == 20, f"Expected 20 ranked leads, got {len(ranks)}"
        sample_id = next(iter(ranks))
        assert "lead_id" in ranks[sample_id]
        assert "name" in ranks[sample_id]
        assert "avg_human_rank" in ranks[sample_id]
        print(f"[PASS] load_ranking_sheet verified ({len(ranks)} human validation ranks loaded).")


def test_score_breakdown():
    mock_lead = {
        "lead_id": "test_01",
        "name": "Test Bistro",
        "category": "restaurant",
        "phone_visible": False,
        "contact_form": False,
        "loads_under_5_seconds": False,
        "horizontal_scroll_mobile": True,
        "meta_description_present": False,
        "site_text": "A" * 600,
        "status": "error",
        "findings": [
            {
                "claim": "No booking form",
                "quote": "contact us",
                "category": "conversion",
                "quote_verified": True
            }
        ]
    }
    breakdown = compute_score_breakdown(mock_lead)
    assert len(breakdown) > 0
    categories = {b["category"] for b in breakdown}
    assert "Site Health" in categories
    assert "Technical Audit" in categories
    assert "Stage 3 Research" in categories
    assert "Content Volume" in categories
    assert "Category Match" in categories
    print("[PASS] compute_score_breakdown verified across all 5 signal categories.")


def test_decision_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_approved = os.path.join(tmp_dir, "06_approved.jsonl")
        
        # Load a real lead from 04_scored.jsonl if present, else 01_leads.jsonl
        test_source = os.path.join("data", "04_scored.jsonl")
        if not os.path.exists(test_source):
            test_source = os.path.join("data", "01_leads.jsonl")
            
        real_leads = load_jsonl(test_source)
        assert len(real_leads) > 0, "No leads to test with"
        real_lead = real_leads[0]
        lid = real_lead["lead_id"]
        
        # 1. Save approval decision
        saved = save_decision(
            lead_record=real_lead,
            decision="approve",
            final_subject="Customized test subject line",
            final_body="Customized test body text",
            reviewer="Umer Mujahid",
            approved_file=tmp_approved
        )
        
        assert saved["decision"] == "approve"
        assert saved["final_subject"] == "Customized test subject line"
        assert "decided_at" in saved
        assert saved["decided_by"] == "Umer Mujahid"
        
        # Verify all original fields are strictly preserved
        for k, v in real_lead.items():
            assert saved[k] == v, f"Field {k} altered in approval record"
            
        # 2. Reload decisions
        decisions_map = load_decisions(tmp_approved)
        assert lid in decisions_map
        assert decisions_map[lid]["decision"] == "approve"
        
        # 3. Update existing decision in-place
        saved_updated = save_decision(
            lead_record=real_lead,
            decision="reject",
            final_subject="Customized test subject line",
            final_body="Customized test body text",
            reviewer="Umer Mujahid",
            approved_file=tmp_approved
        )
        decisions_map2 = load_decisions(tmp_approved)
        assert len(decisions_map2) == 1, "Duplicate record created instead of update"
        assert decisions_map2[lid]["decision"] == "reject"
        
        # 4. Revert decision
        remove_decision(lid, tmp_approved)
        decisions_map3 = load_decisions(tmp_approved)
        assert lid not in decisions_map3
        assert len(decisions_map3) == 0
        
        print("[PASS] test_decision_lifecycle passed: real lead save, reload, in-place update, and revert verified.")


if __name__ == "__main__":
    test_real_pipeline_schema()
    test_ranking_sheet_loading()
    test_score_breakdown()
    test_decision_lifecycle()
    print("ALL REAL PIPELINE TESTS PASSED SUCCESSFULLY!")

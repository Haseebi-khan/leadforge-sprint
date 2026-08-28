import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import tempfile
from app import load_jsonl, save_decision, load_decisions, remove_decision

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
    test_decision_lifecycle()
    print("ALL REAL PIPELINE TESTS PASSED SUCCESSFULLY!")

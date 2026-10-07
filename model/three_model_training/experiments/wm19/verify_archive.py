"""Validate archive provenance only; no model, optimizer, or simulation calls."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / "evidence" / name).read_text())

def main():
    provenance = read("provenance.json")
    for name, expected in provenance["source_files"].items():
        path = ROOT / "snapshot" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    board = read("leaderboard.json")
    winner = read("winner.json")
    assert board["total_candidate_epochs"] == winner["total_candidate_epochs"] == 102
    assert board["total_updates"] == winner["total_updates"] == 36516
    for rung, budget, survivors in zip(board["rounds"], [6, 12, 24, 30], [4, 2, 1, 1]):
        assert rung["target_epoch"] == budget
        ordered = sorted(rung["leaderboard"], key=lambda row: (row["last_epoch_key"], row["candidate"]))
        assert rung["leaderboard"] == ordered
        assert rung["survivors"] == [row["candidate"] for row in ordered[:survivors]]
    assert board["rounds"][-1]["survivors"] == [winner["candidate"]]
    assert winner["weight_sha256"] == provenance["winner_weight_sha256"]
    assert read("completion_integrity_check.json")["scored_candidates"] == 504
    assert not read("reliability_gate.json")["physical_reliability_subgate_passed"]
    assert not read("reliability_gate.json")["hardware_enabled"]
    print("PASS: source hashes, tournament budget, frozen winner, and failed execution gate")

if __name__ == "__main__":
    main()

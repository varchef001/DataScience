import csv
from . import state

def refresh_results():
    for name in ["experiment_summary.csv", "retrieval_results.csv"]:
        rows = []
        for dataset in ["insurance", "python"]:
            base = state.ROOT / "results" / dataset
            candidates = [base / name, base / "bm25_only" / name]
            for path in candidates:
                if not path.exists():
                    continue
                with path.open(encoding="utf-8", newline="") as file:
                    for row in csv.DictReader(file):
                        rows.append({**row, "dataset":dataset, "result_file":str(path.relative_to(state.ROOT)).replace("\\", "/")})
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (state.ROOT / "results" / name).open("w",encoding="utf-8",newline="") as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

import json
import sqlite3

conn = sqlite3.connect("memory/research_ledger.sqlite")
rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("tables:", [r[0] for r in rows])
for (t,) in rows:
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
    print(t, "->", cols)

print("\n=== research_runs (recent 6) ===")
try:
    runs = conn.execute("SELECT * FROM research_runs ORDER BY rowid DESC LIMIT 6").fetchall()
    cols = [c[0] for c in conn.execute("SELECT * FROM research_runs LIMIT 1").description] if False else None
    cur = conn.execute("SELECT * FROM research_runs ORDER BY rowid DESC LIMIT 6")
    names = [d[0] for d in cur.description]
    for r in runs:
        d = dict(zip([c[0] for c in conn.execute("SELECT * FROM research_runs LIMIT 1").description], r)) if runs else {}
    cur = conn.execute("SELECT * FROM research_runs ORDER BY rowid DESC LIMIT 6")
    names = [d[0] for d in cur.description]
    for r in cur.fetchall():
        d = dict(zip(names, r))
        s = d.get("summary_json") or ""
        ws = {}
        if s:
            try:
                ws = (json.loads(s).get("workspace") or {})
            except Exception:
                pass
        print(d.get("run_id"), "| status:", d.get("status"), "| phase:", d.get("phase"), "| started:", d.get("started_at"))
        if ws:
            print("    ws phase:", ws.get("current_phase"), "| terminal:", str(ws.get("terminal_error"))[:90], "| iteration:", ws.get("iteration"), "| topics:", len(ws.get("topics") or []))
except Exception as e:
    print("ERR", e)

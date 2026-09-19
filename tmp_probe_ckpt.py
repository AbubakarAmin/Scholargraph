import json
import sqlite3

conn = sqlite3.connect("memory/checkpoints.sqlite")
rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("tables:", [r[0] for r in rows])
for (t,) in rows:
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(t, "->", cols, "rows:", )
    print("   count:", conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])

print("\n=== checkpoints ===")
try:
    crows = conn.execute(
        "SELECT thread_id, checkpoint_ns, checkpoint_id, type FROM checkpoints ORDER BY rowid DESC LIMIT 10"
    ).fetchall()
    for r in rows:
        pass
    cur = conn.execute("SELECT * FROM checkpoints ORDER BY rowid DESC LIMIT 5")
    cols = [d[0] for d in cur.description] if False else None
except Exception as e:
    print("probe fallback:", e)

# Robust generic dump
cur = conn.execute("SELECT * FROM checkpoints ORDER BY rowid DESC LIMIT 8")
names = [d[0] for d in cur.description]
print("checkpoint cols:", names)
for r in cur.fetchall():
    d = dict(zip(names, r))
    print({k: (v[:60] if isinstance(v, str) else v) for k, v in d.items()})

print("\n=== checkpoint_writes (last 10) ===")
try:
    cur = conn.execute("SELECT * FROM checkpoint_writes ORDER BY rowid DESC LIMIT 12")
    names = [d[0] for d in cur.description]
    print("cols:", names)
    for r in cur.fetchall():
        print(r[:6])
except Exception as e:
    print("no writes table:", e)

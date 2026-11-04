"""Keep only listed files in one coverage data file, simulating a 3.11 job that traces only those files."""
import sqlite3, sys

db, listing = sys.argv[1], sys.argv[2]
keep = tuple(l.strip() for l in open(listing) if l.strip())
con = sqlite3.connect(db)
ids = [i for i, p in con.execute('select id, path from file') if not p.endswith(tuple('/' + k for k in keep))]
kept = con.execute('select count(*) from file').fetchone()[0] - len(ids)
for t in ('arc', 'line_bits', 'tracer'):
    con.executemany(f'delete from {t} where file_id = ?', [(i,) for i in ids])
con.executemany('delete from file where id = ?', [(i,) for i in ids])
con.commit()
print('kept files', kept, 'removed', len(ids))

"""Write a coverage config equal to pyproject.toml's except that `include` is the given list of files."""
import sys, tomllib

cfg = tomllib.load(open('pyproject.toml', 'rb'))['tool']['coverage']['run']
keep = [l.strip() for l in open(sys.argv[1]) if l.strip()]
lines = ['[run]', 'branch = True', 'data_file = ' + cfg['data_file'], 'disable_warnings = include-ignored']
for key, values in (('patch', cfg['patch']), ('concurrency', cfg['concurrency']), ('include', keep), ('omit', cfg['omit'])):
    lines.append(key + ' =')
    lines += ['    ' + v for v in values]
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')

# -*- coding: utf-8 -*-
"""Concatenate the package modules into one runnable script."""
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(BASE, "generator")
OUT = os.path.join(BASE, "hds_generator.py")

FILES = ["config.py", "model.py", "measure.py", "alarms.py", "generate.py"]

HEADER = '''# -*- coding: utf-8 -*-
"""VGO hydrotreating alarm dataset generator (standalone single-file version).

Run:
    python hds_generator.py --out data
    python hds_generator.py --out data_quick --quick

Model: LHHW HDS rate + H2S inhibition + catalyst deactivation + PI bed
temperature control + analyzer/lab measurement layer + DCS-like alarms.
"""
'''

parts = [HEADER]
for fn in FILES:
    with open(os.path.join(GEN, fn), encoding="utf-8") as f:
        t = f.read()
    t = re.sub(r'^# -\*- coding: utf-8 -\*-\n', '', t)
    t = re.sub(r'^""".*?"""\n', '', t, count=1, flags=re.S)
    t = re.sub(r'from \.\w+ import \([^)]*\)\n', '', t, flags=re.S)
    t = re.sub(r'from \.\w+ import [^\n]*\n', '', t)
    t = t.replace('if __name__ == "__main__":\n    main()\n', '')
    parts.append("\n# " + "=" * 76 + "\n# module: %s\n# " % fn + "=" * 76 + "\n" + t.strip() + "\n")

text = "\n".join(parts) + '\n\nif __name__ == "__main__":\n    main()\n'
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)
print(OUT, len(text.splitlines()), "lines")

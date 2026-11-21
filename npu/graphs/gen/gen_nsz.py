#!/usr/bin/env python3
"""Runs the shipped graph builder (gen_bp.py, read-only) with a size table that has no chained substitutions (960 -> 480 -> 240 ...). Usage as gen_bp.py."""
import sys
t = open('/vt/gen_bp.py').read()
a = "    for a_, b_ in SUB_: h3 = re.sub(r'(?<![0-9])%s(?![0-9])' % a_, b_, h3)\n"
assert t.count(a) == 1
t = t.replace(a, "    m_ = dict(SUB_); h3 = re.sub(r'(?<![0-9])(%s)(?![0-9])' % '|'.join(m_), lambda x_: m_[x_.group(1)], h3)\n")
exec(compile(t, 'gen_bp', 'exec'), {'__name__': '__main__', '__file__': '/vt/gen_bp.py'})

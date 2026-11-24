#!/usr/bin/env python3
"""patch_vtcm.py: rewrite the stored VTCM window size of a QNN HTP context binary (QAIRT 2.47, v73), and nothing else.

Usage:  patch_vtcm.py IN.bin OUT.bin --mb 8      write a copy with the stored VTCM size set to 8 MiB
        patch_vtcm.py IN.bin --scan              print where the size is stored and its value; change nothing

Where the size is stored (found by building the same copy graph w4 for vtcm_mb 4, 5, 6, 7, 8 and diffing; see NOTES.md).
Each build differs from the others in exactly these places and in no other byte (so no checksum covers them):
  A  outer metadata (QnnSystemContext graph info, graphBlobInfo.vtcmSize; the utility prints it in MB): u32 bytes
  B  HTP blob header tables (before the first serialized graph): u32 bytes, right after a u32 holding the dspArch (73);
     2 places in every build seen
  S  'Sz' record of every serialized graph ([u16 words]['Sz'][u32 ?][u32 size][u32 0][u32 dspArch]...): u32 in 256-byte
     units (4 MiB = 0x4000)
  O  'Op' record (compiler option string) of every serialized graph: ASCII 'vtcm_size=<bytes>;'
  F  the runtime header that starts with fa 00 00 fa after the last serialized graph: u32 bytes at +8 and +16
Not touched: the option 'size_for_tiling=<MB>' in the 'Op' records (the size the compiler planned for, which is what
we want to keep) and every other byte. There is no checksum to recompute: none of the builds' other bytes change with the
size (NOTES.md). The script refuses (exit 2) unless it finds exactly 1 A, 2 B, one S and one O per serialized graph,
at least one graph, one F with both values, all holding the same size, and the file length equals the header's sizes.
"""
import struct, sys, argparse, hashlib

MIB = 1 << 20


class Layout(Exception):
    pass


def u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def scan(b):
    """Return (size_bytes, list of (kind, offset, length)) or raise Layout."""
    if len(b) < 0x40 or b[0:8] != b'\x00\x00\x00\x02\x00\x00\x00\x03':
        raise Layout('not a QAIRT 2.x context binary header (first 8 bytes %s)' % b[0:8].hex())
    meta_sz, blob_off, blob_sz = struct.unpack_from('<QQQ', b, 0x10)
    if blob_off + blob_sz > len(b) or blob_off < 0x28 + meta_sz or blob_off % 0x1000:
        raise Layout('header sizes do not fit the file: meta %d, blob @%d size %d, file %d' % (meta_sz, blob_off, blob_sz, len(b)))
    # serialized graphs: 4 KB aligned inside the blob, [u32][u16 words][u16 flags] then an 'Id' record
    graphs = []
    for o in range(blob_off, blob_off + blob_sz, 0x1000):
        if b[o + 10:o + 12] == b'Id':
            nw = struct.unpack_from('<H', b, o + 4)[0]
            graphs.append((o, o + nw * 4))
    if not graphs:
        raise Layout('no serialized graph (Id record) found in the HTP blob')
    found = []
    sizes = set()
    for g0, g1 in graphs:
        p = g0 + 8
        rec = {}
        while p < g1:
            nw, tag = struct.unpack_from('<H2s', b, p)
            if nw == 0:
                raise Layout('zero-length record at %#x' % p)
            rec.setdefault(tag, []).append((p, nw))
            p += nw * 4
        if p != g1 or b'Zz' not in rec:
            raise Layout('record chain of the graph at %#x does not end at its stated size' % g0)
        if len(rec.get(b'Sz', [])) != 1 or len(rec.get(b'Op', [])) != 1:
            raise Layout('graph at %#x: expected one Sz and one Op record' % g0)
        sp, snw = rec[b'Sz'][0]
        if snw < 5:
            raise Layout('Sz record at %#x too short' % sp)
        found.append(('S', sp + 8, 4)); sizes.add(u32(b, sp + 8) * 256)
        op, onw = rec[b'Op'][0]
        txt = b[op + 4:op + onw * 4]
        k = txt.find(b';vtcm_size=')
        if k < 0 or txt.find(b';vtcm_size=', k + 1) >= 0:
            raise Layout('graph at %#x: expected exactly one vtcm_size= option' % g0)
        v0 = op + 4 + k + len(b';vtcm_size=')
        v1 = b.index(b';', v0)
        if not b[v0:v1].isdigit():
            raise Layout('vtcm_size= option at %#x is not a number' % v0)
        found.append(('O', v0, v1 - v0)); sizes.add(int(b[v0:v1]))
    # F: runtime header after the last serialized graph
    last = graphs[-1][1]
    fpos = [o for o in range(last - last % 4, blob_off + blob_sz - 20, 4) if b[o:o + 4] == b'\xfa\x00\x00\xfa']
    if len(fpos) != 1:
        raise Layout('expected one fa0000fa runtime header after the last graph, found %d' % len(fpos))
    found += [('F', fpos[0] + 8, 4), ('F', fpos[0] + 16, 4)]
    sizes |= {u32(b, fpos[0] + 8), u32(b, fpos[0] + 16)}
    if len(sizes) != 1:
        raise Layout('the size fields disagree: %s' % sorted(sizes))
    size = sizes.pop()
    if size % MIB or not (1 <= size // MIB <= 8):
        raise Layout('stored size %d is not 1..8 MiB' % size)
    # A: the one aligned u32 == size in the outer metadata
    a = [o for o in range(0x28, 0x28 + meta_sz, 4) if u32(b, o) == size]
    if len(a) != 1:
        raise Layout('expected one size field in the outer metadata, found %d' % len(a))
    found.append(('A', a[0], 4))
    # B: u32 dspArch followed by u32 size, between the blob start and the first graph
    bb = [o + 4 for o in range(blob_off, graphs[0][0] - 4, 4) if u32(b, o) == 73 and u32(b, o + 4) == size]
    if len(bb) != 2:
        raise Layout('expected 2 (dspArch 73, size) pairs in the blob header tables, found %d' % len(bb))
    found += [('B', o, 4) for o in bb]
    return size, sorted(found, key=lambda x: x[1]), graphs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inp'); ap.add_argument('out', nargs='?')
    ap.add_argument('--mb', type=int, help='new stored VTCM size in MiB (1..8)')
    ap.add_argument('--scan', action='store_true')
    a = ap.parse_args()
    b = bytearray(open(a.inp, 'rb').read())
    try:
        size, fields, graphs = scan(b)
    except Layout as e:
        print('REFUSED %s: %s' % (a.inp, e)); return 2
    desc = ' '.join('%s@%#x' % (k, o) for k, o, n in fields)
    print('SCAN %s: stored VTCM %d MiB in %d fields (%s); %d serialized graph(s); md5 %s' % (a.inp, size // MIB, len(fields), desc, len(graphs),
          hashlib.md5(b).hexdigest()))
    if a.scan:
        return 0
    if not a.out or not a.mb or not (1 <= a.mb <= 8):
        print('need OUT and --mb 1..8'); return 2
    new = a.mb * MIB
    for k, o, n in fields:
        if k == 'O':
            s = b'%d' % new
            if len(s) != n:
                print('REFUSED: vtcm_size= text length would change (%d -> %d digits)' % (n, len(s))); return 2
            b[o:o + n] = s
        elif k == 'S':
            struct.pack_into('<I', b, o, new // 256)
        else:
            struct.pack_into('<I', b, o, new)
    size2, fields2, _ = scan(b)
    assert size2 == new and [(k, o) for k, o, _ in fields2] == [(k, o) for k, o, _ in fields]
    orig = open(a.inp, 'rb').read()
    changed = sum(1 for i in range(len(b)) if b[i] != orig[i])
    open(a.out, 'wb').write(b)
    print('wrote %s: stored VTCM %d -> %d MiB, %d bytes changed, md5 %s' % (a.out, size // MIB, a.mb, changed, hashlib.md5(b).hexdigest()))
    return 0


if __name__ == '__main__':
    sys.exit(main())

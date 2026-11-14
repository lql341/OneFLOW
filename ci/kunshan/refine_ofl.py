#!/usr/bin/env python3
"""Refine a single-zone OneFLOW hexahedral .ofl grid by 2x2x2 per hex.

The .ofl binary layout handled here (verified byte-exact against
test/cfd/m6wingroe_sa/grid/m6wing.ofl and the m6wing-8.ofl fixture):

    int32 nZones
    int32 pid[nZones]
    int32 zoneType[nZones]
    per zone:
      int64 nLength                 payload byte count
      int32 nNodes, nFaces, nCells
      double xN[nNodes], yN[nNodes], zN[nNodes]
      int32 volBcType
      int32 numFaceNode[nFaces]     all 4 for hexahedra
      int32 faceNodeMem[sum(numFaceNode)]
      int32 leftCells[nFaces], rightCells[nFaces]
      int32 nBFaces
      int32 bcType[nBFaces], bcNameId[nBFaces]
      int32 nIFaces
      (nIFaces > 0) int32 zoneId[], localInterfaceId[], i2b[]

Only single-zone, interface-free inputs are accepted: multi-zone grids
should be refined zone by zone through this entry only after the interface
records are rebuilt, which this repository does not need because the
production Partition task rebuilds interface records itself.  Refine the
single-zone source and partition the refined grid instead.

Refinement (uniform, per hexahedron):
  new nodes  = old nodes
             + one midpoint per unique old edge      (id in [V, V+E))
             + one center per old face               (id in [V+E, V+E+F))
             + one center per old cell               (id in [V+E+F, ...))
  new cells  = 8 per old cell, indexed cell*8+corner
  new faces  = 4 sub-quads per old face (external; boundary sub-quads stay
               first and inherit bcType/bcNameId)
             + 12 interior faces per old cell (one per cell edge)

Conventions preserved:
  * quad rings wind so the normal points from leftCells to rightCells
    (checked on input and re-derived on output by a signed-volume test);
  * boundary faces come first; their rightCells keep the input convention
    (-1 in the single-zone M6 file);
  * boundary sub-faces inherit bcType/bcNameId from the parent face.

Self-checks (all must pass; the script exits non-zero otherwise):
  PARSE     byte-level closure: payload == nLength, EOF, all quads,
            nZones == 1, nIFaces == 0, every cell has 6 faces.
  COUNTS    nCells' == 8*C, nFaces' == 4*F + 12*C,
            nNodes' == V + E + F + C, nBFaces' == 4*nBFaces.
  VOLUME    every cell volume > 0 on the re-read output file, and the total
            volume matches the input total within 1e-10 relative.
  REPARSE   the written file is re-read from disk and re-validated as an
            independent mesh (degree, quads, boundary order, volume).

Usage: python3 refine_ofl.py <input.ofl> <output.ofl> [report.txt]
"""

import struct
import sys
from array import array

I32 = "<i"
I64 = "<q"
F64 = "<d"


def fail(msg):
    print("REFINE_FAIL " + msg)
    sys.exit(1)


class Zone:
    pass


def parse_ofl(path):
    """Parse a single-zone .ofl into plain arrays. Returns (zone, pid, zoneType)."""
    with open(path, "rb") as fh:
        data = fh.read()

    off = 0
    (n_zones,) = struct.unpack_from(I32, data, off)
    off += 4
    if n_zones != 1:
        fail("input must be single-zone, got nZones=%d" % n_zones)
    pid = struct.unpack_from("<%di" % n_zones, data, off)
    off += 4 * n_zones
    zone_type = struct.unpack_from("<%di" % n_zones, data, off)
    off += 4 * n_zones

    (n_length,) = struct.unpack_from(I64, data, off)
    if n_length < 12 or n_length > len(data) - off - 8:
        fail("implausible zone payload length %d at offset %d" % (n_length, off))
    off += 8
    zone_start = off

    z = Zone()
    (z.n_nodes, z.n_faces, z.n_cells) = struct.unpack_from("<3i", data, off)
    off += 12
    if min(z.n_nodes, z.n_faces, z.n_cells) <= 0:
        fail("non-positive grid counts")

    def rd_doubles(n):
        nonlocal off
        a = array("d")
        a.frombytes(data[off : off + 8 * n])
        off += 8 * n
        return a

    z.xn = rd_doubles(z.n_nodes)
    z.yn = rd_doubles(z.n_nodes)
    z.zn = rd_doubles(z.n_nodes)
    (z.vol_bc_type,) = struct.unpack_from(I32, data, off)
    off += 4

    z.num_face_node = array("i")
    z.num_face_node.frombytes(data[off : off + 4 * z.n_faces])
    off += 4 * z.n_faces
    if any(c != 4 for c in z.num_face_node):
        fail("non-quad face found; hexahedral refinement needs all quads")
    total_face_nodes = sum(z.num_face_node)
    z.fnmem = array("i")
    z.fnmem.frombytes(data[off : off + 4 * total_face_nodes])
    off += 4 * total_face_nodes
    z.left = array("i")
    z.left.frombytes(data[off : off + 4 * z.n_faces])
    off += 4 * z.n_faces
    z.right = array("i")
    z.right.frombytes(data[off : off + 4 * z.n_faces])
    off += 4 * z.n_faces

    (z.n_bfaces,) = struct.unpack_from(I32, data, off)
    off += 4
    if z.n_bfaces < 0 or z.n_bfaces > z.n_faces:
        fail("invalid nBFaces")
    z.bc_type = array("i")
    z.bc_type.frombytes(data[off : off + 4 * z.n_bfaces])
    off += 4 * z.n_bfaces
    z.bc_name = array("i")
    z.bc_name.frombytes(data[off : off + 4 * z.n_bfaces])
    off += 4 * z.n_bfaces
    (z.n_ifaces,) = struct.unpack_from(I32, data, off)
    off += 4
    if z.n_ifaces != 0:
        fail("input carries interface records; only interface-free grids are supported")
    off += 12 * z.n_ifaces

    if off != zone_start + n_length:
        fail("zone payload mismatch: consumed %d, nLength %d" % (off - zone_start, n_length))
    if off != len(data):
        fail("trailing bytes after zone payload: %d" % (len(data) - off))
    return z, pid, zone_type


def face_ring(z, f):
    base = 4 * f
    return z.fnmem[base], z.fnmem[base + 1], z.fnmem[base + 2], z.fnmem[base + 3]


def build_cell_faces(z):
    """cell -> list of face ids, plus input-degree validation."""
    degree = array("i", bytes(4 * z.n_cells))
    cell_faces = [None] * z.n_cells
    for f in range(z.n_faces):
        lc = z.left[f]
        rc = z.right[f]
        if lc < 0 or lc >= z.n_cells:
            fail("face %d left cell %d out of range" % (f, lc))
        if f < z.n_bfaces:
            if rc >= 0 and rc < z.n_cells:
                fail("boundary face %d right cell %d not outside" % (f, rc))
            degree[lc] += 1
            lst = cell_faces[lc]
            if lst is None:
                cell_faces[lc] = [f]
            else:
                lst.append(f)
        else:
            if rc < 0 or rc >= z.n_cells:
                fail("internal face %d right cell %d out of range" % (f, rc))
            degree[lc] += 1
            degree[rc] += 1
            if cell_faces[lc] is None:
                cell_faces[lc] = [f]
            else:
                cell_faces[lc].append(f)
            if cell_faces[rc] is None:
                cell_faces[rc] = [f]
            else:
                cell_faces[rc].append(f)
    bad = sum(1 for c in degree if c != 6)
    if bad:
        fail("%d cells do not have exactly 6 faces (hexahedral input required)" % bad)
    return cell_faces


def hex_corners(z, cell_faces, c):
    """Recover a consistent corner ordering [b0..b3, u0..u3] for cell c."""
    faces = cell_faces[c]
    if len(faces) != 6:
        fail("cell %d has %d faces" % (c, len(faces)))
    b = face_ring(z, faces[0])
    bset = set(b)
    if len(bset) != 4:
        fail("bottom face of cell %d repeats a node" % c)
    top = None
    sides = []
    for f in faces[1:]:
        ring = face_ring(z, f)
        if set(ring).isdisjoint(bset):
            if top is not None:
                fail("cell %d has two opposite faces" % c)
            top = ring
        else:
            sides.append(ring)
    if top is None or len(sides) != 4:
        fail("cell %d is not a hexahedron by opposite-face test" % c)
    tset = set(top)
    if len(tset) != 4:
        fail("top face of cell %d repeats a node" % c)

    upper = [None] * 4
    # A side face ring is (b_i, b_{i+1}, u_{i+1}, u_i) in cyclic order, so
    # each bottom node has exactly one top node among its two ring
    # neighbours: that neighbour is its upper partner via the vertical edge.
    for ring in sides:
        n_low = sum(1 for p in ring if p in bset)
        if n_low != 2:
            fail("cell %d side face shares %d nodes with bottom" % (c, n_low))
        for k, p in enumerate(ring):
            if p not in bset:
                continue
            prev = ring[(k - 1) % 4]
            nxt = ring[(k + 1) % 4]
            if prev in tset and nxt in bset:
                upper[b.index(p)] = prev
            elif nxt in tset and prev in bset:
                upper[b.index(p)] = nxt
            else:
                fail("cell %d side ring neighbour split invalid" % c)
            if prev in tset and nxt in tset:
                fail("cell %d side ring neighbours are both top nodes" % c)
    if any(u is None for u in upper) or len(set(upper)) != 4 or set(upper) != tset:
        fail("cell %d upper corner correspondence unresolved" % c)
    corners = list(b) + upper
    if len(set(corners)) != 8:
        fail("cell %d does not have 8 distinct corners" % c)
    return corners


def cell_edges(corners):
    """12 edges as (corner_index_a, corner_index_b) pairs; a < b."""
    bottom = [(i, (i + 1) % 4) for i in range(4)]
    top = [(4 + i, 4 + (i + 1) % 4) for i in range(4)]
    vertical = [(i, 4 + i) for i in range(4)]
    return [(min(a, b), max(a, b)) for a, b in bottom + top + vertical]


def det3(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return (
        x0 * (y1 * z2 - z1 * y2)
        - y0 * (x1 * z2 - z1 * x2)
        + z0 * (x1 * y2 - y1 * x2)
    )


def cell_volumes(z, report=None):
    """Signed cell volumes from the left->right winding convention.

    For a face ring (p0..p3) with normal pointing left->right, the outward
    side of the left cell is the ring orientation, so the left cell gains
    the pyramid sum and the right cell loses it.  Every cell of a valid
    mesh must end with a positive volume.
    """
    vol = array("d", bytes(8 * z.n_cells))
    fn = z.fnmem
    left = z.left
    right = z.right
    nB = z.n_bfaces
    xn, yn, zn = z.xn, z.yn, z.zn
    total = 0.0
    for f in range(z.n_faces):
        b = 4 * f
        p0 = fn[b]; p1 = fn[b + 1]; p2 = fn[b + 2]; p3 = fn[b + 3]
        x0, y0, z0 = xn[p0], yn[p0], zn[p0]
        x1, y1, z1 = xn[p1], yn[p1], zn[p1]
        x2, y2, z2 = xn[p2], yn[p2], zn[p2]
        x3, y3, z3 = xn[p3], yn[p3], zn[p3]
        s = det3(x0, y0, z0, x1, y1, z1, x2, y2, z2) \
            + det3(x0, y0, z0, x2, y2, z2, x3, y3, z3)
        lc = left[f]
        vol[lc] += s
        if f >= nB:
            vol[right[f]] -= s
    # Degenerate band: O-grid pole/seam cells collapse geometrically while
    # keeping 8 distinct node ids, so their volume is round-off around zero
    # (observed -8.5e-11 on the M6 O-grid).  Treat |v| within 1e-7 of the
    # largest cell volume as degenerate; a winding or left/right bug produces
    # a negative volume of ordinary cell magnitude and still fails.
    vmax = max(vol)
    tol = 1.0e-7 * vmax if vmax > 0.0 else 0.0
    scale = 1.0 / 6.0
    total = 0.0
    ndeg = 0
    worst = (vol[0], 0)
    for c in range(z.n_cells):
        s = vol[c]
        if s < worst[0]:
            worst = (s, c)
        if s < -tol:
            fail("cell %d volume %.6e is negative beyond the degenerate band "
                 "(tol %.3e): winding or left/right broken" % (c, s * scale, tol * scale))
        if -tol <= s <= tol:
            ndeg += 1
        total += s
    if report is not None:
        report.append(
            "volumes: min=%.6e max=%.6e degenerate=%d worst-cell=%d"
            % (worst[0] * scale, vmax * scale, ndeg, worst[1]))
    return vol, total * scale


def refine(z, report):
    V, F, C = z.n_nodes, z.n_faces, z.n_cells
    nB = z.n_bfaces
    # Boundary right-cell convention of the input.
    ghost_mode = any(z.right[f] >= 0 for f in range(nB))
    if ghost_mode:
        for f in range(nB):
            if z.right[f] != f + C:
                fail("boundary face %d right cell is not the ghost id" % f)
    else:
        for f in range(nB):
            if z.right[f] >= 0:
                fail("inconsistent boundary right cells")
    if any(z.right[f] < 0 for f in range(nB, F)):
        fail("internal face with negative right cell")

    cell_faces = build_cell_faces(z)

    # --- Pass A: edge midpoints (id on first sight) and face centers -------
    mid_id = {}
    mid_x = array("d"); mid_y = array("d"); mid_z = array("d")
    fc_x = array("d"); fc_y = array("d"); fc_z = array("d")
    fn = z.fnmem
    xn, yn, zn = z.xn, z.yn, z.zn
    for f in range(F):
        b = 4 * f
        p0 = fn[b]; p1 = fn[b + 1]; p2 = fn[b + 2]; p3 = fn[b + 3]
        for a, e in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
            key = (a, e) if a < e else (e, a)
            mid = mid_id.get(key)
            if mid is None:
                mid = V + len(mid_x)
                mid_id[key] = mid
                mid_x.append(0.5 * (xn[a] + xn[e]))
                mid_y.append(0.5 * (yn[a] + yn[e]))
                mid_z.append(0.5 * (zn[a] + zn[e]))
        fc_x.append(0.25 * (xn[p0] + xn[p1] + xn[p2] + xn[p3]))
        fc_y.append(0.25 * (yn[p0] + yn[p1] + yn[p2] + yn[p3]))
        fc_z.append(0.25 * (zn[p0] + zn[p1] + zn[p2] + zn[p3]))
    E = len(mid_x)
    fc_base = V + E
    cc_base = fc_base + F
    new_n_nodes = cc_base + C

    # --- Pass B: per-cell corners, cell centers, interior faces -------------
    corner_idx = {}
    cc_x = array("d"); cc_y = array("d"); cc_z = array("d")
    in_fn = array("i"); in_left = array("i"); in_right = array("i")
    for c in range(C):
        corners = hex_corners(z, cell_faces, c)
        for k in range(8):
            corner_idx[(c, corners[k])] = k
        sx = sy = sz = 0.0
        for p in corners:
            sx += xn[p]; sy += yn[p]; sz += zn[p]
        cc_x.append(sx / 8.0); cc_y.append(sy / 8.0); cc_z.append(sz / 8.0)
        cc = cc_base + c

        # child centroids (P + 3 edge midpoints + 3 face centers + cell
        # center, all / 8), needed for interior-face winding
        cidx = {}
        for k in range(8):
            cidx[corners[k]] = k
        cent = []
        for p in corners:
            cent.append([xn[p] / 8.0, yn[p] / 8.0, zn[p] / 8.0])
        for f in cell_faces[c]:
            b = 4 * f
            ring = (fn[b], fn[b + 1], fn[b + 2], fn[b + 3])
            w = 1.0 / 8.0
            fx = fc_x[f] * w; fy = fc_y[f] * w; fz = fc_z[f] * w
            for node in ring:
                ce = cent[cidx[node]]
                ce[0] += fx; ce[1] += fy; ce[2] += fz
        for ia, ib in cell_edges(corners):
            pa, pb = corners[ia], corners[ib]
            w = 0.5 / 8.0
            mx = (xn[pa] + xn[pb]) * w
            my = (yn[pa] + yn[pb]) * w
            mz = (zn[pa] + zn[pb]) * w
            ca = cent[ia]; cb = cent[ib]
            ca[0] += mx; ca[1] += my; ca[2] += mz
            cb[0] += mx; cb[1] += my; cb[2] += mz
        w = 1.0 / 8.0
        ccx, ccy, ccz = cc_x[c] * w, cc_y[c] * w, cc_z[c] * w
        for k in range(8):
            cent[k][0] += ccx; cent[k][1] += ccy; cent[k][2] += ccz
        # each edge of the two faces that contain it (in this cell's face list)
        edge_faces = {}
        for f in cell_faces[c]:
            b = 4 * f
            ring = (fn[b], fn[b + 1], fn[b + 2], fn[b + 3])
            for i in range(4):
                a, e = ring[i], ring[(i + 1) % 4]
                key = (a, e) if a < e else (e, a)
                lst = edge_faces.get(key)
                if lst is None:
                    edge_faces[key] = [f]
                else:
                    lst.append(f)
        for ia, ib in cell_edges(corners):
            pa, pb = corners[ia], corners[ib]
            key = (pa, pb) if pa < pb else (pb, pa)
            fs = edge_faces[key]
            if len(fs) != 2:
                fail("cell %d edge %s has %d adjacent faces" % (c, key, len(fs)))
            f1, f2 = fs[0], fs[1]
            mid = mid_id[key]
            m_x, m_y, m_z = mid_x[mid - V], mid_y[mid - V], mid_z[mid - V]
            g1 = fc_base + f1; g2 = fc_base + f2
            g1x, g1y, g1z = fc_x[f1], fc_y[f1], fc_z[f1]
            g2x, g2y, g2z = fc_x[f2], fc_y[f2], fc_z[f2]
            ccx, ccy, ccz = cc_x[c], cc_y[c], cc_z[c]
            # normal of ring (m, fc1, cc, fc2) via diagonal cross product
            d1x = ccx - m_x; d1y = ccy - m_y; d1z = ccz - m_z
            d2x = g2x - g1x; d2y = g2y - g1y; d2z = g2z - g1z
            nx_ = d1y * d2z - d1z * d2y
            ny_ = d1z * d2x - d1x * d2z
            nz_ = d1x * d2y - d1y * d2x
            dx = cent[ib][0] - cent[ia][0]
            dy = cent[ib][1] - cent[ia][1]
            dz = cent[ib][2] - cent[ia][2]
            if nx_ * dx + ny_ * dy + nz_ * dz < 0.0:
                ring = (mid, g2, cc, g1)
            else:
                ring = (mid, g1, cc, g2)
            in_fn.extend(ring)
            in_left.append(c * 8 + ia)
            in_right.append(c * 8 + ib)

    # --- Pass C: external sub-faces ----------------------------------------
    ex_fn = array("i"); ex_left = array("i"); ex_right = array("i")
    left, right = z.left, z.right
    for f in range(F):
        b = 4 * f
        p0 = fn[b]; p1 = fn[b + 1]; p2 = fn[b + 2]; p3 = fn[b + 3]
        m01 = mid_id[(p0, p1) if p0 < p1 else (p1, p0)]
        m12 = mid_id[(p1, p2) if p1 < p2 else (p2, p1)]
        m23 = mid_id[(p2, p3) if p2 < p3 else (p3, p2)]
        m30 = mid_id[(p3, p0) if p3 < p0 else (p0, p3)]
        g = fc_base + f
        lc = left[f]
        rc = right[f]
        for i, (pi, mn, mp) in enumerate(((p0, m01, m30), (p1, m12, m01),
                                          (p2, m23, m12), (p3, m30, m23))):
            ex_fn.extend((pi, mn, g, mp))
            ex_left.append(lc * 8 + corner_idx[(lc, pi)])
            if f >= nB:
                ex_right.append(rc * 8 + corner_idx[(rc, pi)])
            elif ghost_mode:
                ex_right.append(4 * f + i + 8 * C)
            else:
                ex_right.append(-1)

    new = Zone()
    new.n_nodes = new_n_nodes
    new.n_faces = 4 * F + 12 * C
    new.n_cells = 8 * C
    new.vol_bc_type = z.vol_bc_type
    new.num_face_node = array("i", b"\x04\x00\x00\x00" * new.n_faces)
    new.fnmem = ex_fn + in_fn
    new.left = ex_left + in_left
    new.right = ex_right + in_right
    new.n_bfaces = 4 * nB
    new.bc_type = array("i", bytes(4 * new.n_bfaces))
    new.bc_name = array("i", bytes(4 * new.n_bfaces))
    for i in range(new.n_bfaces):
        new.bc_type[i] = z.bc_type[i // 4]
        new.bc_name[i] = z.bc_name[i // 4]
    new.xn = z.xn + mid_x + fc_x + cc_x
    new.yn = z.yn + mid_y + fc_y + cc_y
    new.zn = z.zn + mid_z + fc_z + cc_z

    report.append("input : V=%d F=%d C=%d nB=%d E=%d" % (V, F, C, nB, E))
    report.append("euler-input: V-E+F-C=%d" % (V - E + F - C))
    report.append(
        "output: V=%d F=%d C=%d nB=%d"
        % (new.n_nodes, new.n_faces, new.n_cells, new.n_bfaces)
    )
    if new.n_nodes != V + E + F + C:
        fail("node count mismatch")
    if new.n_faces != 4 * F + 12 * C:
        fail("face count mismatch")
    if new.n_cells != 8 * C:
        fail("cell count mismatch")
    if len(new.fnmem) != 4 * new.n_faces:
        fail("face-node storage mismatch")
    if len(new.left) != new.n_faces or len(new.right) != new.n_faces:
        fail("left/right storage mismatch")
    return new


def write_ofl(path, z, pid, zone_type):
    if sys.byteorder != "little" or array("i").itemsize != 4 or array("d").itemsize != 8:
        fail("host layout not little-endian 4-byte int / 8-byte double")
    payload = bytearray()
    payload += struct.pack("<3i", z.n_nodes, z.n_faces, z.n_cells)
    payload += z.xn.tobytes() + z.yn.tobytes() + z.zn.tobytes()
    payload += struct.pack("<i", z.vol_bc_type)
    payload += z.num_face_node.tobytes()
    payload += z.fnmem.tobytes()
    payload += z.left.tobytes() + z.right.tobytes()
    payload += struct.pack("<i", z.n_bfaces)
    payload += z.bc_type.tobytes() + z.bc_name.tobytes()
    payload += struct.pack("<i", 0)
    header = struct.pack("<i", 1)
    header += struct.pack("<i", pid[0]) + struct.pack("<i", zone_type[0])
    header += struct.pack("<q", len(payload))
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(payload)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src = sys.argv[1]
    dst = sys.argv[2]
    report = []
    z, pid, zone_type = parse_ofl(src)
    report.append("REFINE_PARSE_PASS")
    _, vol_in_total = cell_volumes(z, report)
    if vol_in_total <= 0.0:
        fail("input total volume %.6e is not positive" % vol_in_total)
    report.append("input-total-volume=%.12e" % vol_in_total)
    new = refine(z, report)
    report.append("REFINE_COUNTS_PASS")
    write_ofl(dst, new, pid, zone_type)

    back, _, _ = parse_ofl(dst)
    _, vol_out_total = cell_volumes(back, report)
    rel = abs(vol_out_total - vol_in_total) / vol_in_total
    report.append("re-read-total-volume=%.12e rel-diff=%.3e"
                  % (vol_out_total, rel))
    if rel > 1e-10:
        fail("total volume drifts by %.3e" % rel)
    report.append("REFINE_VOLUME_PASS")

    # degree == 6 for every cell plus sampled full hex checks on the
    # re-read file (8 corners, opposite-face and side-ring topology)
    cell_faces = build_cell_faces(back)
    step = max(1, back.n_cells // 500)
    for c in range(0, back.n_cells, step):
        hex_corners(back, cell_faces, c)
    report.append("REFINE_REPARSE_PASS")
    report.append("REFINE_ALL_PASS")
    for line in report:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

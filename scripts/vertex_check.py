#!/usr/bin/env python3
"""Independent Fraction-exact vertex enumerator + Charnes-Cooper bounds.

Verifies the JS exact tableau simplex against direct vertex enumeration:
for every compiled credal polytope (A w = b, w >= 0) it enumerates all basic
feasible solutions exactly with Fractions, computes evidence support bounds
[e_min, e_max] and the conditional interval [min N/E, max N/E] over vertices,
then compares against the JS solveCredal decision.

  usage: node scripts/crosscheck.mjs 400 > /tmp/cases.json
         python3 scripts/vertex_check.py /tmp/cases.json
"""
import json
import sys
from fractions import Fraction
from itertools import combinations


def rank_and_solve(A, b, n_cols):
    """Gaussian-eliminate the m x (n+1) augmented system. Returns (rank_A,
    solution or None). If rank_A == n_cols the solution (when consistent) is
    unique and returned; non-unique systems return (rank_A, None)."""
    m = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    rank = 0
    for c in range(n_cols):
        r = next((i for i in range(rank, m) if M[i][c] != 0), None)
        if r is None:
            continue
        M[rank], M[r] = M[r], M[rank]
        for i in range(m):
            if i != rank and M[i][c] != 0:
                f = M[i][c] / M[rank][c]
                for j in range(c, n_cols + 1):
                    M[i][j] -= f * M[rank][j]
        rank += 1
    for i in range(rank, m):
        if M[i][n_cols] != 0:
            return rank, None  # inconsistent
    if rank < n_cols:
        return rank, None  # underdetermined
    x = [Fraction(0)] * n_cols
    for i in reversed(range(n_cols)):
        x[i] = (M[i][n_cols] - sum(M[i][j] * x[j] for j in range(i + 1, n_cols))) / M[i][i]
    return rank, x


def vertices(rows, n_cols):
    """All vertices of {A x = b, x >= 0} exactly.

    A point x is a vertex iff its support columns are linearly independent.
    Every support extends to a column basis B of size r = rank(A); solving
    A_B x_B = b (full column rank => unique candidate) recovers x. Enumerating
    all r-subsets therefore covers every vertex, degenerate included.
    """
    m = len(rows)
    r, _ = rank_and_solve(
        [[Fraction(v) for v in row["a"]] for row in rows],
        [Fraction(row["b"]) for row in rows],
        n_cols,
    )
    if r is None:
        return []
    verts = []
    for S in combinations(range(n_cols), r):
        A_S = [[row["a"][j] for j in S] for row in rows]
        rankS, xS = rank_and_solve(A_S, [row["b"] for row in rows], r)
        if rankS < r or xS is None or any(v < 0 for v in xS):
            continue
        x = [Fraction(0)] * n_cols
        for i, j in enumerate(S):
            x[j] = xS[i]
        if not any(v == x for v in verts):
            verts.append(x)
    return verts


def dot(a, x):
    return sum(Fraction(ai) * xi for ai, xi in zip(a, x))


def main():
    cases = json.load(open(sys.argv[1]))
    agree = checked = skipped = 0
    farkas_ok = 0
    disagreements = []
    tally = {}
    for case in cases:
        dec = case["decision"]
        c = case.get("compiled")
        if not c:
            tally[dec.get("code", "COMPILE_ERROR")] = tally.get(dec.get("code", "COMPILE_ERROR"), 0) + 1
            skipped += 1
            continue
        rows = [{"a": [Fraction(v) for v in r["a"]], "b": Fraction(r["b"])} for r in c["rows"]]
        n = c["worlds"]
        verts = vertices(rows, n)
        eMask = [Fraction(v) for v in c["eMask"]]
        nMask = [Fraction(v) for v in c["nMask"]] if c.get("nMask") else None

        if not verts:
            expected = ("REFUSED", "CONFLICTING_FIGURES")
            ok = dec["status"] == expected[0] and dec["code"] == expected[1]
            if not ok:
                disagreements.append((case["t"], "infeasible expected", dec))
            else:
                agree += 1
            checked += 1
            tally["infeasible"] = tally.get("infeasible", 0) + 1
            # the refusal must carry a Farkas certificate over the ORIGINAL
            # compiled rows: len(y) == m, y^T A >= 0 colwise, y^T b < 0.
            y = dec.get("details", {}).get("farkas")
            if y is None or len(y) != len(rows):
                disagreements.append((case["t"], "farkas-missing-or-wrong-dim", dec))
                continue
            y = [Fraction(v) for v in y]
            yA = [sum(y[i] * rows[i]["a"][j] for i in range(len(rows))) for j in range(n)]
            yb = sum(y[i] * rows[i]["b"] for i in range(len(rows)))
            if all(v >= 0 for v in yA) and yb < 0:
                farkas_ok += 1
            else:
                disagreements.append(
                    (case["t"], "farkas-cert-invalid", {"yA_neg": [str(v) for v in yA if v < 0], "yb": str(yb)})
                )
            continue

        eVals = [dot(eMask, v) for v in verts]
        eMin, eMax = min(eVals), max(eVals)

        # conditional bounds over positive-support vertices
        cond = [dot(nMask, v) / e for v, e in zip(verts, eVals) if e > 0] if nMask else []
        if eMax == 0:
            ok = dec["status"] == "REFUSED" and dec["code"] == "DEGENERATE_PARTITION"
            tag = "degenerate"
        elif eMin == 0:
            ok = dec["status"] == "REFUSED" and dec["code"] == "ZERO_SUPPORT_POSSIBLE"
            tag = "zero-support-possible"
        elif min(cond) == max(cond):
            ok = dec["status"] == "ANSWER" and Fraction(int(dec["value"]["n"]), int(dec["value"]["d"])) == min(cond)
            tag = "answer"
        else:
            ok = (
                dec["status"] == "REFUSED"
                and dec["code"] == "UNDERDETERMINED_INTERVAL"
                and Fraction(*map(int, dec["details"]["interval"]["min"].split("/"))) == min(cond)
                and Fraction(*map(int, dec["details"]["interval"]["max"].split("/"))) == max(cond)
            )
            tag = "interval"
        checked += 1
        tally[tag] = tally.get(tag, 0) + 1
        if ok:
            agree += 1
        else:
            disagreements.append((case["t"], tag, dec.get("code") or dec.get("value"), (min(cond) if cond else None, max(cond) if cond else None, eMin, eMax)))

    print(f"checked={checked} skipped={skipped} agree={agree} farkas_ok={farkas_ok}")
    print("tally:", tally)
    if disagreements:
        print(f"DISAGREEMENTS ({len(disagreements)}):")
        for d in disagreements[:20]:
            print(" ", d)
        sys.exit(1)
    print("EXACT AGREEMENT")


if __name__ == "__main__":
    main()

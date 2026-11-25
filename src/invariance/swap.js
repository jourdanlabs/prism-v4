// invariance-gate :: Slot-2 literal number-swap permutation engine.
// Sites are disjoint literal substrings in X (e.g. "20%" ... "80%",
// "five eighths" ... "seven twelfths"). A permutation assignment assign[i]=j
// means "site i receives the literal text originally at site j". Rebuilds X
// deterministically and remaps offsets for differing lengths.

// assign: array of length m; assign[i] = source site index whose text goes to site i.
export function applyPermutation(X, sites, assign) {
  const sorted = [...sites].sort((a, b) => a.s - b.s);
  const out = [];
  let cursor = 0;
  const remapped = new Array(sorted.length);
  const deltaAt = []; // {pos, delta} for generic span remap
  for (let i = 0; i < sorted.length; i++) {
    const site = sorted[i];
    const newText = sorted[assign[i]].text;
    out.push(X.slice(cursor, site.s));
    const ns = out.join('').length;
    out.push(newText);
    const ne = ns + newText.length;
    remapped[i] = { s: ns, e: ne, text: newText, value: sorted[assign[i]].value };
    deltaAt.push({ oldS: site.s, oldE: site.e, newS: ns, newE: ne });
    cursor = site.e;
  }
  out.push(X.slice(cursor));
  const Xp = out.join('');

  // remap any span {s,e} in original coordinates to X' coordinates
  const remapSpan = (sp) => {
    let s = sp.s;
    let e = sp.e;
    for (const d of deltaAt) {
      const shift = d.newE - d.newS - (d.oldE - d.oldS);
      if (d.oldE <= s) {
        // span entirely after this site: shift by cumulative delta — handled below
      }
    }
    // recompute properly: walk sites in order accumulating delta
    s = remapOffset(sp.s, deltaAt, false);
    e = remapOffset(sp.e, deltaAt, true);
    return { s, e };
  };
  return { X: Xp, remappedSites: remapped, remapSpan };
}

function remapOffset(off, deltaAt, isEnd) {
  // apply cumulative length deltas of sites entirely before `off`;
  // a site containing `off` maps proportionally to its start.
  let acc = 0;
  let inside = null;
  for (const d of deltaAt) {
    if (d.oldE <= off) acc += d.newE - d.newS - (d.oldE - d.oldS);
    if (d.oldS <= off && off <= d.oldE) inside = d;
  }
  if (inside) return isEnd ? inside.newE : inside.newS;
  return off + acc;
}

// All pairwise swaps (i<j). Skips swaps between identical literal text
// (permuting equal literals is a no-op and would be vacuous).
export function pairwiseSwaps(sites) {
  const out = [];
  for (let i = 0; i < sites.length; i++) {
    for (let j = i + 1; j < sites.length; j++) {
      if (sites[i].text === sites[j].text) continue;
      const assign = sites.map((_, k) => k);
      assign[i] = j;
      assign[j] = i;
      out.push(assign);
    }
  }
  return out;
}

// All m! permutations (for small m), identity excluded.
export function allPermutations(m) {
  const out = [];
  const used = new Array(m).fill(false);
  const cur = [];
  const rec = () => {
    if (cur.length === m) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < m; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out.filter((a) => a.some((v, i) => v !== i));
}

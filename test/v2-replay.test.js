// v2-replay regression tests — Bulma's published V2-gate fixtures replayed
// through the public gate. Each case that previously released wrong now
// refuses typed; each oracle release still releases exactly.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runReplay } from '../scripts/controlled-replay.mjs';

const results = runReplay();

for (const r of results) {
  test(`v2-replay: ${r.id}`, () => {
    assert.ok(r.pass, `want ${JSON.stringify(r.expect)} got ${r.got}`);
  });
}

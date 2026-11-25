import test from 'node:test';
import assert from 'node:assert/strict';
import { jsonToProgram, makeOllamaExtractor, ExtractionError } from '../src/live/extractor.js';

const partitionSentence = 'Every device is either Vermilion or Teal.';
const quiverSentence =
  'The quiver reads positive for 30% of Vermilion devices and 20% of Teal devices.';
const tremorSentence =
  'The tremor reads positive for 30% of Vermilion devices and 20% of Teal devices.';
const passage = [
  'Vermilion is one kind of device.',
  'Teal is one kind of device.',
  partitionSentence,
  quiverSentence,
  tremorSentence,
].join(' ');

function assertWithin(text, sentence, span) {
  const start = text.indexOf(sentence);
  assert.notEqual(start, -1);
  assert.ok(span.s >= start);
  assert.ok(span.e <= start + sentence.length);
}

const originalInput = {
  hyps: [
    { id: 'h0', name: 'Vermilion', span: 'Vermilion is one kind of device.' },
    { id: 'h1', name: 'Teal', span: 'Teal is one kind of device.' },
  ],
  events: [
    { id: 'e0', name: 'quiver', span: 'The quiver reads positive' },
    { id: 'e1', name: 'tremor', span: 'The tremor reads positive' },
  ],
  partition: ['h0', 'h1'],
  partitionSpan: partitionSentence,
  priors: [],
  liks: [
    { ev: 'e0', hyp: 'h1', pol: 'POS', value: '1/5', span: '20% of Teal devices' },
    { ev: 'e1', hyp: 'h1', pol: 'POS', value: '1/5', span: '20% of Teal devices' },
  ],
  obs: [],
  query: { target: 'h0', given: [] },
};

const leadingLiks = [
  { ev: 'e0', hyp: 'h0', pol: 'POS', value: '3/10', span: '30% of Vermilion devices' },
  { ev: 'e1', hyp: 'h0', pol: 'POS', value: '3/10', span: '30% of Vermilion devices' },
];

test('grounds repeated rates and exhaustive partition in their sentences', () => {
  const { program, errors } = jsonToProgram({
    ...originalInput,
    liks: [...leadingLiks, ...originalInput.liks],
  }, passage);
  assert.deepEqual(errors, []);
  assert.equal(program.decls.filter((d) => d.op === 'LIK').length, 4);
  for (const [id, name, sentence] of [
    ['e0', 'quiver', quiverSentence],
    ['e1', 'tremor', tremorSentence],
  ]) {
    const event = program.decls.find((d) => d.op === 'EVENT' && d.id === id);
    assert.ok(event);
    assert.equal(passage.slice(event.span.s, event.span.e), name);
    const lik = program.decls.find((d) => d.op === 'LIK' && d.ev === id && d.hyp === 'h1');
    assert.ok(lik);
    assert.deepEqual(lik.value, { n: 1n, d: 5n });
    const sentenceStart = passage.indexOf(sentence);
    assert.notEqual(sentenceStart, -1);
    const trailingText = '20% of Teal devices';
    const trailingS = sentenceStart + sentence.indexOf(trailingText);
    assert.deepEqual(lik.span, { s: trailingS, e: trailingS + trailingText.length });
    assert.equal(passage.slice(lik.span.s, lik.span.e), trailingText);
    assertWithin(passage, sentence, lik.span);
    const leading = program.decls.find((d) => d.op === 'LIK' && d.ev === id && d.hyp === 'h0');
    assert.ok(leading);
    assert.deepEqual(leading.value, { n: 3n, d: 10n });
    const leadingText = '30% of Vermilion devices';
    const leadingS = sentenceStart + sentence.indexOf(leadingText);
    assert.deepEqual(leading.span, { s: leadingS, e: leadingS + leadingText.length });
    assert.equal(passage.slice(leading.span.s, leading.span.e), leadingText);
    assertWithin(passage, sentence, leading.span);
  }
  const part = program.decls.find((d) => d.op === 'PART_EXHAUSTIVE');
  assert.ok(part);
  assertWithin(passage, partitionSentence, part.span);
  const excerpt = passage.slice(part.span.s, part.span.e);
  assert.ok(excerpt.includes('Vermilion'));
  assert.ok(excerpt.includes('Teal'));
});

test('rejects original trailing-only rates without leading anchors at extraction boundary', async (t) => {
  const { errors } = jsonToProgram(
    { ...originalInput, liks: [...originalInput.liks] },
    passage,
  );
  const findings = errors.filter((message) => message.startsWith('UNSTABLE_BINDING:'));
  assert.equal(findings.length, 2);
  assert.equal(new Set(findings).size, 2);
  const hasFinding = (ev, trailing, leading) =>
    findings.some(
      (message) =>
        message.includes(`"20%" at [${trailing}]`) &&
        message.includes(`inherits ${ev} from leading literal "30%"`) &&
        message.includes(`at [${leading}]`) &&
        message.includes(
          `but that leading site is not established by a matching declaration for ${ev}`,
        ),
    );
  assert.ok(hasFinding('e0', '162,165', '133,136'));
  assert.ok(hasFinding('e1', '242,245', '213,216'));
  let fetchCalls = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetchCalls += 1;
    return new Response(
      JSON.stringify({ message: { content: JSON.stringify(originalInput) } }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );
  });
  const extract = makeOllamaExtractor({ model: 'test-model', host: 'http://127.0.0.1:11434' });
  await assert.rejects(
    () => extract(passage),
    (err) => {
      assert.ok(err instanceof ExtractionError);
      for (const finding of findings) assert.ok(err.message.includes(finding));
      return true;
    },
  );
  assert.equal(fetchCalls, 1);
});

test('reports a LIK error when jointly owned occurrence is ambiguous', () => {
  const adverse = [
    'Teal is one kind of device.',
    'The quiver is one signal.',
    'The quiver reads positive for 20% of Teal devices.',
    'The quiver reads positive for 20% of Teal devices.',
  ].join(' ');
  const { errors } = jsonToProgram({
    hyps: [{ id: 'h0', name: 'Teal', span: 'Teal is one kind of device.' }],
    events: [{ id: 'e0', name: 'quiver', span: 'The quiver is one signal.' }],
    partition: [],
    priors: [],
    liks: [
      { ev: 'e0', hyp: 'h0', pol: 'POS', value: '1/5', span: '20% of Teal devices' },
    ],
    obs: [],
    query: { target: 'h0', given: [] },
  }, adverse);
  assert.ok(errors.some((message) => /LIK|likelihood/i.test(message)));
});

test('rejects a unique verbatim LIK quote bound to the wrong event', () => {
  const X = [
    'Vermilion is one kind of device.',
    'Teal is one kind of device.',
    'In Vermilion devices, the quiver test flags 30% of Vermilion devices.',
    'In Vermilion devices, the tremor test flags 40% of Vermilion devices.',
  ].join(' ');
  const { errors } = jsonToProgram({
    hyps: [
      { id: 'h0', name: 'Vermilion', span: 'Vermilion is one kind of device.' },
      { id: 'h1', name: 'Teal', span: 'Teal is one kind of device.' },
    ],
    events: [
      { id: 'e0', name: 'quiver', span: 'the quiver test' },
      { id: 'e1', name: 'tremor', span: 'the tremor test' },
    ],
    partition: [],
    priors: [],
    liks: [
      { ev: 'e1', hyp: 'h0', pol: 'POS', value: '3/10', span: 'the quiver test flags 30% of Vermilion devices' },
    ],
    obs: [],
    query: { target: 'h0', given: [] },
  }, X);
  assert.ok(errors.some((message) => /UNSTABLE_BINDING/.test(message)));
});

test('reports an invalid partitionSpan when a hyp name folds two categories into one', () => {
  const adverse = [
    'Vermilion is one kind of device.',
    'Teal is one kind of device.',
    'Every device is either Vermilion or Teal or Indigo.',
  ].join(' ');
  const { errors } = jsonToProgram({
    hyps: [
      { id: 'h0', name: 'Vermilion', span: 'Vermilion is one kind of device.' },
      { id: 'h1', name: 'Teal or Indigo', span: 'Teal is one kind of device.' },
    ],
    events: [],
    partition: ['h0', 'h1'],
    partitionSpan: 'Every device is either Vermilion or Teal or Indigo.',
    priors: [],
    liks: [],
    obs: [],
    query: { target: 'h0', given: [] },
  }, adverse);
  assert.ok(errors.some((message) => /partitionSpan/i.test(message)));
});

test('reports a LIK shared literal error when distinct events claim one numeric site', () => {
  const hyps = [
    { id: 'h0', name: 'Vermilion', span: 'Vermilion is one kind of device.' },
    { id: 'h1', name: 'Teal', span: 'Teal is one kind of device.' },
  ];
  const shared = [
    'Vermilion is one kind of device.',
    'Teal is one kind of device.',
    'Beside the buzzer, the lamp reads positive for 30% of Vermilion devices.',
  ].join(' ');
  const sharedResult = jsonToProgram({
    hyps,
    events: [
      { id: 'e0', name: 'lamp', span: 'the lamp' },
      { id: 'e1', name: 'buzzer', span: 'the buzzer' },
    ],
    partition: [],
    priors: [],
    liks: [
      { ev: 'e0', hyp: 'h0', pol: 'POS', value: '3/10', span: 'the lamp reads positive for 30% of Vermilion devices' },
      { ev: 'e1', hyp: 'h0', pol: 'POS', value: '3/10', span: 'Beside the buzzer, the lamp reads positive for 30% of Vermilion devices' },
    ],
    obs: [],
    query: { target: 'h0', given: [] },
  }, shared);
  assert.ok(sharedResult.errors.some((message) => /LIK/.test(message) && /shared literal/i.test(message)));

  const distinct = [
    'Vermilion is one kind of device.',
    'Teal is one kind of device.',
    'The lamp reads positive for 30% of Vermilion devices.',
    'The buzzer reads positive for 30% of Vermilion devices.',
  ].join(' ');
  const distinctResult = jsonToProgram({
    hyps,
    events: [
      { id: 'e0', name: 'lamp', span: 'The lamp' },
      { id: 'e1', name: 'buzzer', span: 'The buzzer' },
    ],
    partition: [],
    priors: [],
    liks: [
      { ev: 'e0', hyp: 'h0', pol: 'POS', value: '3/10', span: 'The lamp reads positive for 30% of Vermilion devices' },
      { ev: 'e1', hyp: 'h0', pol: 'POS', value: '3/10', span: 'The buzzer reads positive for 30% of Vermilion devices' },
    ],
    obs: [],
    query: { target: 'h0', given: [] },
  }, distinct);
  assert.deepEqual(distinctResult.errors, []);
});

test('reports a LIK error when another rate immediately governs the target hyp', () => {
  const adverse = [
    'Vermilion is one kind of device.',
    'Teal is one kind of device.',
    'Vermilion devices use the quiver test, which flags 20% in calibration checks but 30% of Teal devices.',
    'Vermilion devices use the tremor test, which flags 20% in calibration checks but 30% of Teal devices.',
  ].join(' ');
  const { errors } = jsonToProgram({
    hyps: [
      { id: 'h0', name: 'Vermilion', span: 'Vermilion is one kind of device.' },
      { id: 'h1', name: 'Teal', span: 'Teal is one kind of device.' },
    ],
    events: [
      { id: 'e0', name: 'quiver', span: 'the quiver test' },
      { id: 'e1', name: 'tremor', span: 'the tremor test' },
    ],
    partition: [],
    priors: [],
    liks: [
      { ev: 'e0', hyp: 'h1', pol: 'POS', value: '1/5', span: '20% in calibration checks but 30% of Teal devices' },
    ],
    obs: [],
    query: { target: 'h0', given: [] },
  }, adverse);
  assert.ok(errors.some((message) => /LIK|likelihood/i.test(message)));
});

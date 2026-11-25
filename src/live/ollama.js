// live eval :: Ollama client (local daemon or remote host). Deterministic
// settings (temperature 0) for reproducible extraction.

const DEFAULT_HOST = 'http://localhost:11434';

export async function ollamaAvailable(host = DEFAULT_HOST, timeoutMs = 5000) {
  try {
    const res = await fetch(`${host}/api/tags`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    const j = await res.json();
    return (j.models ?? []).map((m) => m.name);
  } catch {
    return null;
  }
}

export async function chat({
  model,
  messages,
  host = DEFAULT_HOST,
  format = 'json',
  timeoutMs = 120000,
  numPredict = 900,
  think = undefined,
  seed = undefined,
}) {
  const res = await fetch(`${host}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    signal: AbortSignal.timeout(timeoutMs),
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      format,
      ...(think !== undefined ? { think } : {}),
      options: { temperature: 0, num_predict: numPredict, ...(seed !== undefined ? { seed } : {}) },
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`);
  const j = await res.json();
  return { content: j.message?.content ?? '', raw: j };
}

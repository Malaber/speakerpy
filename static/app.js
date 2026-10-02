'use strict';
const $ = id => document.getElementById(id);
const terminal = new Set(['complete', 'failed', 'cancelled']);
let library = [], speakers = [], parsedText = '', comparisonText = '', stream = null, watchedId = null;
let voiceSignature = '', finishedVoices = '', polling = false;
const cards = new Map();
const voiceCards = new Map();

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function message(text = '') { $('message').textContent = text; $('message').hidden = !text; }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || response.statusText));
  }
  return response.json();
}
function action(button, fn) {
  button.addEventListener('click', async () => {
    button.disabled = true;
    try { message(); await fn(); } catch (error) { message(error.message); }
    finally { button.disabled = false; updateGenerate(); }
  });
}
function mappings() {
  return Object.fromEntries([...$('speakers').querySelectorAll('select')].map(select => [select.dataset.speaker, select.value]));
}
function renderSpeakers() {
  const previous = mappings();
  $('speakers').replaceChildren();
  speakers.forEach((speaker, index) => {
    const row = node('div', undefined, 'speaker-row'), label = node('label', speaker), select = node('select');
    select.id = `speaker-${index}`; label.htmlFor = select.id; select.dataset.speaker = speaker;
    if (!library.length) select.add(new Option('Waiting for voice samples…', ''));
    library.forEach(voice => select.add(new Option(voice.name, voice.id)));
    select.value = library.some(v => v.id === previous[speaker]) ? previous[speaker] : (library[index % library.length]?.id || '');
    select.addEventListener('change', updateGenerate);
    row.append(label, select); $('speakers').append(row);
  });
  $('randomize').disabled = !speakers.length || !library.length;
  updateGenerate();
}
function updateGenerate() {
  $('generate').disabled = !speakers.length || !library.length || parsedText !== $('dialogue').value || Object.values(mappings()).some(v => !v);
}
async function parseText() {
  const text = $('dialogue').value;
  const result = await api('/parse', {text});
  if (text !== $('dialogue').value) return;
  speakers = result.speakers; parsedText = text;
  $('parse-summary').textContent = `${result.turns} turns · ${result.total} chunks`;
  renderSpeakers();
}
async function refreshVoices() {
  const result = await api('/voices');
  library = result.voices; comparisonText = result.comparison_text;
  $('comparison-text').textContent = comparisonText;
  if (result.errors.length) message(`Some voice folders need attention: ${result.errors.join('; ')}`);
  const signature = JSON.stringify(library);
  if (signature !== voiceSignature) {
    voiceSignature = signature;
    const ids = new Set(library.map(voice => voice.id));
    for (const [id, entry] of voiceCards) if (!ids.has(id)) { entry.el.remove(); voiceCards.delete(id); }
    $('voices').querySelector('.empty-library')?.remove();
    if (!library.length) $('voices').append(node('p', 'Preparing your first three samples. Follow their progress in the queue. You can also add a voice below.', 'hint empty-library'));
    for (const voice of library) {
      const existing = voiceCards.get(voice.id);
      // Preserve playback and in-progress description edits when another voice finishes.
      if (existing?.fingerprint === voice.fingerprint) continue;
      const card = node('article', undefined, 'voice-card');
      const player = node('audio'); player.controls = true; player.preload = 'none';
      player.src = `/voices/${encodeURIComponent(voice.id)}/audio?v=${voice.fingerprint}`;
      player.setAttribute('aria-label', `${voice.name} sample`);
      const label = node('label', 'Voice characteristics');
      const description = node('textarea'); description.rows = 3; description.maxLength = 1000;
      description.value = voice.description; description.setAttribute('aria-label', `${voice.name} voice characteristics`);
      label.append(description);
      const regenerate = node('button', 'Regenerate sample', 'secondary');
      action(regenerate, async () => {
        await api(`/voices/${encodeURIComponent(voice.id)}/regenerate`, {description: description.value});
        message(`Regeneration of ${voice.name} queued. The current sample stays available until it succeeds.`);
        await refreshJobs();
      });
      card.append(node('h3', voice.name), node('p', voice.language, 'hint'), player, label, regenerate);
      if (existing) existing.el.replaceWith(card); else $('voices').append(card);
      voiceCards.set(voice.id, {el: card, fingerprint: voice.fingerprint});
    }
    renderSpeakers();
  }
}
function renderJob(job) {
  let entry = cards.get(job.id);
  const signature = JSON.stringify(job);
  if (entry?.signature === signature) return;
  if (!entry) {
    entry = {el: node('article', undefined, 'job')}; cards.set(job.id, entry);
    $('jobs').prepend(entry.el);
  }
  entry.signature = signature;
  const card = entry.el; card.replaceChildren();
  const head = node('div', undefined, 'job-head');
  head.append(node('strong', job.kind === 'voice' ? `Voice · ${job.title}` : `Conversation · ${job.id.slice(0, 6)}`), node('span', job.state, 'job-state'));
  card.append(head);
  if (!terminal.has(job.state)) {
    const progress = node('progress'); progress.max = job.total; progress.value = job.completed;
    progress.setAttribute('aria-label', `${job.completed} of ${job.total} chunks`); card.append(progress);
  }
  let status = `${job.completed} / ${job.total} chunks · ${job.progress}%`;
  if (job.state === 'queued') status = `Waiting · position ${job.queue_position}`;
  if (job.cached) status += ` · ${job.cached} cached`;
  if (job.state === 'assembling') status += ' · Assembling WAV';
  if (job.state === 'encoding') status += ' · Encoding MP3';
  card.append(node('p', status));
  if (job.current) card.append(node('p', `${job.current.speaker} / ${job.current.voice} — ${job.current.text.slice(0, 150)}`));
  if (job.error) card.append(node('p', job.error, 'error'));
  if (job.downloads.mp3 || job.downloads.wav) {
    const player = node('audio'); player.controls = true; player.preload = 'none';
    player.src = job.downloads.mp3 || job.downloads.wav; card.append(player);
  }
  const actions = node('div', undefined, 'job-actions');
  for (const [format, url] of Object.entries(job.downloads)) {
    const link = node('a', `Download ${format.toUpperCase()}`); link.href = url; link.download = `conversation.${format}`; actions.append(link);
  }
  if (!terminal.has(job.state)) {
    const cancel = node('button', 'Cancel', 'secondary'); cancel.disabled = job.state === 'cancelling';
    action(cancel, async () => { renderJob(await api(`/jobs/${job.id}/cancel`, {})); }); actions.append(cancel);
  } else if (job.state !== 'complete') {
    const retry = node('button', 'Retry', 'secondary');
    action(retry, async () => { await api(`/jobs/${job.id}/retry`, {}); await refreshJobs(); }); actions.append(retry);
    const log = node('a', 'Worker log'); log.href = `/jobs/${job.id}/log`; log.target = '_blank'; log.rel = 'noopener'; actions.append(log);
  }
  card.append(actions);
}
async function refreshJobs() {
  const jobs = await api('/jobs');
  const active = jobs.filter(job => !terminal.has(job.state));
  $('queue-count').textContent = `${active.length} active`;
  if (jobs.length) $('jobs').querySelector('.empty')?.remove();
  [...jobs].reverse().forEach(renderJob);
  const ids = new Set(jobs.map(job => job.id));
  for (const [id, entry] of cards) if (!ids.has(id)) { entry.el.remove(); cards.delete(id); }
  const completed = jobs.filter(j => j.kind === 'voice' && j.state === 'complete').map(j => j.id).join();
  if (completed !== finishedVoices) { finishedVoices = completed; await refreshVoices(); }
  // One SSE connection avoids exhausting browser connection slots with queued jobs.
  const watch = active.find(j => j.state !== 'queued') || active.at(-1);
  if (watch?.id !== watchedId) {
    stream?.close(); stream = null; watchedId = watch?.id;
    if (watch) {
      stream = new EventSource(`/jobs/${watch.id}/events`);
      stream.onmessage = event => {
        const job = JSON.parse(event.data); renderJob(job);
        if (terminal.has(job.state)) { stream?.close(); watchedId = null; }
      };
    }
  }
}
action($('parse'), parseText);
action($('refresh-voices'), refreshVoices);
action($('randomize'), async () => {
  if (library.length < speakers.length) throw new Error(`Distinct voices need ${speakers.length} samples; add ${speakers.length - library.length} more first.`);
  const shuffled = [...library];
  for (let i = shuffled.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }
  [...$('speakers').querySelectorAll('select')].forEach((select, i) => { select.value = shuffled[i].id; });
});
action($('generate'), async () => {
  const pause = Number($('pause').value);
  if (!Number.isInteger(pause) || pause < 0 || pause > 3000) throw new Error('Pause must be between 0 and 3000 milliseconds.');
  await api('/jobs', {text: $('dialogue').value, voices: mappings(), language: $('language').value, pause_ms: pause});
  await refreshJobs();
});
$('dialogue').addEventListener('input', () => {
  updateGenerate(); $('parse-summary').textContent = 'Text changed · parse again';
  try { localStorage.setItem('speakerpy-dialogue', $('dialogue').value); } catch (_) { /* optional draft storage */ }
});
$('add-voice').addEventListener('submit', async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  try {
    await api('/voices', {name: $('voice-name').value, description: $('voice-description').value, ref_text: comparisonText, language: 'German'});
    $('add-voice').reset(); message('Voice sample queued. It will appear in the library when ready.'); await refreshJobs();
  } catch (error) { message(error.message); } finally { button.disabled = false; }
});
(async () => {
  try {
    const draft = localStorage.getItem('speakerpy-dialogue'); if (draft) $('dialogue').value = draft;
  } catch (_) { /* optional draft storage */ }
  try {
    const health = await api('/health');
    health.languages.forEach(language => $('language').add(new Option(language, language)));
    $('language').value = 'German';
    $('memory-mode').textContent = `One worker · model released every ${health.recycle_chunks} chunk${health.recycle_chunks === 1 ? '' : 's'}.`;
    $('connection').textContent = 'Local · connected';
    if (!health.ffmpeg) message('Install ffmpeg to enable conversation export.');
    await refreshVoices(); await parseText(); await refreshJobs();
  } catch (error) { message(error.message); }
  setInterval(async () => {
    if (polling) return; polling = true;
    try { await refreshJobs(); $('connection').textContent = 'Local · connected'; }
    catch (_) { $('connection').textContent = 'Reconnecting…'; }
    finally { polling = false; }
  }, 2000);
})();

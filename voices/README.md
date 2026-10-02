# Voice files

An empty library automatically queues three different German samples on startup,
all reading the same comparison text. Add or regenerate voices in the browser.
The library is discovered from these files on every refresh; there is no database.

You can also import a voice by creating a folder such as `anna/` with:

- `reference.wav`: nonempty mono/stereo audio, at most 60 seconds (a short, clean
  recording is preferable).
- `voice.json`:

```json
{
  "name": "Anna",
  "language": "German",
  "ref_text": "An exact transcript of reference.wav",
  "description": "An optional description for future regeneration"
}
```

Use letters, numbers, `_` or `-` in folder names. Click **Refresh from disk** after
manual edits. Regeneration keeps one previous WAV/JSON pair (`*.previous`).
Queued conversations keep immutable copies of their selected references, so a
later regeneration does not change already queued speech. Generated metadata and
audio are local data, not source-controlled assets.

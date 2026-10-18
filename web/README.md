# GeneAI web

Phase 3 playback lab. Setup, the export contract, and honesty rules are in the repository README.

```bash
npm install
npm test
npm run build
npm start
```

Bundled replays live in `public/fixtures/`. They are trimmed mock-mode exports: the event `task` field is omitted, and `integrity_sha256` is still the engine hash of the full file.

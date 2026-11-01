# Development notes

## Agent / AI onboarding

Start with **[AGENTS.md](AGENTS.md)** — architecture diagrams and the image→GLB pipelines. **[CLAUDE.md](CLAUDE.md)** has setup commands and mesh gotchas. **[README.md](README.md)** is the product-facing MVP story.

## Hybrid PoC (Modal + photo bake)

Offline CLI: `poetry run python scripts/run_hybrid_poc.py --front <jpg> --back <jpg> --provider replicate|modal|hf|skip`.

Tokens live in `.env` (`HF_TOKEN`, `REPLICATE_API_TOKEN`). Copy from `.env-base`. Never commit `.env`.

One-time Modal Hugging Face secret (so the A10G worker can pull `tencent/Hunyuan3D-2mv`):

```bash
make modal-setup    # browser login, once
make modal-secret   # creates huggingface-secret from HF_TOKEN in .env
make modal-deploy   # first image build can take 20–40 min
make modal-prefetch # optional: warm the HF volume
```

`--provider skip --build frontend/assets/models/1-build.glb` runs extract + photo bake with no GPU. After a new hybrid GLB is written to `frontend/assets/models/<n>-hybrid.glb`, add `n` to `HAS_HYBRID` in `frontend/js/app.js`.

`--provider hf` talks to the TRELLIS.2 Space over HTTP (not `gradio_client`, which conflicts with the TripoSR-pinned transformers). If that API does not return a GLB, download one from the Space UI and pass `--provider skip --build`.

## D2 diagrams

Architecture sources: `_docs/diagrams/*.d2`  
Rendered assets: `_docs/diagrams/svg/*.svg` (referenced from AGENTS.md)

Install CLI (if needed):

```bash
curl -fsSL https://d2lang.com/install.sh | sh
```

Render one diagram:

```bash
d2 _docs/diagrams/overview.d2 _docs/diagrams/svg/overview.svg
```

### D2 MCP (Cursor)

Namespace: **`user-d2`**

| Tool | When to use |
|------|-------------|
| `compile-d2` | Validate syntax before committing `.d2` changes |
| `render-d2` | PNG/SVG/ASCII preview |
| `fetch_d2_cheat_sheet` | D2 syntax reference |

**Tip:** Pass diagram content via the `code` argument if `file_path` fails to read the repo from the MCP sandbox; the local `d2` CLI always reads workspace files.

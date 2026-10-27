# Claude Code guidance

Follow [AGENTS.md](AGENTS.md) for shared repository instructions, the wiki schema,
and ingest/query/lint workflows. Start at [wiki/index.md](wiki/index.md); use
[wiki/log.md](wiki/log.md) to recover recent knowledge changes.

Use the [README](README.md) and [PyTorch quickstart](docs/PYTORCH_QUICKSTART.md)
for setup and execution. The [architecture summary](wiki/architecture.md)
identifies public imports and paired NumPy/PyTorch modules. Test routes and
reproducibility requirements live in [validation](wiki/validation.md).

Before working on K3D HTML integration, read the
[HTML API notes](docs/visualization/k3d/api/HTML_JAVASCRIPT_API.md) and
[JavaScript patterns](docs/visualization/k3d/api/JAVASCRIPT_PATTERNS.md).
Existing findings include globally accessible HTML handlers, resolving a
promise-valued K3D instance, and checking the exported object/API before changing
camera, clipping, or volume data. Verify browser behavior for the actual export;
these notes do not guarantee compatibility across K3D versions.

Use the [visualization topic](wiki/visualization.md) for backend references and
historical performance findings. Keep reusable project knowledge there rather
than duplicating long examples or readiness claims in this instruction file.

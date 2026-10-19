# Changes

## Dielectric response

- Adds `--dielectric-method legacy|slab|sheet2d`, density/thickness/probe-q controls,
  anti-resonant response terms, and the opt-in `--long-wavelength` Hamiltonian flag.
- Slab and sheet runs write `omega`, `Re[eps]`, `Im[eps]`, `Re[alpha2D]`, and
  `Im[alpha2D]` columns; metadata and filenames identify the response method.
- Adds strict-2D K-boost caches, alpha2D plotting, long-wavelength-aware caches, and
  pipeline output-directory discovery for dielectric gather jobs.
- The default `--dielectric-method` is now `slab`. Scripts that relied on the old
  un-normalized output must pass `--dielectric-method legacy --antiresonant false`.
- Sheet2D cache format bumped to v2 (adds `ground_state_index_q`); v1 caches are
  recomputed automatically.

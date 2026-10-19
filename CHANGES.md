# Changes

## Dielectric response

- Adds `--dielectric-method legacy|slab|sheet2d`, density/thickness/probe-q controls,
  anti-resonant response terms, and the opt-in `--long-wavelength` Hamiltonian flag.
- Slab and sheet runs write `omega`, `Re[eps]`, `Im[eps]`, `Re[alpha2D]`, and
  `Im[alpha2D]` columns; metadata and filenames identify the response method.
- Adds strict-2D K-boost caches, alpha2D plotting, long-wavelength-aware caches, and
  pipeline output-directory discovery for dielectric gather jobs.

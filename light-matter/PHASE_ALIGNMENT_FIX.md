# Phase Alignment Issue in Initial Guess Vectors

## Problem

When using eigenvectors from a previous k-point as initial guesses for the next k-point, the band structure becomes noisy instead of smooth.

## Root Cause

### 1. Inconsistent Phase Between k-points

Eigenvectors of a Hermitian matrix have an **arbitrary global phase**. Two mathematically equivalent eigenvectors can differ by a factor of e^(iθ):

```
|ψ⟩ and e^(iθ)|ψ⟩ are the same physical state
```

When you compute eigenvectors at different k-points, SLEPc can return them with **different random phases**. This causes:

- **Phase discontinuities** between successive k-points
- **Poor initial guess quality** because the phases don't match
- **Noisy convergence** as the solver has to "fix" the phase mismatch

### 2. The Bug in the Original Code

In `create_petsc_vecs_from_array2`, lines 327-331:

```rust
// Copy data from the Array2 column to our temporary buffer
// Set real parts from initial_vecs, imaginary parts to zero  ← Comment says zero
for j in 0..vec_size {
    vec_data[2*j] = initial_vecs[[j, i]].re;     // Real part
    vec_data[2*j + 1] = initial_vecs[[j, i]].im; // But code copies im! ← BUG
}
```

The **comment** says to set imaginary parts to zero, but the **code** copies them from the previous eigenvector. This means:

- You're passing **unaligned phases** as initial guesses
- The phases from k-point n are likely **incompatible** with k-point n+1
- This creates a **bad starting point** that actually hurts convergence

## The Solution: Phase Alignment

### Why Phase Alignment Works

For a smooth band structure, we want eigenvectors to vary **continuously** with k. This requires choosing a **consistent gauge** (phase convention) across all k-points.

The standard approach is to align the phase so that:

1. **Maximum element is real and positive**: This provides a consistent reference
2. **Smooth variation**: Adjacent k-points will have similar phases
3. **Better initial guess**: The solver starts closer to the true solution

### Implementation

The fixed version:

```rust
// Find the element with largest magnitude
let mut max_mag = 0.0;
let mut max_idx = 0;
for j in 0..vec_size {
    let mag = initial_vecs[[j, i]].norm();
    if mag > max_mag {
        max_mag = mag;
        max_idx = j;
    }
}

// Calculate phase factor to rotate largest element to be real and positive
let phase_factor = if max_mag > 1e-14 {
    let max_elem = initial_vecs[[max_idx, i]];
    // Phase factor is e^(-i*arg(max_elem))
    c64::new(max_elem.re / max_mag, -max_elem.im / max_mag)
} else {
    c64::new(1.0, 0.0)
};

// Apply phase alignment
for j in 0..vec_size {
    let rotated = initial_vecs[[j, i]] * phase_factor;
    vec_data[2*j] = rotated.re;
    vec_data[2*j + 1] = rotated.im;
}
```

### How It Works

1. **Find reference**: Identify the element with largest magnitude
2. **Calculate rotation**: Compute e^(-iθ) where θ = arg(max_element)
3. **Apply rotation**: Multiply all components by this phase factor
4. **Result**: The largest element becomes real and positive

This ensures that:
- Each eigenvector has a **consistent phase reference**
- Adjacent k-points have **similar phases** (smooth variation)
- Initial guesses are **good approximations** of the true solution

## Alternative: Real-Only Initial Guess

For Hermitian problems, you can also just use the real parts:

```rust
for j in 0..vec_size {
    vec_data[2*j] = initial_vecs[[j, i]].re;
    vec_data[2*j + 1] = 0.0;  // Set imaginary part to zero
}
```

This works because:
- Hermitian matrices can have **real eigenvectors** (up to global phase)
- Real initial guesses avoid phase ambiguity entirely
- Often sufficient for band structure calculations

However, phase-aligned complex initial guesses are generally **better** because they preserve more information from the previous k-point.

## Mathematical Background

### Gauge Freedom

In quantum mechanics, physical states are equivalence classes:

```
[|ψ⟩] = {e^(iθ)|ψ⟩ : θ ∈ [0, 2π)}
```

When computing band structures, you must choose a **gauge** (phase convention). Common choices:

1. **Maximum element gauge**: Largest component is real positive (what we use)
2. **First element gauge**: First non-zero component is real positive
3. **Parallel transport**: Minimize phase change between k-points
4. **Real gauge**: Force eigenvectors to be real (when possible)

### Why Phase Matters for Initial Guesses

Consider two adjacent k-points:

- **k₁**: Eigenvector |ψ(k₁)⟩ with random phase θ₁
- **k₂**: True eigenvector |ψ(k₂)⟩ with optimal phase θ₂

If you use |ψ(k₁)⟩ as initial guess for k₂:

- **Without phase alignment**: Starting point is e^(iθ₁)|ψ(k₁)⟩
  - Phase mismatch: Δθ = θ₂ - θ₁ (random!)
  - Poor overlap: |⟨ψ(k₂)|e^(iθ₁)ψ(k₁)⟩|² reduced by phase error
  
- **With phase alignment**: Starting point is e^(iθ_opt)|ψ(k₁)⟩
  - Phase matched: Δθ ≈ 0 (smooth variation)
  - Good overlap: |⟨ψ(k₂)|e^(iθ_opt)ψ(k₁)⟩|² ≈ 1

This dramatically improves convergence speed and quality.

## Testing the Fix

To verify the fix works:

1. **Without initial guess**: Bands should be smooth (baseline)
2. **With initial guess (old code)**: Bands are noisy (the bug)
3. **With initial guess (fixed code)**: Bands should be smooth again

The phase alignment ensures that the initial guess actually **helps** convergence rather than hurting it.

## Additional Considerations

### For Non-Hermitian Problems

If your matrix is not Hermitian, eigenvectors may be genuinely complex. In this case:

- Phase alignment is **essential**
- Real-only initial guess won't work well
- Use the phase-aligned version

### For Degenerate Eigenvalues

If eigenvalues are degenerate (or nearly so):

- Eigenvectors can mix between k-points
- Phase alignment may not fully solve the issue
- Consider using **parallel transport** instead
- May need to track eigenvector "identity" across k-points

### Performance Impact

Phase alignment adds minimal overhead:

- O(n) operation per eigenvector (find max element)
- One complex multiplication per component
- Negligible compared to eigenvalue solve time

## Summary

**The Bug**: Copying imaginary parts without phase alignment

**The Fix**: Align phases by rotating largest element to real positive

**The Result**: Smooth, well-converged band structures with accelerated convergence

**Key Insight**: For initial guess vectors to be effective, they must be in a **consistent gauge** across k-points. Phase alignment provides this consistency.

# Box2D in JAX (JAX2D) Notes

## Summary of effort
- Implemented a JAX2D-based LunarLander environment matching Gymnasium's API and quirks where feasible.
- Built a sanity-check harness to compare Gymnasium vs JAX2D with aligned initial conditions (fixed terrain, no initial impulse, no wind, shared action sequences).
- Added timing and batched benchmarks to understand performance characteristics.

## Key findings
- **Exact matching is not feasible**: Gymnasium uses Box2D (C++), while JAX2D is a different physics engine. Even with identical initial conditions and actions, trajectories diverge due to different contact handling and solver details.
- **Reset alignment is possible**: Initial lander position/velocity and terrain can be aligned by fixing terrain height, disabling initial impulse, and disabling wind.
- **Single-env performance**: Box2D is much faster on CPU for small numbers of envs (expected due to optimized C++).
- **Batched performance**: JAX2D can be competitive or faster when running large batches of environments on GPU (e.g., 4096 envs), especially when the entire rollout is JIT-compiled.
- **Sparse contacts**: JAX2D uses dense arrays and masked computation, which is inefficient for sparse contacts compared to C++ engines that prune collision pairs.

## Conclusion
- For small numbers of environments and CPU execution, **Box2D (Gymnasium) remains the practical choice**.
- A JAX2D reimplementation is only justified if you need **large-scale vectorization** or **end-to-end JAX workflows**.

## Notes on benchmarking
- Benchmarking must separate **JIT compilation time** from **execution time**.
- GPU speedups appear only at sufficiently large batch sizes; small batches often underutilize the GPU.

## Artifacts
- Sanity/benchmark script: `tools/compare_lunarlander.py` (uses fixed terrain to align initial conditions).
- Saved plots (if headless): `tools/compare_lunarlander_obs.png`, `tools/compare_lunarlander_reward.png`.

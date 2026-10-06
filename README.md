# off-mgo

Experimental runtime for **offline/module-batched multi-GPU MoE CPU offloading**.

The initial target is Qwen3-30B-A3B on 4 H100 ranks. Several decode attention micro-batches are accumulated per layer into a large routed MoE batch, then active experts are materialized ephemerally across ranks for exact execution.

The first milestone is characterization rather than a fixed policy:
- measure H2D / dispatch / expert-compute / return critical-path fractions versus aggregated batch size;
- compare static EP, compute-only, communication-only, and joint single-copy placement;
- proceed to full module-batched decode only if the layer-level headroom is material.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for batching definitions, experiment grids, code reuse strategy, go/no-go criteria, and implementation order.

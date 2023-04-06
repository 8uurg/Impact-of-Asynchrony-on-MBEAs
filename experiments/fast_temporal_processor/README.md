# Fast Temporal Trace Processor
Processing the temporal traces produced by the simulator can be expensive. The current notebook re-computes the quantiles for each time value from scratch (of which there are potentially **millions**), which takes approximately $\Omega(n)$ time, and up to $\Omega(n \log n)$ time.

A lot of computational overlap between neighboring time points exists: exactly one data point has been added or removed between two records in the dataframe. It is possible to compute the quantiles over these time windows incrementally, updating based on the previous state using a heap, in $O(\log n)$ worst case, or given the maximum size of the heap $k$, where $k \leq n$: $O(\log k)$.

An exceedingly considerable speedup, considering that the heap is bounded by a constant in size.
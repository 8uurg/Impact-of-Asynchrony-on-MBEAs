# Fast Temporal Trace Processor
Processing the temporal traces produced by the simulator can be expensive. The current notebook re-computes the quantiles for each time value from scratch (of which there are potentially **millions**), which takes approximately $\Omega(n)$ time, and up to $\Omega(n \log n)$ time.

A lot of computational overlap between neighboring time points exists: exactly one data point has been added or removed between two records in the dataframe. It is possible to compute the quantiles over these time windows incrementally, updating based on the previous state using a heap, in $O(\log n)$ worst case, or given the maximum size of the heap $k$, where $k \leq n$: $O(\log k)$.

An exceedingly considerable speedup, considering that the heap is bounded by a constant in size.

# Credit
 DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability

This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.

Project leaders: Peter A.N. Bosman, Tanja Alderliesten
Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
Main code developer: Arthur Guijt

"""Helper functions for windowed graphs, surrogates, zigzag barcodes and graph null models.

Zigzag barcodes are computed with tdadyn (https://github.com/Semyon-ksa/tdadyn), which uses
Dionysus 2 as its zigzag backend; static Betti numbers are computed with Dionysus 2 directly.
"""
import numpy as np


def zz_window_starts(n_volumes, window_tr, stride_tr):
    """Window starts in REAL time -- never on a censoring-compacted series."""
    if window_tr > n_volumes:
        return np.empty(0, dtype=int)
    return np.arange(0, n_volumes - window_tr + 1, stride_tr, dtype=int)


def zz_window_censor_fraction(censor, starts, window_tr):
    """Censored fraction of each window; compare against a threshold to build the mask."""
    cum = np.concatenate([[0], np.cumsum(np.asarray(censor).astype(int))])
    return (cum[starts + window_tr] - cum[starts]) / float(window_tr)


def zz_window_correlations(ts, starts, window_tr):
    """Pearson correlation per window, explicitly symmetrised, zero diagonal.

    ts is [n_time, n_nodes]. Float asymmetry of ~1e-16 from corrcoef will fail
    downstream symmetry checks, hence the explicit symmetrisation.
    """
    ts = np.asarray(ts, dtype=np.float64)
    out = np.empty((len(starts), ts.shape[1], ts.shape[1]), dtype=np.float64)
    for i, s in enumerate(starts):
        c = np.corrcoef(ts[s:s + window_tr].T)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 0.0)
        out[i] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def zz_adjacency(corr, density=None, rule="density", k=None):
    """0/1 adjacency from one correlation matrix. Positive correlations only.

    rule: 'density' (top density*n_pairs edges), 'knn_or', 'knn_and'.
    kNN takes k, not density -- calibrate it with zz_calibrate_knn first, because
    a kNN graph at target density d does NOT have achieved density d.
    """
    n = corr.shape[0]
    if rule == "density":
        if density is None:
            raise ValueError("rule='density' needs density=")
        iu = np.triu_indices(n, 1)
        vals = corr[iu]
        keep_n = max(1, min(int(round(density * vals.size)), vals.size))
        order = np.argsort(vals)[::-1][:keep_n]
        order = order[vals[order] > 0.0]
        mask = np.zeros(vals.size, dtype=bool)
        mask[order] = True
        a = np.zeros_like(corr)
        a[iu] = mask
        a = np.maximum(a, a.T)
        np.fill_diagonal(a, 0.0)
        return a
    if k is None:
        raise ValueError("kNN rules need k=; use zz_calibrate_knn")
    work = np.where(corr > 0.0, corr, -np.inf)
    np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.repeat(np.arange(n), k), idx.ravel()] = True
    directed &= np.isfinite(work)
    a = (directed | directed.T) if rule == "knn_or" else (directed & directed.T)
    out = a.astype(np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def zz_calibrate_knn(corr_stack, target_density, rule="knn_or", k_grid=None):
    """Pick k so the kNN graph ACHIEVES target_density, and report what it achieves.

    Comparing kNN against a density threshold at matched TARGET density credits the
    denser graph with extra edge churn it gets for free. Match achieved density.
    """
    if k_grid is None:
        k_grid = list(range(2, 61))
    n = corr_stack.shape[1]
    n_pairs = n * (n - 1) / 2.0
    best = None
    for k in k_grid:
        dens = np.mean([zz_adjacency(c, rule=rule, k=k).sum() / 2.0 / n_pairs
                        for c in corr_stack])
        gap = abs(dens - target_density)
        if best is None or gap < best[2]:
            best = (k, dens, gap)
    return {"k": best[0], "achieved_density": best[1], "target_density": target_density}


def zz_graph_diagnostics(adj_stack):
    """Per-step edge switching and the isolated-node channel that fakes H0 events."""
    adj_stack = np.asarray(adj_stack)
    n = adj_stack.shape[1]
    iu = np.triu_indices(n, 1)
    e = adj_stack[:, iu[0], iu[1]] > 0
    isolated = adj_stack.sum(2) == 0
    return {"n_windows": len(adj_stack),
            "achieved_density": float(e.sum(1).mean() / e.shape[1]),
            "median_switches": float(np.median((e[:-1] ^ e[1:]).sum(1))),
            "mean_isolated": float(isolated.sum(1).mean()),
            "isolated_flips_per_step": float((isolated[:-1] ^ isolated[1:]).sum(1).mean()),
            "ever_isolated": int(isolated.any(0).sum())}


def zz_phase_randomise(ts, rng=None, shared_phase=True):
    """Pritchard-Theiler surrogate. shared_phase=True is the method.

    One random phase offset per frequency, applied to ALL channels, preserves each
    channel's spectrum AND the static cross-covariance while destroying
    non-stationarity. Independent phases per channel (shared_phase=False) destroy
    cross-covariance too -- that surrogate resembles nothing and makes any test
    trivially significant. It is here only as a negative control.
    """
    if rng is None:
        rng = np.random.default_rng()
    ts = np.asarray(ts, dtype=np.float64)
    n_time = ts.shape[0]
    spec = np.fft.rfft(ts, axis=0)
    n_freq = spec.shape[0]
    hi = n_freq - 1 if n_time % 2 == 0 else n_freq
    if shared_phase:
        phases = np.zeros(n_freq)
        phases[1:hi] = rng.uniform(0.0, 2.0 * np.pi, size=hi - 1)
        if n_time % 2 == 0:
            phases[-1] = rng.choice([0.0, np.pi])
        rotated = spec * np.exp(1j * phases)[:, None]
    else:
        phases = np.zeros((n_freq, ts.shape[1]))
        phases[1:hi] = rng.uniform(0.0, 2.0 * np.pi, size=(hi - 1, ts.shape[1]))
        rotated = spec * np.exp(1j * phases)
    return np.fft.irfft(rotated, n=n_time, axis=0)


def zz_barcode_statistics(adj_stack, max_homology_dim=1, max_nodes=400):
    """Zigzag barcode of a 0/1 adjacency sequence, via tdadyn + Dionysus.

    Uses tdadyn's UNION_RULE (union of complexes, not flag complex of the union of
    graphs) and its event conventions. Returns barcode summary statistics and timing;
    the event log is deliberately NOT returned as a result -- on data without ground
    truth it is uninterpretable, and the group/surrogate contrast is the instrument.
    """
    import time
    from tdadyn.core.contracts import adjacency_from_snapshot
    from tdadyn.reference import flag_complex
    from tdadyn.zigzag import union_complex, barcode_from, index_set, event_log

    t0 = time.perf_counter()
    complexes = [flag_complex(adjacency_from_snapshot(a, edge_rule="weight_above",
                                                      edge_threshold=0.5),
                              max_homology_dim=max_homology_dim, max_nodes=max_nodes)
                 for a in adj_stack]
    t_flag = time.perf_counter() - t0
    diagram = []
    for i, c in enumerate(complexes):
        diagram.append(c)
        if i + 1 < len(complexes):
            diagram.append(union_complex(c, complexes[i + 1]))
    t0 = time.perf_counter()
    barcodes = barcode_from(diagram, backend="dionysus",
                            max_homology_dim=max_homology_dim)
    t_zigzag = time.perf_counter() - t0
    log = event_log(barcodes, indices=index_set(len(complexes)))
    stats = {"n_snapshots": len(complexes), "n_positions": len(diagram),
             "invisible_bars": log.invisible_bars, "censored_ends": log.censored_ends,
             "t_flag_s": t_flag, "t_zigzag_s": t_zigzag,
             "mean_complex_size": float(np.mean([sum(len(c.faces[d]) for d in c.faces)
                                                 for c in complexes]))}
    for dim, bars in barcodes.items():
        lengths = [death - birth + 1 for birth, death in bars]
        stats["n_bars_h%d" % dim] = len(bars)
        stats["total_length_h%d" % dim] = int(sum(lengths))
    return stats


def zz_surrogate_test(real_value, surrogate_values):
    """Two-sided rank p (Davison-Hinkley) and position relative to the surrogate range.

    With m surrogates the smallest attainable p is 2/(m+1), and the chance of the
    real value falling outside the surrogate RANGE under the null is also 2/(m+1) --
    compare observed exceedance against that, not against zero.
    """
    v = np.asarray(surrogate_values, dtype=np.float64)
    centre = v.mean()
    sd = v.std(ddof=1)
    p = (1 + int(np.sum(np.abs(v - centre) >= abs(real_value - centre)))) / (len(v) + 1)
    return {"real": float(real_value), "surr_mean": float(centre), "surr_sd": float(sd),
            "surr_min": float(v.min()), "surr_max": float(v.max()),
            "z": float((real_value - centre) / sd) if sd > 0 else float("nan"),
            "outside_range": bool(real_value < v.min() or real_value > v.max()),
            "p_two_sided": p, "p_floor": 2.0 / (len(v) + 1)}


def zz_degree_null(adjacency, rng, n_swap_per_edge=5):
    """One degree-preserving rewiring of a binary graph (double edge swap).

    The only null worth running on a thresholded connectivity graph: nulls that
    do not preserve degree (Erdos-Renyi, distance-preserving) are rejected by
    every subject and test nothing. beta_0 is absorbed by this null -- that is
    the finding, not a failure -- while beta_1 is not.
    """
    import networkx as nx
    import numpy as _np
    A = (_np.asarray(adjacency) > 0).astype(int)
    g = nx.from_numpy_array(A)
    n_edges = g.number_of_edges()
    if n_edges > 1:
        nx.double_edge_swap(g, nswap=n_swap_per_edge * n_edges,
                            max_tries=200 * n_swap_per_edge * n_edges,
                            seed=int(rng.integers(1_000_000_000)))
    return nx.to_numpy_array(g)


def zz_flag_betti(adjacency, max_dim=2):
    """beta_0 and beta_1 of the FLAG COMPLEX of a binary graph.

    Never use the cyclomatic number E - V + beta_0 in place of beta_1: that is
    the cycle rank of the graph, and on connectivity graphs it overestimates the
    flag-complex beta_1 by more than an order of magnitude (847 against 21 at
    density 0.10 in our cohort), because the triangles that fill those cycles
    are exactly what the flag complex adds.
    """
    import dionysus as _dio
    import numpy as _np
    from scipy.spatial.distance import squareform as _sf
    A = _np.asarray(adjacency)
    D = _np.where(A > 0, 0.5, 2.0)
    _np.fill_diagonal(D, 0.0)
    f = _dio.fill_rips(_sf(D, checks=False).astype("float32"), max_dim, 1.0)
    dgms = _dio.init_diagrams(_dio.homology_persistence(f), f)
    def _b(dim):
        if dim >= len(dgms):
            return 0
        return int(sum(1 for p in dgms[dim] if p.birth <= 0.5 < p.death))
    return {"beta0": _b(0), "beta1": _b(1)}

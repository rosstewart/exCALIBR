"""Array-task packing and in-task submission order (scheduling only).

Which array task a compact job (one dataset x bootstrap, all K x restarts)
lands in, and the order a task submits its fits to its process pool, cannot
change any fit: every fit is seeded by its own job kwargs and best-fit
selection breaks val_ll ties on fit_idx (hpc/run_array_task.py), so results
are bitwise the same in any placement/order on the CPU path. (On the GPU path
a task's fits are batched together, so placement changes batch composition,
which may move results at floating-point level.)

Where it is used:
- hpc/prepare.py writes per-compact-job fit-cost estimates (jobs/costs.json)
  and a balanced default layout of the array files;
- hpc/make_layout.py (called by submit_array.sh when TASKS is set) lays the
  compact jobs out for the machine at submission: task count, CPUs per task
  and concurrency are submission-time choices;
- hpc/run_array_task.py submits each task's slowest fits first.

Cost model
----------
Estimated fit seconds (relative units are all that matter for packing):

    sec/iter ~= exp(A) * K * N**B * p**C * MODE[mode]
    iters    ~= ITER_FRACTION[mode] * max_em_iters

with N training rows, p readout columns, K components, mode = gaussian /
q1 / q2 restart. Fitted on the 48 plain criteria-D production fits (BRCA1
functional, BRCA2/TSC2 combined, TP53 functional; K = 3, 4; see
docs/fit_speed_report.md): Spearman 0.83 against measured fit seconds; the
K exponent is fixed at 1 (40-iteration timings at K = 2..6 scale ~linearly).
A greedy-restart seed is a chain of 1 + n_slots sequential fits, so its cost
is multiplied accordingly.
"""
import heapq
import math

A, B, C = -6.491, 0.363, 0.245
MODE = {"gaussian": 1.0, "q1": 0.476, "q2": 1.462}
ITER_FRACTION = {"gaussian": 0.35, "q1": 0.9, "q2": 1.0}


def _mode(job):
    kw = job.get("kwargs") or {}
    if kw.get("force_gaussian"):
        return "gaussian"
    q = kw.get("latent_q", job.get("nominal_latent_q", 2))
    return "q1" if q == 1 else "q2"


def fit_cost(job, n_rows, n_cols, multivariate=True):
    """Estimated seconds for one fit (or one greedy chain) of a minimal or
    full job dict, given the training-data shape."""
    kw = job.get("kwargs") or {}
    K = int(job.get("num_components", 2))
    mode = _mode(job) if multivariate else "q2"
    cap = kw.get("max_em_iters", 50000 if multivariate else 10000)
    per_iter = math.exp(A) * K * max(n_rows, 1) ** B * max(n_cols, 1) ** C * MODE[mode]
    cost = per_iter * ITER_FRACTION[mode] * cap
    gm = job.get("greedy_meta")
    if gm is not None and gm.get("mode") != "gaussian":
        cost *= 1 + int(gm.get("n_slots", 0))
    return cost


def _shape(cjob):
    x = (cjob.get("shared_data") or {}).get("train_observations")
    if x is None:
        return 1, 1
    shp = getattr(x, "shape", None) or (len(x), len(x[0]) if len(x) else 1)
    return int(shp[0]), int(shp[1]) if len(shp) > 1 else 1


def cjob_fit_costs(cjob):
    """Estimated cost of every fit (greedy chain) in one compact job."""
    n, p = _shape(cjob)
    mv = bool(cjob.get("multivariate", False))
    out = []
    for k, lst in cjob.items():
        if k.startswith("jobs_") and isinstance(lst, list):
            out += [fit_cost(j, n, p, mv) for j in lst]
    return out


def makespan(costs, workers):
    """Longest-processing-time-first list schedule on `workers` CPUs."""
    heap = [0.0] * max(1, workers)
    for c in sorted(costs, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + c)
    return max(heap) if costs else 0.0


def assign(cjob_costs, n_tasks, cpus=None):
    """LPT-assign compact jobs (given as lists of fit costs) to n_tasks bins
    by total estimated work, then order the bins longest first: by predicted
    wall time on `cpus` CPUs if given, else by total work (SLURM starts array
    tasks in index order, so this approximates longest-first across the
    array). Each bin lists its compact jobs in original order."""
    n_tasks = min(max(1, n_tasks), max(1, len(cjob_costs)))
    order = sorted(range(len(cjob_costs)), key=lambda i: -sum(cjob_costs[i]))
    heap = [(0.0, t) for t in range(n_tasks)]
    bins = [[] for _ in range(n_tasks)]
    for i in order:
        load, t = heapq.heappop(heap)
        bins[t].append(i)
        heapq.heappush(heap, (load + sum(cjob_costs[i]), t))
    bins = [sorted(b) for b in bins if b]
    if cpus:
        key = [makespan([c for i in b for c in cjob_costs[i]], cpus) for b in bins]
    else:
        key = [sum(sum(cjob_costs[i]) for i in b) for b in bins]
    return [b for _, b in sorted(zip(key, bins), key=lambda kb: -kb[0])]


def run_wall(task_walls, slots):
    """Whole-array wall time when SLURM runs tasks in index order on at most
    `slots` concurrent tasks."""
    heap = [0.0] * max(1, min(slots, len(task_walls)))
    for w in task_walls:
        heapq.heappush(heap, heapq.heappop(heap) + w)
    return max(heap) if task_walls else 0.0


def evaluate(bins, cjob_costs, cpus, slots=None):
    """Predicted per-task wall times, CPU use within tasks, reserved CPU time
    and (if `slots` is given) whole-array wall time with tasks started in
    index order, `slots` at a time."""
    spans = [makespan([c for i in b for c in cjob_costs[i]], cpus) for b in bins]
    work = sum(sum(c) for c in cjob_costs)
    reserved = cpus * sum(spans)
    return dict(n_tasks=len(bins), cpus=cpus, max_task=max(spans) if spans else 0.0,
                utilisation=work / reserved if reserved else 0.0, reserved=reserved, work=work,
                run_wall=run_wall(spans, slots) if slots else None)


def choose_tasks(cjob_costs, cpus, slots, tol=0.02):
    """Task count minimising the predicted whole-array wall time for `slots`
    concurrent tasks of `cpus` CPUs; among counts within `tol` of the best,
    the one reserving the fewest CPU-hours. Returns (bins, stats)."""
    n = len(cjob_costs)
    if n == 0:
        return [], evaluate([], cjob_costs, cpus, slots)
    cands = sorted({min(n, max(1, int(round(slots * f)))) for f in
                    (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)} | {n})
    res = []
    for k in cands:
        bins = assign(cjob_costs, k, cpus)
        res.append((evaluate(bins, cjob_costs, cpus, slots), bins))
    best = min(st["run_wall"] for st, _ in res)
    st, bins = min(((st, b) for st, b in res if st["run_wall"] <= (1 + tol) * best),
                   key=lambda sb: sb[0]["reserved"])
    return bins, st


def suggest_cpus(cjob_costs, bins, options=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64), slack=1.1):
    """Smallest CPUs-per-task whose predicted longest task is within `slack`
    of the bound set by the single slowest fit chain (adding CPUs beyond this
    mostly reserves idle cores). Returns (cpus, {cpus: stats})."""
    floor = max((max(cjob_costs[i]) for b in bins for i in b if cjob_costs[i]), default=0.0)
    table = {c: evaluate(bins, cjob_costs, c) for c in options}
    for c in options:
        if table[c]["max_task"] <= slack * floor:
            return c, table
    return options[-1], table


def contiguous(n_jobs, jobs_per_array):
    """The legacy layout: consecutive slices of jobs_per_array compact jobs."""
    return [list(range(s, min(s + jobs_per_array, n_jobs))) for s in range(0, n_jobs, jobs_per_array)]

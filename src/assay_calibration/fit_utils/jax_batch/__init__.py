"""GPU-batched EM fitting (JAX), an opt-in alternative to the per-job NumPy
path in ``cfusn/fit.py`` / ``cfusn/update_steps.py``.

Batches many independent bootstrap/fit-idx jobs for the same dataset into a
single vectorized computation (leading ``batch`` axis) instead of running one
fit per CPU process. See ``slurm/README.md`` and the plan this was built
from for the batching rationale.

Untested on GPU as of authoring — see ``tests/test_batch_em_parity.py`` and
validate against the NumPy reference before trusting production output.
"""

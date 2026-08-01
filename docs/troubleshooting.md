← [Back to README](../README.md)

# Troubleshooting

**"Insufficient samples"**
- Need at least 3 sample categories
- Check for empty samples (all NaN scores)

**SLURM jobs fail**
- Verify `SLURM_ACCOUNT` and `SLURM_PARTITION` match your cluster
- Check `module avail anaconda` and set `CONDA_MODULE` accordingly
- Check logs in `<output_dir>/logs/`

**Low number of valid fits**
- Increase `--n-bootstraps` or `--fits-per-bootstrap`
- Check for score range issues (all variants at same score)

**GPU runs (JAX)**
- Requires `excalibr-gpu.yml` environment (Python 3.11 + JAX)
- Test: `python -c "import jax; print(jax.devices())"`
- Use `CUDA_VISIBLE_DEVICES=0` to pin to a specific GPU

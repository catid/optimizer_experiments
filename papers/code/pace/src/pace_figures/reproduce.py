"""Expand a paper-figure config into the ``pace-train`` commands that produce its runs.

By default the commands are printed. Pass ``run=true`` (with ``data=<prepared dataset>``)
to execute them and write the result JSONs.

    pace-figure fig=list            # list the figures
    pace-figure fig=fig1            # print the pace-train commands for Figure 1
    pace-figure fig=fig1 run=true data=/path/to/smoltalk   # train them and write results

Each figure config (``conf/fig/<id>.yaml``) lists the runs behind that figure; every run
is a complete ``pace-train`` spec, so a printed command can be run directly.
"""

import os
import subprocess
import sys
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]   # src/pace_figures -> src -> repo root
FIGMAP_PATH = _HERE / "FIGMAP.yaml"


def _print_map():
    figmap = yaml.safe_load(open(FIGMAP_PATH))
    print(f"{'fig=<id>':38} {'paper':8} description")
    print("-" * 90)
    for fid, info in figmap.items():
        print(f"{fid:38} {info['paper']:8} {info['desc']}")
    print(f"\n{len(figmap)} figures catalogued. Print one figure's run configs with:  "
          f"pace-figure fig=<id>")


def _command(fig, curve, extra):
    o = curve["optimizer"]
    parts = [
        "pace-train",
        f"model={curve['model']}",
        f"dataset={fig.get('dataset', 'smoltalk')}",
        f"optimizer={o['name']}",
        f"schedule={curve['schedule']}",
        f"seed={curve['seed']}",
        f"optimizer.lr={o['lr']}",
        f"optimizer.lambda_pullback={o.get('lambda_pullback', 0.0)}",
    ]
    if o.get("ema_kappa") is not None:
        parts.append(f"optimizer.ema_kappa={o['ema_kappa']}")
    if o.get("ema_update_freq") is not None:
        parts.append(f"optimizer.ema_update_freq={o['ema_update_freq']}")
    return parts + list(extra)


def _expand_sweep(sw):
    """Expand a `sweep` block (a `grid` cartesian product over a `base`) into runs.

    Grid keys may be dotted (`optimizer.lr`) or plain (`schedule`, `seed`, `model`).
    """
    import itertools
    base = sw.get("base", {})
    grid = {k: list(v) for k, v in (sw.get("grid", {}) or {}).items()}
    keys = list(grid)
    combos = itertools.product(*[grid[k] for k in keys]) if keys else [()]
    out = []
    for combo in combos:
        c = {"model": sw["model"], "method": base.get("method", "PACE"),
             "schedule": base.get("schedule", "const"), "seed": base.get("seed", 42),
             "optimizer": dict(base.get("optimizer", {}))}
        for k, v in zip(keys, combo):
            if k.startswith("optimizer."):
                c["optimizer"][k.split(".", 1)[1]] = v
            elif k in ("schedule", "seed", "model", "method"):
                c[k] = v
            else:
                c["optimizer"][k] = v
        out.append(c)
    return out


def expand_figure(fig):
    """Return all runs for a figure config: explicit ``curves`` plus expanded sweeps."""
    curves = list(fig.get("curves", []) or [])
    sweeps = ([fig["sweep"]] if fig.get("sweep") else []) + list(fig.get("sweeps", []) or [])
    for sw in sweeps:
        curves += _expand_sweep(sw)
    return curves


@hydra.main(version_base=None, config_path="conf", config_name="reproduce")
def main(cfg: DictConfig):
    fig = cfg.fig
    if fig.id == "list":
        _print_map()
        return

    curves = expand_figure(fig)
    if not curves:
        print(f"No run specs for '{fig.id}' yet — see the README or `pace-figure fig=list`.")
        return

    extra = []
    if cfg.get("data"):
        extra.append(f"dataset.path={cfg.data}")
    if cfg.get("max_steps"):
        extra.append(f"training.max_steps={cfg.max_steps}")
    if cfg.get("eval_samples"):
        extra.append(f"training.eval_samples={cfg.eval_samples}")
    if cfg.get("eval_interval"):
        extra.append(f"training.eval_interval={cfg.eval_interval}")
    run = bool(cfg.get("run", False))
    limit = cfg.get("limit")

    print(f"# {fig.id} ({fig.get('paper', '')}): {fig.get('desc', '')} — {len(curves)} runs\n")
    out_root = Path(cfg.get("out") or (Path.cwd() / "runs" / fig.id))
    # When running from a source checkout (not pip-installed), make `pace` importable
    # for the child process; harmless when the package is installed
    src = REPO / "src"
    base_env = dict(os.environ)
    if src.is_dir():
        base_env["PYTHONPATH"] = str(src) + os.pathsep + base_env.get("PYTHONPATH", "")
    for i, c in enumerate(curves):
        cmd = _command(fig, c, extra)
        print(" ".join(cmd))
        if run and (limit is None or i < limit):
            label = f"{c['model']}_{c['method']}_{c['schedule']}_s{c['seed']}"
            env = dict(base_env, PACE_RUN_DIR=str(out_root / label))
            subprocess.run([sys.executable, "-m", "pace.helpers.train"] + cmd[1:], env=env, check=True)

    if not run:
        ds = fig.get("dataset", "smoltalk")
        print(f"\n# Run these (after preparing the {ds} dataset, see README) with, e.g.:")
        print(f"#   pace-figure fig={fig.id} run=true data=/path/to/{ds}")
        return


if __name__ == "__main__":
    main()

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from evo import CONFIG, build_tag, parse_float_list, run_evolution


def main() -> None:
    r_values = [value / 10.0 for value in range(21, 51)]
    output_dir = Path("data/evolution/r_scan_2p1_5p0")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.perf_counter()
    print(f"running {len(r_values)} r values: {r_values[0]:.1f} to {r_values[-1]:.1f}", flush=True)
    print(f"output dir: {output_dir}", flush=True)

    contribution_actions = parse_float_list(CONFIG.contribution_actions)
    punishment_actions = parse_float_list(CONFIG.punishment_actions)
    did_warmup = False
    for idx, r_value in enumerate(r_values, start=1):
        base_params = replace(CONFIG, r=r_value, output_dir=output_dir)
        tag = build_tag(base_params, contribution_actions, punishment_actions)
        csv_path = output_dir / f"{tag}.csv"
        if csv_path.exists():
            print(f"\n[{idx}/{len(r_values)}] r={r_value:.1f} skip existing: {csv_path}", flush=True)
            continue

        print(f"\n[{idx}/{len(r_values)}] r={r_value:.1f} start", flush=True)
        start = time.perf_counter()
        params = replace(
            base_params,
            warmup=not did_warmup,
        )
        csv_path, figure_paths = run_evolution(params)
        did_warmup = True
        elapsed = time.perf_counter() - start
        print(f"[{idx}/{len(r_values)}] r={r_value:.1f} done: {elapsed:.2f}s", flush=True)
        print(f"csv: {csv_path}", flush=True)
        for path in figure_paths:
            print(f"plot: {path}", flush=True)

    print(f"\nall done: {time.perf_counter() - start_all:.2f}s", flush=True)


if __name__ == "__main__":
    main()

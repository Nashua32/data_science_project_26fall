"""Generate local, open-loop identification data using the Williams-Otto reactor.

Run from the project root::

    python -m src.simulation.data_generator_n4sid

Or run this file directly. Outputs default to ``data/n4sid``. Two independent
random-step experiments are generated for estimation and validation. No NARX
lags are added: N4SID takes synchronized U (N, 2), Y (N, 6), and sample time.

Time is in minutes, QB in dm3/min, Tr in kelvin, and concentrations in mol/dm3.
At row k, Y[k] is the concentration at t[k]; U[k] is applied on [t[k], t[k+1]).
The final input is held for one additional sample interval. There is no direct
feedthrough from the reactor inputs to the measured concentrations.

Defaults target a local linear approximation: QB = 270 +/- 18 dm3/min and
Tr = 360.65 +/- 2 K. Adjust amplitudes and dwell times for your application,
then validate the resulting linear model; excitation alone does not guarantee
that a particular N4SID order or horizon is identifiable.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

if __package__:
    from .william_otto_reactor_model import WilliamsOttoReactor
else:
    from william_otto_reactor_model import WilliamsOttoReactor


INPUT_NAMES = ("QB", "Tr")
OUTPUT_NAMES = ("cA", "cB", "cC", "cP", "cE", "cG")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def random_steps(n_samples, center, amplitude, hold_steps, rng):
    """Independent multilevel random steps, with randomized hold durations.

    This is random-step excitation, not a maximal-length binary PRBS. Each
    channel draws its own amplitudes and dwell times from its own RNG stream.
    """
    signal = np.empty(n_samples)
    start = 0
    while start < n_samples:
        stop = min(n_samples, start + int(rng.choice(hold_steps)))
        signal[start:stop] = center + amplitude * rng.uniform(-1.0, 1.0)
        start = stop
    return signal


def simulate_experiment(reactor, U, dt, y0):
    """Integrate each constant-input interval, including every discontinuity."""
    switches = np.flatnonzero(np.any(np.diff(U, axis=0) != 0, axis=1)) + 1
    boundaries = np.r_[0, switches, len(U)]
    Y = np.empty((len(U), 6))
    state = np.asarray(y0, dtype=float).copy()
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        held_input = tuple(U[start])
        times = np.arange(start, stop + 1, dtype=float) * dt
        solution = solve_ivp(
            lambda t, c: reactor.odes(t, c, lambda _: held_input),
            (times[0], times[-1]),
            state,
            t_eval=times,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(f"Integration failed at {times[0]} min: {solution.message}")
        Y[start:stop] = solution.y[:, :-1].T
        state = solution.y[:, -1]
    if not np.isfinite(Y).all() or np.min(Y) < -1e-8:
        raise RuntimeError("Simulation produced nonfinite or negative concentrations.")
    return Y


def generate_dataset(
    n_samples=12000,
    dt=0.1,
    seed=42,
    operating_input=(270.0, 360.65),
    amplitudes=(18.0, 2.0),
    hold_times=(0.5, 1.0, 2.0, 4.0, 8.0),
    noise_std=0.0,
):
    """Return raw and operating-point-centered data without writing files.

    ``noise_std`` is additive Gaussian output measurement noise in mol/dm3.
    Noise is not clipped, since clipping would bias measurements near zero.
    No process noise or actuator dynamics are added. Centering uses the known
    nominal equilibrium, equally for training and validation (no data leakage).
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative.")
    u0 = np.asarray(operating_input, dtype=float)
    amplitude = np.asarray(amplitudes, dtype=float)
    if u0.shape != (2,) or amplitude.shape != (2,):
        raise ValueError("operating_input and amplitudes must each have two entries.")
    if not np.isfinite(u0).all() or not np.isfinite(amplitude).all() or np.any(amplitude <= 0):
        raise ValueError("Operating inputs must be finite and amplitudes positive.")
    reactor = WilliamsOttoReactor()
    bounds = np.array([reactor.QB_bounds, reactor.Tr_bounds])
    if np.any(u0 - amplitude < bounds[:, 0]) or np.any(u0 + amplitude > bounds[:, 1]):
        raise ValueError("Excitation exceeds the reactor's QB or Tr bounds.")
    holds = np.asarray(hold_times, dtype=float)
    if holds.ndim != 1 or holds.size == 0 or not np.isfinite(holds).all() or np.any(holds < dt):
        raise ValueError("hold_times must be a nonempty sequence of finite times >= dt.")
    hold_steps = np.rint(holds / dt).astype(int)
    if not np.allclose(hold_steps * dt, holds, rtol=1e-9, atol=1e-12):
        raise ValueError("Each hold time must be an integer multiple of dt.")
    y0 = reactor.find_steady_state(*u0)
    residual = reactor.odes(0.0, y0, lambda _: tuple(u0))
    if not np.isfinite(y0).all() or np.any(y0 < 0) or np.max(np.abs(residual)) > 1e-7:
        raise RuntimeError("Nominal steady state is not a valid physical equilibrium.")
    streams = np.random.SeedSequence(seed).spawn(3)
    U = np.column_stack([
        random_steps(n_samples, u0[j], amplitude[j], hold_steps,
                     np.random.default_rng(streams[j]))
        for j in range(2)
    ])
    Y_clean = simulate_experiment(reactor, U, dt, y0)
    Y = Y_clean + np.random.default_rng(streams[2]).normal(0.0, noise_std, Y_clean.shape)
    return {
        "time_min": np.arange(n_samples) * dt,
        "U": U,
        "Y": Y,
        "Y_clean": Y_clean,
        "U_centered": U - u0,
        "Y_centered": Y - y0,
        "u_operating": u0,
        "y_operating": y0,
        "Ts_min": np.array(dt),
        "input_names": np.array(INPUT_NAMES),
        "output_names": np.array(OUTPUT_NAMES),
    }


def save_dataset(dataset, output_dir, name):
    """Save readable raw CSV and N4SID-ready NumPy arrays with metadata."""
    frame = pd.DataFrame(dataset["Y"], columns=OUTPUT_NAMES)
    frame.insert(0, "time_min", dataset["time_min"])
    for j, channel in enumerate(INPUT_NAMES):
        frame[channel] = dataset["U"][:, j]
    frame.to_csv(output_dir / f"{name}.csv", index=False)
    np.savez_compressed(output_dir / f"{name}.npz", **dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "n4sid")
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--validation-samples", type=int, default=6000)
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval in minutes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qb-mean", type=float, default=270.0)
    parser.add_argument("--tr-mean", type=float, default=360.65, help="Kelvin.")
    parser.add_argument("--qb-amplitude", type=float, default=18.0)
    parser.add_argument("--tr-amplitude", type=float, default=2.0, help="Kelvin.")
    parser.add_argument("--hold-times", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0], help="Input dwell times in minutes.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Output noise standard deviation in mol/dm3.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.train_samples < 2 or args.validation_samples < 2:
        parser.error("Both sample counts must be >= 2")
    options = dict(
        dt=args.dt,
        operating_input=(args.qb_mean, args.tr_mean),
        amplitudes=(args.qb_amplitude, args.tr_amplitude),
        hold_times=args.hold_times,
        noise_std=args.noise_std,
    )
    # Independent experiments start at the same equilibrium. Do not concatenate
    # them as a single continuous trajectory when fitting an identification model.
    train = generate_dataset(n_samples=args.train_samples, seed=args.seed, **options)
    validation = generate_dataset(n_samples=args.validation_samples, seed=args.seed + 1, **options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataset in (("train", train), ("validation", validation)):
        save_dataset(dataset, args.output_dir, name)
        print(f"{name}: U {dataset['U'].shape}, Y {dataset['Y'].shape}")
    metadata = {
        "sample_time_unit": "min",
        "input_units": ["dm3/min", "K"],
        "output_units": ["mol/dm3"] * 6,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "train_seed": args.seed,
        "validation_seed": args.seed + 1,
        "excitation": "Independent uniform random steps with randomized dwell times",
        "alignment": "Y[k] at t[k]; U[k] held from t[k] to t[k+1]",
        "centering": "Subtract the shared nominal input and equilibrium output; no scaling",
        "n4sid_usage": "Use U_centered, Y_centered and Ts_min; no precomputed lags",
        "reactor_parameters": WilliamsOttoReactor.DEFAULT_PARAMS,
        **options,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved CSV, NPZ and metadata to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

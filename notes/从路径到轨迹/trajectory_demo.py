import csv
import math
from pathlib import Path

DELTA = (0.4, 0.2)
VMAX = (0.30, 0.25)
AMAX = (0.60, 0.50)
JMAX = (2.00, 2.00)
KV, KA, KJ = 15 / 8, 10 * math.sqrt(3) / 3, 60


def required_time(delta, vmax, amax, jmax):
    distance = abs(delta)
    return max(KV * distance / vmax,
               math.sqrt(KA * distance / amax),
               (KJ * distance / jmax) ** (1 / 3))


def state(delta, duration, u):
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = 30 * u**2 * (1 - u)**2
    dds = 60 * u - 180 * u**2 + 120 * u**3
    ddds = 60 - 360 * u + 360 * u**2
    return (delta * s, delta * ds / duration,
            delta * dds / duration**2, delta * ddds / duration**3)


def peaks(delta, duration):
    return (KV * abs(delta) / duration,
            KA * abs(delta) / duration**2,
            KJ * abs(delta) / duration**3)


if __name__ == "__main__":
    axis_times = [required_time(d, v, a, j)
                  for d, v, a, j in zip(DELTA, VMAX, AMAX, JMAX)]
    duration = max(axis_times)
    print(f"common duration = {duration:.6f} s")
    for i, delta in enumerate(DELTA):
        v, a, j = peaks(delta, duration)
        assert v <= VMAX[i] + 1e-12
        assert a <= AMAX[i] + 1e-12
        assert j <= JMAX[i] + 1e-12
        print(f"axis {i}: peak v={v:.6f}, a={a:.6f}, j={j:.6f}")
        for u, expected in ((0.0, 0.0), (1.0, delta)):
            q, qd, qdd, _ = state(delta, duration, u)
            assert abs(q - expected) < 1e-12
            assert abs(qd) < 1e-12 and abs(qdd) < 1e-12

    rows = []
    for k in range(1001):
        u = k / 1000
        x = state(DELTA[0], duration, u)
        y = state(DELTA[1], duration, u)
        assert abs(y[0] - 0.5 * x[0]) < 1e-12
        for i, values in enumerate((x, y)):
            assert abs(values[1]) <= VMAX[i] + 1e-12
            assert abs(values[2]) <= AMAX[i] + 1e-12
            assert abs(values[3]) <= JMAX[i] + 1e-12
        rows.append((u * duration, *x, *y))
        output_path = Path(__file__).resolve().parent / "trajectory.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "vx", "ax", "jx",
                         "y", "vy", "ay", "jy"])
        writer.writerows(rows)

    short = 0.8 * duration
    violations = []
    for i, delta in enumerate(DELTA):
        values = peaks(delta, short)
        for label, value, limit in zip(
                ("v", "a", "j"), values, (VMAX[i], AMAX[i], JMAX[i])):
            if value > limit + 1e-12:
                violations.append(f"axis {i} {label}={value:.6f}>{limit}")
    assert violations
    print(f"shortened duration = {short:.6f} s")
    print("violations: " + "; ".join(violations))
    print("Endpoint, analytic peak, sampled state and synchronization checks passed.")

#!/usr/bin/env python3
"""Report whether a cross-backend run actually used the device path.

Why this exists
    Two separate opt-ins gate the flagship device chain, and setting only one
    produces a mixed configuration that is both slow and easy to mislabel:

      ONEFLOW_ALLOW_VISCOUS_HIP_BATCH     -> UseHipBatchAdapter (fused flux)
      ONEFLOW_ALLOW_VISCOUS_HIP_GRADIENT  -> UseHipGradient / device recon

    Job 128983927 set only the first: the capability report said enabled=1 and
    HIP kernel time was non-zero, yet gradient_kernel stayed at 0 because the
    gradient ran on the host.  Reading a wall-clock number from that run
    produced a "24x slower" conclusion about the device path that described the
    host instead.

    The only authoritative statements are the binary's own capability report
    and a non-zero device gradient timer.  A threshold on rhs_inviscid is not
    one: the same per-step cost can come from a slow device kernel or from the
    host fallback, and HIP kernel only measures the launch call.

Usage
    parse-xb-device-path.py <hip-log> <hip-stage.tsv>

Exit status
    0  device flux and device gradient both active (timing is usable)
    2  either one is not active, or the evidence needed is missing
"""
import pathlib
import re
import sys


def read_timers(path):
    timers = {}
    try:
        text = pathlib.Path(path).read_text(errors="replace")
    except OSError:
        return timers
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        try:
            timers[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return timers


def batch_decision(log_text):
    """Return (enabled, reason) from the capability report, or (None, None)."""
    match = re.search(r"HIP_BATCH_DIAGNOSTIC enabled=(\d) reason=(\S+)", log_text)
    if match is None:
        return None, None
    return int(match.group(1)) == 1, match.group(2)


def main(argv):
    if len(argv) != 3:
        print("usage: parse-xb-device-path.py <hip-log> <hip-stage.tsv>",
              file=sys.stderr)
        return 2

    log_path, stage_path = argv[1], argv[2]
    try:
        log_text = pathlib.Path(log_path).read_text(errors="replace")
    except OSError:
        print("HIP_LOG_MISSING " + log_path)
        print("TIMING_USABLE=no")
        return 2
    enabled, reason = batch_decision(log_text)
    timers = read_timers(stage_path)

    gradient_kernel = timers.get("gradient_kernel", 0.0)
    recon_kernel = timers.get("recon_kernel", 0.0)

    print("batch_decision_enabled=" +
          ("unknown" if enabled is None else ("1" if enabled else "0")))
    print("batch_decision_reason=" + (reason or "unknown"))
    print("gradient_kernel=%.6g" % gradient_kernel)
    print("recon_kernel=%.6g" % recon_kernel)

    # A device gradient must have launched at least once; the timer accumulates
    # over the whole run, so any positive value proves execution.
    gradient_on = gradient_kernel > 0.0
    flux_on = enabled is True

    print("DEVICE_FLUX_ON=" + ("yes" if flux_on else "no"))
    print("DEVICE_GRADIENT_ON=" + ("yes" if gradient_on else "no"))
    if flux_on and gradient_on:
        print("TIMING_USABLE=yes")
        return 0

    if flux_on != gradient_on:
        print("TIMING_UNUSABLE=yes reason=mixed_device_host_paths")
    elif enabled is None:
        print("TIMING_UNUSABLE=yes reason=no_capability_report")
    else:
        print("TIMING_UNUSABLE=yes reason=host_fallback")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

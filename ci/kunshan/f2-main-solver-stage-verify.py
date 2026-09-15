#!/usr/bin/env python3
"""Compare legacy CPU, CPU batch, and HIP batch main-solver stage traces."""
import argparse
import math
import pathlib
import struct


MAGIC = bytes((79, 70, 83, 84, 71, 48, 49, 0))
HEADER = struct.Struct("<8sIIiiIIQ")
KIND_NAMES = {1: "face", 2: "inviscid_residual", 3: "state"}
ARRAY_NAMES = {
    1: ("qf1", "qf2", "invflux"),
    2: ("inviscid_residual",),
    3: ("state",),
}


def read_trace(path):
    path = pathlib.Path(path)
    data = path.read_bytes()
    records = []
    offset = 0
    while offset < len(data):
        if offset + HEADER.size > len(data):
            raise SystemExit("STAGE_TRACE_HEADER_FAIL path={}".format(path))
        header = HEADER.unpack_from(data, offset)
        offset += HEADER.size
        magic, kind, sequence, outer_step, grid_level, n_eq, n_arrays, n_items = header
        if magic != MAGIC or kind not in ARRAY_NAMES:
            raise SystemExit("STAGE_TRACE_HEADER_FAIL path={}".format(path))
        if n_arrays != len(ARRAY_NAMES[kind]) or n_eq == 0 or n_items == 0:
            raise SystemExit("STAGE_TRACE_SHAPE_FAIL path={}".format(path))
        count = n_eq * n_items
        arrays = []
        for _ in range(n_arrays):
            end = offset + count * 8
            if end > len(data):
                raise SystemExit("STAGE_TRACE_SIZE_FAIL path={}".format(path))
            arrays.append(struct.unpack_from("<{}d".format(count), data, offset))
            offset = end
        records.append({
            "key": (kind, sequence, outer_step, grid_level, n_eq, n_items),
            "kind": kind,
            "n_items": n_items,
            "arrays": arrays,
        })
    if not records:
        raise SystemExit("STAGE_TRACE_EMPTY path={}".format(path))
    return records


def compare_pair(reference_name, reference, candidate_name, candidate, abs_tol, rel_tol):
    reference_keys = [record["key"] for record in reference]
    candidate_keys = [record["key"] for record in candidate]
    if reference_keys != candidate_keys:
        raise SystemExit(
            "STAGE_TRACE_RECORD_FAIL reference={} candidate={}".format(
                reference_keys, candidate_keys))

    overall_absolute = 0.0
    overall_scaled = 0.0
    for ref_record, candidate_record in zip(reference, candidate):
        kind = ref_record["kind"]
        n_items = ref_record["n_items"]
        for array_name, left, right in zip(
                ARRAY_NAMES[kind], ref_record["arrays"], candidate_record["arrays"]):
            if not all(math.isfinite(value) for value in left + right):
                raise SystemExit(
                    "STAGE_TRACE_FINITE_FAIL candidate={} array={}".format(
                        candidate_name, array_name))
            absolute = max(abs(a - b) for a, b in zip(left, right))
            scaled = max(
                abs(a - b) / max(abs(a), abs(b), 1.0)
                for a, b in zip(left, right))
            if any(
                    abs(a - b) > abs_tol + rel_tol * max(abs(a), abs(b), 1.0)
                    for a, b in zip(left, right)):
                raise SystemExit(
                    "STAGE_TRACE_TOLERANCE_FAIL reference={} candidate={} "
                    "kind={} array={} max_absolute={:.17g} max_scaled={:.17g}".format(
                        reference_name, candidate_name, KIND_NAMES[kind], array_name,
                        absolute, scaled))
            overall_absolute = max(overall_absolute, absolute)
            overall_scaled = max(overall_scaled, scaled)
            print(
                "STAGE_TRACE candidate={} kind={} sequence={} array={} n={} "
                "max_absolute={:.17g} max_scaled={:.17g}".format(
                    candidate_name, KIND_NAMES[kind], ref_record["key"][1],
                    array_name, len(left), absolute, scaled))

            if array_name in ("qf1", "qf2", "state"):
                density = right[:n_items]
                pressure = right[4 * n_items:5 * n_items]
                if min(density) <= 0.0 or min(pressure) <= 0.0:
                    raise SystemExit(
                        "STAGE_TRACE_PHYSICAL_FAIL candidate={} array={}".format(
                            candidate_name, array_name))
                print(
                    "STAGE_PHYSICAL candidate={} array={} finite=true "
                    "min_density={:.17g} min_pressure={:.17g}".format(
                        candidate_name, array_name, min(density), min(pressure)))

    print(
        "STAGE_TRACE_PAIR_PASS reference={} candidate={} records={} "
        "max_absolute={:.17g} max_scaled={:.17g}".format(
            reference_name, candidate_name, len(reference),
            overall_absolute, overall_scaled))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_trace")
    parser.add_argument("cpu_batch_trace")
    parser.add_argument("hip_batch_trace")
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-11)
    args = parser.parse_args()

    legacy = read_trace(args.legacy_trace)
    cpu_batch = read_trace(args.cpu_batch_trace)
    hip_batch = read_trace(args.hip_batch_trace)
    compare_pair(
        "legacy", legacy, "cpu_batch", cpu_batch,
        args.absolute_tolerance, args.relative_tolerance)
    compare_pair(
        "legacy", legacy, "hip_batch", hip_batch,
        args.absolute_tolerance, args.relative_tolerance)
    print("STAGE_TRACE_PASS records={}".format(len(legacy)))


if __name__ == "__main__":
    main()

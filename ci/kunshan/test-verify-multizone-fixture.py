#!/usr/bin/env python3
"""Tests for the external multi-zone fixture metadata gate."""

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("verify-multizone-fixture.py")
SPEC = importlib.util.spec_from_file_location("verify_multizone_fixture", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def manifest():
    return {
        "schema": "oneflow.multizone.fixture/v1",
        "ranks": 2,
        "zones": [
            {"zone": 0, "rank": 0, "owned_cells": 10, "ghost_cells": 2},
            {"zone": 1, "rank": 1, "owned_cells": 12, "ghost_cells": 2},
        ],
        "interfaces": [
            {
                "source": {
                    "rank": 0,
                    "zone": 0,
                    "face": 3,
                    "cell": 4,
                    "role": "OwnedCell",
                },
                "target": {
                    "rank": 1,
                    "zone": 1,
                    "face": 5,
                    "cell": 12,
                    "role": "GhostCell",
                },
                "equations": 5,
                "send_values": 5,
                "receive_values": 5,
                "memory_space": "HostStaged",
            },
            {
                "source": {
                    "rank": 1,
                    "zone": 1,
                    "face": 5,
                    "cell": 6,
                    "role": "OwnedCell",
                },
                "target": {
                    "rank": 0,
                    "zone": 0,
                    "face": 3,
                    "cell": 10,
                    "role": "GhostCell",
                },
                "equations": 5,
                "send_values": 5,
                "receive_values": 5,
                "memory_space": "HostStaged",
            },
        ],
    }


class MultiZoneFixtureVerifierTest(unittest.TestCase):
    def test_valid_manifest(self):
        summary = MODULE.validate_manifest(manifest(), 2)
        self.assertEqual(summary, {"ranks": 2, "zones": 2, "interfaces": 2})

    def test_rejects_self_neighbor(self):
        value = manifest()
        value["interfaces"][0]["target"]["rank"] = 0
        with self.assertRaisesRegex(ValueError, "cannot exchange with itself"):
            MODULE.validate_manifest(value, 2)

    def test_rejects_wrong_endpoint_role(self):
        value = manifest()
        value["interfaces"][0]["source"]["role"] = "GhostCell"
        with self.assertRaisesRegex(ValueError, "OwnedCell source"):
            MODULE.validate_manifest(value, 2)

    def test_allows_zero_length_message(self):
        value = manifest()
        for interface in value["interfaces"]:
            interface["send_values"] = 0
            interface["receive_values"] = 0
        summary = MODULE.validate_manifest(value, 2)
        self.assertEqual(summary["interfaces"], 2)

    def test_rejects_target_cell_outside_zone(self):
        value = manifest()
        value["interfaces"][0]["target"]["cell"] = 14
        with self.assertRaisesRegex(ValueError, "outside target zone"):
            MODULE.validate_manifest(value, 2)

    def test_cli_reports_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "fixture.json"
            path.write_text(json.dumps(manifest()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--metadata",
                    str(path),
                    "--expected-ranks",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MULTIZONE_FIXTURE_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()

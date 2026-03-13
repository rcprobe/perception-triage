#!/usr/bin/env python3
"""Example driver that builds the failures DB from JSON inputs."""

from triage.cli import main  # reuse CLI entrypoint


if __name__ == "__main__":
    # Delegate to the CLI main for consistent behavior.
    main()

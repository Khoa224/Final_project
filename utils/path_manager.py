"""
Path Manager - Handles all directory and file path configurations
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PathManager:
    """Manages all paths and directories for the VLSI flow"""

    def __init__(self, benchmark_name, base_dir=None):
        """
        Initialize path manager

        Args:
            benchmark_name: Name of the benchmark (e.g., 'adaptec1')
            base_dir: Base directory path (auto-detected if None)
        """
        self.benchmark = benchmark_name
        self.base_dir = self._resolve_base_dir(base_dir)

        # Set up all directory paths
        self.dreamplace_dir = self.base_dir / "DREAMPlace" / "install"
        self.router_dir = self.base_dir / "nthuRouter3"
        self.result_dir = self.base_dir / "test_result"
        self.benchmark_dir = self.dreamplace_dir / "benchmarks" / "ispd2005" / benchmark_name

        # Ensure result directory exists
        self.result_dir.mkdir(exist_ok=True)

        self._log_paths()

    def _resolve_base_dir(self, base_dir):
        """Resolve the base directory path"""
        if base_dir is None:
            # Check if we're in a container
            if os.path.exists('/kaggle') or os.path.exists('/.dockerenv'):
                base_dir = Path.cwd()
            else:
                # Use __file__ if available, otherwise current directory
                try:
                    base_dir = Path(__file__).parent.parent.resolve()
                except (NameError, TypeError):
                    base_dir = Path.cwd()
        else:
            # Ensure base_dir is a Path object
            base_dir = Path(base_dir) if not isinstance(base_dir, Path) else base_dir
            base_dir = base_dir.resolve()

        # Validate that base_dir is not None
        if base_dir is None:
            raise ValueError("base_dir cannot be None. Please provide a valid directory path.")

        return base_dir

    def _log_paths(self):
        """Log all configured paths"""
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"DREAMPlace directory: {self.dreamplace_dir}")
        logger.info(f"Router directory: {self.router_dir}")
        logger.info(f"Result directory: {self.result_dir}")
        logger.info(f"Benchmark directory: {self.benchmark_dir}")

    def get_benchmark_files(self):
        """Get paths to all benchmark input files"""
        return {
            'nodes': self.benchmark_dir / f"{self.benchmark}.nodes",
            'nets': self.benchmark_dir / f"{self.benchmark}.nets",
            'scl': self.benchmark_dir / f"{self.benchmark}.scl",
            'aux': self.benchmark_dir / f"{self.benchmark}.aux"
        }

    def get_placement_output_paths(self):
        """Get possible locations for placement output file"""
        placement_filename = f"{self.benchmark}.gp.pl"
        return [
            self.dreamplace_dir / "results" / self.benchmark / placement_filename,
            self.base_dir / "DREAMPlace" / "results" / self.benchmark / placement_filename,
            Path("DREAMPlace/install/results") / self.benchmark / placement_filename,
            Path("DREAMPlace/results") / self.benchmark / placement_filename,
        ]

    def get_placer_script_paths(self):
        """Get possible locations for Placer.py script"""
        return [
            self.dreamplace_dir / "dreamplace" / "Placer.py",
            self.base_dir / "DREAMPlace" / "dreamplace" / "Placer.py",
            Path("DREAMPlace/install/dreamplace/Placer.py"),
            Path("DREAMPlace/dreamplace/Placer.py"),
        ]

    def get_routing_input_file(self):
        """Get path to routing input file"""
        return self.result_dir / f"{self.benchmark}_routing_input.gr"

    def get_routing_output_file(self):
        """Get path to routing output file"""
        return self.result_dir / f"{self.benchmark}_routing_output.txt"

    def get_router_executable(self):
        """Get path to router executable"""
        return self.router_dir / "NthuRoute"


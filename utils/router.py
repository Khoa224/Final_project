"""
Router - Handles nthuRouter3 global routing operations
"""

import subprocess
import shutil
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Router:
    """Handles nthuRouter3 global routing execution"""

    def __init__(self, path_manager):
        """
        Initialize router

        Args:
            path_manager: PathManager instance for directory/file paths
        """
        self.path_manager = path_manager
        self.benchmark = path_manager.benchmark

    def run(self, max_iteration_p2=150, init_box_size_p2=25, box_expand_size_p2=1,
            overflow_threshold=0, max_iteration_p3=20, init_box_size_p3=10,
            box_expand_size_p3=15, monotonic_routing=0):
        """
        Run nthuRouter3 global routing

        Args:
            max_iteration_p2: Phase 2 max iterations
            init_box_size_p2: Phase 2 initial box size
            box_expand_size_p2: Phase 2 box expand size
            overflow_threshold: Overflow threshold
            max_iteration_p3: Phase 3 max iterations
            init_box_size_p3: Phase 3 initial box size
            box_expand_size_p3: Phase 3 box expand size
            monotonic_routing: Enable monotonic routing (0 or 1)

        Returns:
            bool: True if routing successful, False otherwise
        """
        logger.info("="*60)
        logger.info("STEP 3: Running nthuRouter3 global routing")
        logger.info("="*60)

        # Validate input file exists
        input_file = self.path_manager.get_routing_input_file()
        if not input_file.exists():
            logger.error(f"Routing input not found: {input_file}")
            return False

        # Validate router executable exists
        router_exe = self.path_manager.get_router_executable()
        if not router_exe.exists():
            logger.error(f"Router executable not found: {router_exe}")
            logger.error("Please build nthuRouter3 first: cd nthuRouter3 && make")
            return False

        # Build command
        output_file = self.path_manager.router_dir / "output"
        cmd = self._build_command(router_exe, input_file, output_file,
                                  max_iteration_p2, init_box_size_p2, box_expand_size_p2,
                                  overflow_threshold, max_iteration_p3, init_box_size_p3,
                                  box_expand_size_p3, monotonic_routing)

        try:
            start_time = time.time()
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info(f"Working directory: {self.path_manager.router_dir}")
            logger.info("Router is running... (this may take several minutes)")

            # Run router
            result = subprocess.run(
                cmd,
                cwd=str(self.path_manager.router_dir),
                check=True,
                capture_output=False,
                text=True
            )

            elapsed = time.time() - start_time
            logger.info(f"Routing completed in {elapsed:.2f} seconds")

            # Process output
            if output_file.exists():
                self._process_output(output_file)
                return True
            else:
                logger.error(f"Routing output not created: {output_file}")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Routing failed with return code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during routing: {e}")
            logger.exception("Full traceback:")
            return False

    def _build_command(self, router_exe, input_file, output_file,
                      max_iteration_p2, init_box_size_p2, box_expand_size_p2,
                      overflow_threshold, max_iteration_p3, init_box_size_p3,
                      box_expand_size_p3, monotonic_routing):
        """Build the router command with all parameters"""
        return [
            str(router_exe),
            f"--input={str(input_file.resolve())}",
            f"--output={str(output_file)}",
            f"--p2-max-iteration={max_iteration_p2}",
            f"--p2-init-box-size={init_box_size_p2}",
            f"--p2-box-expand-size={box_expand_size_p2}",
            f"--overflow-threshold={overflow_threshold}",
            f"--p3-max-iteration={max_iteration_p3}",
            f"--p3-init-box-size={init_box_size_p3}",
            f"--p3-box-expand-size={box_expand_size_p3}",
            f"--monotonic-routing={monotonic_routing}"
        ]

    def _process_output(self, output_file):
        """Process and copy routing output"""
        logger.info(f"Routing result: {output_file}")

        # Copy to result directory
        dest = self.path_manager.get_routing_output_file()
        shutil.copy(output_file, dest)
        logger.info(f"Copied to: {dest}")

        # Show summary statistics
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
                logger.info("Routing summary:")
                for line in lines[-20:]:  # Show last 20 lines
                    logger.info(f"  {line.rstrip()}")
        except Exception as e:
            logger.warning(f"Could not read output file: {e}")


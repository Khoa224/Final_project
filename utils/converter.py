"""
Converter - Handles placement to routing format conversion
"""

import time
import logging
from pathlib import Path
from Placement_to_routing_converter import RoutingBenchmarkGenerator

logger = logging.getLogger(__name__)


class Converter:
    """Handles conversion from placement to routing format"""

    def __init__(self, path_manager):
        """
        Initialize converter

        Args:
            path_manager: PathManager instance for directory/file paths
        """
        self.path_manager = path_manager
        self.benchmark = path_manager.benchmark
        self.generator = RoutingBenchmarkGenerator()

    def run(self, tile_size=35, adjustment_factor=50, safe_guard_factor=90, mode=2):
        """
        Convert placement to routing format

        Args:
            tile_size: Routing tile size
            adjustment_factor: Adjustment factor percentage
            safe_guard_factor: Safe guard factor percentage
            mode: Conversion mode

        Returns:
            bool: True if conversion successful, False otherwise
        """
        logger.info("="*60)
        logger.info("STEP 2: Converting placement to routing format")
        logger.info("="*60)

        # Get file paths
        file_paths = self._get_file_paths()
        if not file_paths:
            return False

        self._log_file_paths(file_paths)

        try:
            start_time = time.time()

            # Phase 0: SCL processing
            if not self.generator.process_scl_file(str(file_paths['scl'])):
                logger.error("SCL file processing failed")
                return False

            # Phase 1: Node processing
            if not self.generator.process_node_file(str(file_paths['nodes'])):
                logger.error("Node file processing failed")
                return False

            # Phase 2: Solution processing
            if not self.generator.process_solution_file(str(file_paths['solution'])):
                logger.error("Solution file processing failed")
                return False

            # Phase 3: Net processing
            pin_bounds = self.generator.process_net_file(str(file_paths['nets']))
            if not pin_bounds:
                logger.error("Net file processing failed")
                return False

            # Phase 4: Benchmark generation
            logger.info(f"Generating routing benchmark (tile={tile_size}, adj={adjustment_factor}%, guard={safe_guard_factor}%, mode={mode})")
            if not self.generator.generate_benchmark(
                str(file_paths['output']), tile_size, adjustment_factor,
                safe_guard_factor, mode, pin_bounds
            ):
                logger.error("Benchmark generation failed")
                return False

            elapsed = time.time() - start_time
            logger.info(f"Conversion completed in {elapsed:.2f} seconds")
            logger.info(f"Routing benchmark: {file_paths['output']}")
            return True

        except Exception as e:
            logger.error(f"Conversion error: {e}")
            logger.exception("Traceback:")
            return False

    def _get_file_paths(self):
        """
        Get all required file paths for conversion

        Returns:
            dict: Dictionary of file paths or None if placement file not found
        """
        benchmark_files = self.path_manager.get_benchmark_files()

        # Find placement solution file
        possible_solution_files = self.path_manager.get_placement_output_paths()
        solution_file = None
        for path in possible_solution_files:
            if path.exists():
                solution_file = path
                break

        if not solution_file:
            logger.error("Placement file not found in any location:")
            for path in possible_solution_files:
                logger.error(f"  - {path}")
            return None

        return {
            'nodes': benchmark_files['nodes'],
            'solution': solution_file,
            'nets': benchmark_files['nets'],
            'scl': benchmark_files['scl'],
            'output': self.path_manager.get_routing_input_file()
        }

    def _log_file_paths(self, file_paths):
        """Log all file paths being used"""
        logger.info(f"Input files:")
        logger.info(f"  Nodes: {file_paths['nodes']}")
        logger.info(f"  Solution: {file_paths['solution']}")
        logger.info(f"  Nets: {file_paths['nets']}")
        logger.info(f"  SCL: {file_paths['scl']}")
        logger.info(f"Output: {file_paths['output']}")


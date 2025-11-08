"""
Complete VLSI Placement-to-Routing Flow
Integrates DREAMPlace, Converter, and nthuRouter3
Docker/Linux Compatible Version - Modular Architecture
"""

import sys
import logging
import time

# Import modular components
from utils import PathManager, FileValidator, Placer, Converter, Router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLSIFlow:
    """Complete VLSI placement to routing flow - Orchestrator"""

    def __init__(self, benchmark_name, base_dir=None):
        """
        Initialize VLSI flow

        Args:
            benchmark_name: Name of the benchmark (e.g., 'adaptec1')
            base_dir: Base directory path (auto-detected if None)
        """
        # Initialize path manager
        self.path_manager = PathManager(benchmark_name, base_dir)
        self.benchmark = benchmark_name

        # Initialize components
        self.placer = Placer(self.path_manager)
        self.converter = Converter(self.path_manager)
        self.router = Router(self.path_manager)
        self.validator = FileValidator()

    def check_files_exist(self):
        """Verify all required input files exist"""
        benchmark_files = self.path_manager.get_benchmark_files()
        return self.validator.validate_benchmark_files(benchmark_files)

    def run_placement(self, use_gpu=1, iterations=2000, target_density=0.9):
        """
        Run DREAMPlace placement

        Args:
            use_gpu: Use GPU (1) or CPU (0)
            iterations: Number of placement iterations
            target_density: Target placement density

        Returns:
            bool: True if successful, False otherwise
        """
        return self.placer.run(use_gpu, iterations, target_density)

    def run_converter(self, tile_size=35, adjustment_factor=50, safe_guard_factor=90, mode=2):
        """
        Convert placement to routing format

        Args:
            tile_size: Routing tile size
            adjustment_factor: Adjustment factor percentage
            safe_guard_factor: Safe guard factor percentage
            mode: Conversion mode

        Returns:
            bool: True if successful, False otherwise
        """
        return self.converter.run(tile_size, adjustment_factor, safe_guard_factor, mode)

    def run_routing(self, max_iteration_p2=150, init_box_size_p2=25, box_expand_size_p2=1,
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
            bool: True if successful, False otherwise
        """
        return self.router.run(
            max_iteration_p2, init_box_size_p2, box_expand_size_p2,
            overflow_threshold, max_iteration_p3, init_box_size_p3,
            box_expand_size_p3, monotonic_routing
        )

    def run_complete_flow(self, use_gpu=1, iterations=2000, target_density=0.9,
                         tile_size=35, adjustment_factor=50, safe_guard_factor=90):
        """
        Execute complete placement to routing flow

        Args:
            use_gpu: Use GPU (1) or CPU (0)
            iterations: Placement iterations
            target_density: Target placement density
            tile_size: Routing tile size
            adjustment_factor: Adjustment factor percentage
            safe_guard_factor: Safe guard factor percentage

        Returns:
            bool: True if entire flow successful, False otherwise
        """
        logger.info("#"*60)
        logger.info(f"# Complete VLSI Flow: {self.benchmark}")
        logger.info("#"*60)

        total_start = time.time()

        # Check prerequisites
        if not self.check_files_exist():
            logger.error("Prerequisites check failed")
            return False

        # Step 1: Placement
        if not self.run_placement(use_gpu, iterations, target_density):
            logger.error("Flow terminated: Placement failed")
            return False

        # Step 2: Conversion
        if not self.run_converter(tile_size, adjustment_factor, safe_guard_factor):
            logger.error("Flow terminated: Conversion failed")
            return False

        # Step 3: Routing
        if not self.run_routing():
            logger.error("Flow terminated: Routing failed")
            return False

        total_elapsed = time.time() - total_start

        logger.info("#"*60)
        logger.info(f"# COMPLETE FLOW FINISHED SUCCESSFULLY")
        logger.info(f"# Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        logger.info("#"*60)
        logger.info(f"Results directory: {self.path_manager.result_dir}")

        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Complete VLSI Placement-to-Routing Flow')
    parser.add_argument('benchmark', help='Benchmark name (e.g., adaptec1)')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU (1) or CPU (0)')
    parser.add_argument('--iterations', type=int, default=2000, help='Placement iterations')
    parser.add_argument('--density', type=float, default=0.9, help='Target density')
    parser.add_argument('--tile-size', type=int, default=35, help='Routing tile size')
    parser.add_argument('--adj-factor', type=int, default=50, help='Adjustment factor %%')
    parser.add_argument('--safe-guard', type=int, default=90, help='Safe guard factor %%')

    args = parser.parse_args()

    flow = VLSIFlow(args.benchmark)
    success = flow.run_complete_flow(
        use_gpu=args.gpu,
        iterations=args.iterations,
        target_density=args.density,
        tile_size=args.tile_size,
        adjustment_factor=args.adj_factor,
        safe_guard_factor=args.safe_guard
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

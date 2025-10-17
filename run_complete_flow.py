"""
Complete VLSI Placement-to-Routing Flow
Integrates DREAMPlace, Converter, and nthuRouter3
Docker/Linux Compatible Version
"""

import subprocess
import os
import sys
import logging
import time
from pathlib import Path

# Import the converter
from Placement_to_routing_converter import RoutingBenchmarkGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLSIFlow:
    """Complete VLSI placement to routing flow"""

    def __init__(self, benchmark_name, base_dir=None):
        # Use current directory if in Docker/Kaggle environment
        if base_dir is None:
            # Check if we're in a container
            if os.path.exists('/kaggle') or os.path.exists('/.dockerenv'):
                base_dir = Path.cwd()
            else:
                base_dir = Path(__file__).parent
        else:
            base_dir = Path(base_dir)

        self.benchmark = benchmark_name
        self.base_dir = base_dir
        self.dreamplace_dir = base_dir / "DREAMPlace" / "install"
        self.router_dir = base_dir / "nthuRouter3"
        self.result_dir = base_dir / "test_result"
        self.benchmark_dir = self.dreamplace_dir / "benchmarks" / "ispd2005" / benchmark_name

        # Ensure result directory exists
        self.result_dir.mkdir(exist_ok=True)

        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"DREAMPlace directory: {self.dreamplace_dir}")

    def check_files_exist(self):
        """Verify all required input files exist"""
        logger.info("Checking input files...")

        required_files = [
            self.benchmark_dir / f"{self.benchmark}.nodes",
            self.benchmark_dir / f"{self.benchmark}.nets",
            self.benchmark_dir / f"{self.benchmark}.scl",
            self.benchmark_dir / f"{self.benchmark}.aux"
        ]

        missing = []
        for f in required_files:
            if not f.exists():
                missing.append(str(f))

        if missing:
            logger.error("Missing required files:")
            for f in missing:
                logger.error(f"  - {f}")
            return False

        logger.info("All required files found")
        return True

    def run_placement(self, use_gpu=1, iterations=2000, target_density=0.9):
        """Run DREAMPlace placement"""
        logger.info("="*60)
        logger.info(f"STEP 1: Running DREAMPlace for {self.benchmark}")
        logger.info("="*60)

        # Try multiple possible locations for Placer.py
        possible_placer_paths = [
            self.dreamplace_dir / "dreamplace" / "Placer.py",
            self.base_dir / "DREAMPlace" / "dreamplace" / "Placer.py",
            Path("DREAMPlace/install/dreamplace/Placer.py"),
            Path("DREAMPlace/dreamplace/Placer.py"),
        ]

        placer_script = None
        working_dir = None

        logger.info("Searching for Placer.py...")
        for path in possible_placer_paths:
            logger.info(f"  Checking: {path}")
            if path.exists():
                placer_script = path
                # Determine working directory
                if "install/dreamplace" in str(path):
                    working_dir = path.parent.parent  # Go up to install/
                elif "DREAMPlace/dreamplace" in str(path):
                    working_dir = path.parent.parent  # Go up to DREAMPlace/
                logger.info(f"  ✓ Found at: {placer_script}")
                break

        if not placer_script:
            logger.error("Placer.py not found in any expected location!")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Directory contents: {list(Path.cwd().iterdir())[:10]}")
            return False

        logger.info(f"Using placer: {placer_script}")
        logger.info(f"Working directory: {working_dir}")

        # Create JSON config dynamically
        config = self._create_placement_config(use_gpu, iterations, target_density)
        config_file = working_dir / f"temp_{self.benchmark}.json"

        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration: GPU={use_gpu}, iterations={iterations}, density={target_density}")

        cmd = [
            sys.executable,
            str(placer_script),
            str(config_file)
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            elapsed = time.time() - start_time

            logger.info(f"DREAMPlace completed in {elapsed:.2f} seconds")

            # Check for output file in multiple locations
            output_locations = [
                working_dir / "results" / self.benchmark / f"{self.benchmark}.gp.pl",
                self.dreamplace_dir / "results" / self.benchmark / f"{self.benchmark}.gp.pl",
                self.base_dir / "DREAMPlace" / "results" / self.benchmark / f"{self.benchmark}.gp.pl",
            ]

            for output_file in output_locations:
                if output_file.exists():
                    logger.info(f"Placement result: {output_file}")
                    return True

            logger.error("Placement output file not found!")
            for loc in output_locations:
                logger.error(f"  Checked: {loc}")
            return False

        except subprocess.CalledProcessError as e:
            logger.error(f"DREAMPlace failed with return code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Error running placement: {e}")
            logger.exception("Full traceback:")
            return False
        finally:
            # Clean up temp config
            if config_file.exists():
                config_file.unlink()

    def _create_placement_config(self, use_gpu, iterations, target_density):
        """Create DREAMPlace configuration"""
        return {
            "aux_input": f"benchmarks/ispd2005/{self.benchmark}/{self.benchmark}.aux",
            "gpu": use_gpu,
            "num_threads": 8,
            "num_bins_x": 512,
            "num_bins_y": 512,
            "global_place_stages": [
                {
                    "num_bins_x": 512,
                    "num_bins_y": 512,
                    "iteration": iterations,
                    "learning_rate": 0.01,
                    "wirelength": "weighted_average",
                    "density_weight": 8e-5,
                    "optimizer": "nesterov"
                }
            ],
            "target_density": target_density,
            "density_weight": 8e-5,
            "gamma": 4.0,
            "random_seed": 1000,
            "scale_factor": 1.0,
            "ignore_net_degree": 100,
            "enable_fillers": 1,
            "gp_noise_ratio": 0.025,
            "global_place_flag": 1,
            "legalize_flag": 1,
            "detailed_place_flag": 0,
            "stop_overflow": 0.1,
            "dtype": "float32",
            "plot_flag": 0,
            "result_dir": "results"
        }

    def run_converter(self, tile_size=35, adjustment_factor=50, safe_guard_factor=90, mode=2):
        """Convert placement to routing format"""
        logger.info("="*60)
        logger.info("STEP 2: Converting placement to routing format")
        logger.info("="*60)

        # File paths - check multiple possible locations
        nodes_file = self.benchmark_dir / f"{self.benchmark}.nodes"

        # Try multiple locations for placement output
        possible_solution_files = [
            self.dreamplace_dir / "results" / self.benchmark / f"{self.benchmark}.gp.pl",
            self.base_dir / "DREAMPlace" / "results" / self.benchmark / f"{self.benchmark}.gp.pl",
            Path("DREAMPlace/install/results") / self.benchmark / f"{self.benchmark}.gp.pl",
            Path("DREAMPlace/results") / self.benchmark / f"{self.benchmark}.gp.pl",
        ]

        solution_file = None
        for path in possible_solution_files:
            if path.exists():
                solution_file = path
                break

        if not solution_file:
            logger.error("Placement file not found in any location:")
            for path in possible_solution_files:
                logger.error(f"  - {path}")
            return False

        net_file = self.benchmark_dir / f"{self.benchmark}.nets"
        scl_file = self.benchmark_dir / f"{self.benchmark}.scl"
        output_file = self.result_dir / f"{self.benchmark}_routing_input.gr"

        logger.info(f"Input files:")
        logger.info(f"  Nodes: {nodes_file}")
        logger.info(f"  Solution: {solution_file}")
        logger.info(f"  Nets: {net_file}")
        logger.info(f"  SCL: {scl_file}")
        logger.info(f"Output: {output_file}")

        generator = RoutingBenchmarkGenerator()

        try:
            start_time = time.time()

            # Phase 0: SCL processing
            if not generator.process_scl_file(str(scl_file)):
                logger.error("SCL file processing failed")
                return False

            # Phase 1: Node processing
            if not generator.process_node_file(str(nodes_file)):
                logger.error("Node file processing failed")
                return False

            # Phase 2: Solution processing
            if not generator.process_solution_file(str(solution_file)):
                logger.error("Solution file processing failed")
                return False

            # Phase 3: Net processing
            pin_bounds = generator.process_net_file(str(net_file))
            if not pin_bounds:
                logger.error("Net file processing failed")
                return False

            # Phase 4: Benchmark generation
            logger.info(f"Generating routing benchmark (tile={tile_size}, adj={adjustment_factor}%, guard={safe_guard_factor}%, mode={mode})")
            if not generator.generate_benchmark(
                str(output_file), tile_size, adjustment_factor,
                safe_guard_factor, mode, pin_bounds
            ):
                logger.error("Benchmark generation failed")
                return False

            elapsed = time.time() - start_time
            logger.info(f"Conversion completed in {elapsed:.2f} seconds")
            logger.info(f"Routing benchmark: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Conversion error: {e}")
            logger.exception("Traceback:")
            return False

    def run_routing(self):
        """Run nthuRouter3 global routing"""
        logger.info("="*60)
        logger.info("STEP 3: Running nthuRouter3 global routing")
        logger.info("="*60)

        input_file = self.result_dir / f"{self.benchmark}_routing_input.gr"

        if not input_file.exists():
            logger.error(f"Routing input not found: {input_file}")
            return False

        # Check if router executable exists
        router_exe = self.router_dir / "NthuRoute"
        if not router_exe.exists():
            logger.error(f"Router executable not found: {router_exe}")
            logger.error("Please build nthuRouter3 first: cd nthuRouter3 && make")
            return False

        # Build command with proper flags
        output_file = self.router_dir / "output"

        cmd = [
            str(router_exe),
            f"--input={str(input_file.resolve())}",
            f"--output={str(output_file)}",
            "--p2-max-iteration=150",
            "--p2-init-box-size=25",
            "--p2-box-expand-size=1",
            "--overflow-threshold=0",
            "--p3-max-iteration=20",
            "--p3-init-box-size=10",
            "--p3-box-expand-size=15",
            "--monotonic-routing=0"
        ]

        try:
            start_time = time.time()
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info(f"Working directory: {self.router_dir}")
            logger.info("Router is running... (this may take several minutes)")

            # Run without capturing output so we can see real-time progress
            result = subprocess.run(
                cmd,
                cwd=str(self.router_dir),
                check=True,
                capture_output=False,  # Changed to False to show output in real-time
                text=True
            )

            elapsed = time.time() - start_time
            logger.info(f"Routing completed in {elapsed:.2f} seconds")

            # Check for output file
            if output_file.exists():
                logger.info(f"Routing result: {output_file}")
                # Copy to result directory
                import shutil
                dest = self.result_dir / f"{self.benchmark}_routing_output.txt"
                shutil.copy(output_file, dest)
                logger.info(f"Copied to: {dest}")

                # Show summary statistics from output file
                try:
                    with open(output_file, 'r') as f:
                        lines = f.readlines()
                        logger.info("Routing summary:")
                        for line in lines[-20:]:  # Show last 20 lines
                            logger.info(f"  {line.rstrip()}")
                except Exception as e:
                    logger.warning(f"Could not read output file: {e}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Routing failed with return code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during routing: {e}")
            logger.exception("Full traceback:")
            return False

    def run_complete_flow(self, use_gpu=1, iterations=2000, target_density=0.9,
                         tile_size=35, adjustment_factor=50, safe_guard_factor=90):
        """Execute complete placement to routing flow"""
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
        logger.info(f"Results directory: {self.result_dir}")

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

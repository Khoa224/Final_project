"""
Placer - Handles DREAMPlace placement operations
"""

import subprocess
import sys
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Placer:
    """Handles DREAMPlace placement execution"""

    def __init__(self, path_manager):
        """
        Initialize placer

        Args:
            path_manager: PathManager instance for directory/file paths
        """
        self.path_manager = path_manager
        self.benchmark = path_manager.benchmark

    def run(self, use_gpu=1, iterations=2000, target_density=0.9):
        """
        Run DREAMPlace placement

        Args:
            use_gpu: Use GPU (1) or CPU (0)
            iterations: Number of placement iterations
            target_density: Target placement density

        Returns:
            bool: True if placement successful, False otherwise
        """
        logger.info("="*60)
        logger.info(f"STEP 1: Running DREAMPlace for {self.benchmark}")
        logger.info("="*60)

        # Find Placer.py script
        placer_script, working_dir = self._find_placer_script()
        if not placer_script:
            return False

        # Create configuration file
        config_file = self._create_config_file(working_dir, use_gpu, iterations, target_density)

        logger.info(f"Configuration: GPU={use_gpu}, iterations={iterations}, density={target_density}")

        # Build and execute command
        cmd = [sys.executable, str(placer_script), str(config_file)]
        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                check=True,
                capture_output=False,
                text=True
            )
            elapsed = time.time() - start_time

            logger.info(f"DREAMPlace completed in {elapsed:.2f} seconds")

            # Verify output file was created
            if self._verify_output_exists():
                return True
            else:
                logger.error("Placement output file not found!")
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

    def _find_placer_script(self):
        """
        Find Placer.py script and determine working directory

        Returns:
            tuple: (placer_script_path, working_directory) or (None, None)
        """
        possible_paths = self.path_manager.get_placer_script_paths()

        logger.info("Searching for Placer.py...")
        for path in possible_paths:
            logger.info(f"  Checking: {path}")
            if path.exists():
                placer_script = path.resolve()
                working_dir = self._determine_working_dir(placer_script)

                logger.info(f"  ✓ Found at: {placer_script}")
                logger.info(f"  ✓ Working directory: {working_dir}")
                return placer_script, working_dir

        logger.error("Placer.py not found in any expected location!")
        return None, None

    def _determine_working_dir(self, placer_script):
        """Determine working directory from placer script path"""
        path_str = str(placer_script).replace('\\', '/')

        if "install/dreamplace" in path_str:
            return placer_script.parent.parent  # Go up to install/
        elif "DREAMPlace/dreamplace" in path_str:
            return placer_script.parent.parent  # Go up to DREAMPlace/
        else:
            # Fallback: just go up two levels from Placer.py
            return placer_script.parent.parent

    def _create_config_file(self, working_dir, use_gpu, iterations, target_density):
        """
        Create DREAMPlace configuration file

        Returns:
            Path: Path to created config file
        """
        config = {
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

        config_file = working_dir / f"temp_{self.benchmark}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return config_file

    def _verify_output_exists(self):
        """Check if placement output file was created"""
        output_locations = self.path_manager.get_placement_output_paths()

        for output_file in output_locations:
            if output_file.exists():
                logger.info(f"Placement result: {output_file}")
                return True

        for loc in output_locations:
            logger.error(f"  Checked: {loc}")
        return False


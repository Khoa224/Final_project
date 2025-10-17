# Project Components and Individual Execution Guide

This project consists of two main components: **DREAMPlace** (VLSI placement tool) and **nthuRouter3** (routing tool).

---

## 🏗️ Project Structure Overview

```
FinalProject/
├── DREAMPlace_local/     # GPU-accelerated VLSI placement toolkit
├── nthuRouter3/          # NTHU Router for global routing
└── pyproject.toml        # Python project dependencies
```

---

## 📦 Component 1: DREAMPlace (VLSI Placement Tool)

**Description**: Deep learning toolkit-enabled GPU-accelerated VLSI placement tool that performs global placement, legalization, and detailed placement.

### Prerequisites

**Python Dependencies** (from `requirements.txt`):
- Python 3.5-3.9
- PyTorch >= 1.6.0
- NumPy >= 1.15.4
- SciPy >= 1.1.0
- matplotlib >= 2.2.2
- cairocffi >= 0.9.0
- shapely >= 1.7.0
- torch_optimizer == 0.3.0
- ncg_optimizer == 0.2.2
- pyunpack >= 0.1.2
- patool >= 1.12

**System Dependencies**:
- CMake
- GCC 7.5+ (with C++17 support)
- Boost >= 1.55.0
- Bison >= 3.3
- CUDA 9.1+ (optional, for GPU acceleration)

### Installation

1. **Install Python dependencies**:
```cmd
cd D:\work\IDE\PyCharm\FinalProject
pip install -r DREAMPlace_local\requirements.txt
```

2. **Build DREAMPlace**:
```cmd
cd DREAMPlace_local
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DPython_EXECUTABLE=python
cmake --build .
cmake --install .
```

### Sub-components and Individual Execution

#### 1.1 Download Benchmarks

**ISPD 2005 & 2015 Benchmarks**:
```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local
python benchmarks\ispd2005_2015.py
```

**ISPD 2019 Benchmarks**:
```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local
python benchmarks\ispd2019.py
```

#### 1.2 Run Full Placement Flow

Execute the complete placement pipeline (global placement → legalization → detailed placement):

```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\install
python dreamplace\Placer.py test\simple.json
```

**Other benchmark examples**:
```cmd
python dreamplace\Placer.py test\ispd2005\adaptec1.json
python dreamplace\Placer.py test\ispd2005\bigblue1.json
python dreamplace\Placer.py test\ispd2005\bigblue2.json
python dreamplace\Placer.py test\ispd2005\bigblue3.json
python dreamplace\Placer.py test\ispd2005\bigblue4.json
```

**View available options**:
```cmd
python dreamplace\Placer.py --help
```

#### 1.3 Run Individual Unit Tests (Testing Individual Operators)

Test specific PyTorch operators used in placement:

**Half-Perimeter Wirelength (HPWL)**:
```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\install
python unittest\ops\hpwl_unittest.py
```

**Weighted Average Wirelength**:
```cmd
python unittest\ops\weighted_average_wirelength_unittest.py
```

**LogSumExp Wirelength**:
```cmd
python unittest\ops\logsumexp_wirelength_unittest.py
```

**Density Overflow Calculation**:
```cmd
python unittest\ops\density_overflow_unittest.py
```

**Electric Potential (Density Penalty)**:
```cmd
python unittest\ops\electric_potential_unittest.py
python unittest\ops\dct_electric_potential_unittest.py
```

**Density Potential**:
```cmd
python unittest\ops\density_potential_unittest.py
```

**Legalization**:
```cmd
python unittest\ops\abacus_legalize_unittest\abacus_legalize_unittest.py
python unittest\ops\greedy_legalize_unittest\greedy_legalize_unittest.py
python unittest\ops\macro_legalize_unittest\macro_legalize_unittest.py
```

**Global Swap (Detailed Placement)**:
```cmd
python unittest\ops\global_swap_unittest\global_swap_unittest.py
```

**Independent Set Matching**:
```cmd
python unittest\ops\independent_set_matching_unittest\independent_set_matching_unittest.py
```

**K-Reorder**:
```cmd
python unittest\ops\k_reorder_unittest\k_reorder_unittest.py
```

**Routing Utilization (RUDY/PinRUDY)**:
```cmd
python unittest\ops\rudy_unittest.py
python unittest\ops\pinrudy_unittest.py
python unittest\ops\pin_utilization_unittest.py
```

**Rectilinear Minimum Spanning Tree Wirelength**:
```cmd
python unittest\ops\rmst_wl_unittest.py
```

**Drawing and Visualization**:
```cmd
python unittest\ops\draw_place_unittest.py
```

**Pin Position Calculation**:
```cmd
python unittest\ops\pin_pos_unittest.py
```

**Move Boundary Checking**:
```cmd
python unittest\ops\move_boundary_unittest.py
```

**Node Area Adjustment**:
```cmd
python unittest\ops\adjust_node_area_unittest.py
```

**FFT Operations**:
```cmd
python unittest\ops\torch_fft_unittest.py
```

**DCT Operations**:
```cmd
python unittest\ops\dct_unittest.py
```

**I/O Placement**:
```cmd
python unittest\ops\place_io_unittest\place_io_unittest.py
```

#### 1.4 Run All Unit Tests

Execute all unit tests at once:
```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\install
python unittest\unittests.py
```

---

## 📦 Component 2: nthuRouter3 (Global Routing Tool)

**Description**: NTHU Router 3 for global routing in VLSI design. This is a C++ based routing tool.

### Prerequisites

**System Dependencies**:
- GCC/G++ compiler
- Make build system
- pthread library

### Build

```cmd
cd D:\work\IDE\PyCharm\FinalProject\nthuRouter3
make
```

This will compile the source code from the following directories:
- `src/flute/` - FLUTE Steiner tree construction
- `src/grdb/` - Global routing database
- `src/misc/` - Miscellaneous utilities
- `src/router/` - Main routing algorithms
- `src/util/` - Utility functions
- `src/spdlog/` - Logging framework

The output will be an executable named `NthuRoute`.

### Run

```cmd
cd D:\work\IDE\PyCharm\FinalProject\nthuRouter3
.\NthuRoute <input_file>
```

**Example with provided benchmark**:
```cmd
.\NthuRoute adaptec1.capo70.2d.35.50.90.gr
```

**Output**: The router will generate routing results in the `output` file.

### Clean Build

```cmd
cd D:\work\IDE\PyCharm\FinalProject\nthuRouter3
make clean
```

---

## 🔄 Typical Workflow

### Complete VLSI Design Flow

1. **Placement (DREAMPlace)**:
   ```cmd
   cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\install
   python dreamplace\Placer.py test\ispd2005\adaptec1.json
   ```
   - Outputs: `.pl` (placement), `.def` (design exchange format)

2. **Routing (nthuRouter3)**:
   ```cmd
   cd D:\work\IDE\PyCharm\FinalProject\nthuRouter3
   .\NthuRoute <placement_result>.gr
   ```
   - Outputs: Global routing solution

---

## 🧪 Testing Individual Components

### Test DREAMPlace Operators Only

To test just the placement algorithms without running full benchmarks:

```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\install
python unittest\ops\<specific_test>_unittest.py
```

### Test Simple Benchmark

Use the simple test case for quick validation:

```cmd
cd D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\install
python dreamplace\Placer.py test\simple.json
```

---

## 📊 Configuration

### DREAMPlace JSON Configuration

Key parameters in JSON files (e.g., `simple.json`):

- `aux_input` / `lef_input` / `def_input`: Input benchmark files
- `gpu`: Enable GPU acceleration (1) or CPU only (0)
- `num_bins_x`, `num_bins_y`: Bin grid dimensions for density calculation
- `target_density`: Target placement density (0.0-1.0)
- `global_place_flag`: Enable global placement (1/0)
- `legalize_flag`: Enable legalization (1/0)
- `detailed_place_flag`: Enable detailed placement (1/0)
- `stop_overflow`: Overflow threshold to stop global placement
- `dtype`: Data type (`float32` or `float64`)

Example configurations are in:
- `test\simple.json` - Simple test case
- `test\ispd2005\*.json` - ISPD 2005 benchmarks

---

## 🐛 Debugging and Development

### Run with Debug Information

Set logging level in Python:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Individual Operator Behavior

Each unit test can be modified to print intermediate results. For example, edit:
```
D:\work\IDE\PyCharm\FinalProject\DREAMPlace_local\unittest\ops\hpwl_unittest.py
```

---

## 📝 Notes

1. **GPU Support**: DREAMPlace can run on both CPU and GPU. GPU provides 30X+ speedup.
2. **Benchmark Sizes**: Start with `simple.json` for testing, then move to larger ISPD benchmarks.
3. **Memory Requirements**: Large benchmarks may require significant RAM (8GB+) and GPU memory.
4. **Build Time**: Initial build of DREAMPlace can take 15-30 minutes depending on your system.
5. **Windows Compatibility**: Some scripts may need adjustments for Windows paths and shell commands.

---

## 🆘 Troubleshooting

### DREAMPlace won't build
- Ensure all dependencies (Boost, Bison, CMake) are installed and in PATH
- Check GCC version (recommend 7.5)
- Verify PyTorch installation matches CUDA version (if using GPU)

### Unit tests fail
- Ensure DREAMPlace is properly installed to the `install/` directory
- Check that Python can find the compiled C++ extensions

### nthuRouter3 won't compile
- Verify GCC/G++ is installed
- Check that pthread library is available
- Try running `make clean` then `make`

---

## 📚 Further Information

- **DREAMPlace Documentation**: See `DREAMPlace_local\README.md`
- **Publications**: Multiple papers published at DAC, ICCAD, TCAD, DATE
- **Source Code**: Main placement logic in `dreamplace\Placer.py`, `NonLinearPlace.py`, `BasicPlace.py`
- **Operators**: Individual operators in `dreamplace\ops\` directory


# VLSI Placement-to-Routing Flow - Usage Guide

## Quick Start

### 1. Run Complete Flow
```bash
# Full flow: Placement → Conversion → Routing
python run_complete_flow.py adaptec1

# With custom parameters
python run_complete_flow.py adaptec1 --gpu 1 --iterations 2000 --density 0.9 --tile-size 35
```

### 2. Run Individual Steps

**Placement only:**
```bash
python run_placement_only.py adaptec1 --gpu 1 --iterations 2000 --density 0.9
```

**Conversion only:**
```bash
python run_converter_only.py adaptec1 --tile-size 35 --adj-factor 50 --safe-guard 90
```

**Routing only:**
```bash
python run_routing_only.py adaptec1
```

---

## Available Benchmarks

Located in: `DREAMPlace_local/install/benchmarks/ispd2005/`

- adaptec1
- adaptec2
- adaptec3
- adaptec4
- bigblue1
- bigblue2
- bigblue3
- bigblue4

---

## Parameters

### Placement Parameters
- `--gpu` - Use GPU (1) or CPU (0). Default: 1
- `--iterations` - Number of placement iterations. Default: 2000
- `--density` - Target cell density (0.0-1.0). Default: 0.9

### Conversion Parameters
- `--tile-size` - Routing grid tile size. Default: 35
- `--adj-factor` - Blockage adjustment factor (%). Default: 50
- `--safe-guard` - Base capacity factor (%). Default: 90

---

## Output Files

### Placement Output
`DREAMPlace_local/install/results/{benchmark}/{benchmark}.gp.pl`

### Routing Input
`test_result/{benchmark}_routing_input.gr`

### Routing Output
`test_result/{benchmark}_routing_output.txt`

---

## Example Full Workflow

```bash
# Step 1: Ensure benchmarks are downloaded
cd DREAMPlace
python benchmarks/ispd2005_2015.py
cd ..

# Step 2: Build router (first time only)
cd nthuRouter3
make
cd ..

# Step 3: Run complete flow
python run_complete_flow.py adaptec1 --gpu 1 --iterations 2000

# Step 4: Check results
ls test_result/
```

---

## Troubleshooting

**GPU not available:**
```bash
python run_complete_flow.py adaptec1 --gpu 0
```

**Routing has overflow:**
- Increase `--safe-guard` to 95-100
- Decrease `--adj-factor` to 30-40
- Increase `--tile-size` to 40-50

**Placement doesn't converge:**
- Increase `--iterations` to 3000-5000
- Adjust `--density` to 0.8-0.95

---

## Advanced Usage

### Custom Flow Script
```python
from run_complete_flow import VLSIFlow

flow = VLSIFlow("adaptec1")

# Run each step with custom settings
flow.run_placement(use_gpu=1, iterations=2000, target_density=0.9)
flow.run_converter(tile_size=35, adjustment_factor=50, safe_guard_factor=90)
flow.run_routing()
```

### Batch Processing
```bash
# Run multiple benchmarks
for bench in adaptec1 adaptec2 adaptec3; do
    python run_complete_flow.py $bench
done
```


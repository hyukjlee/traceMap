## Overview

This tool generates interactive HTML dashboards for analyzing PyTorch profiler trace files (`.pt.trace.json.gz`) from vLLM workloads. It provides visualization capabilities for GPU kernel profiling and performance analysis.
![image](https://github.com/user-attachments/assets/aecef59e-f76b-458f-a515-1e2f2dacbbcc)

### Features
- Interactive HTML DashboardGenerate a standalone HTML report to zoom, pan, and inspect individual kernel execution events.
- Side-by-Side Trace ComparisonCompare two trace files to easily spot regressions or improvements in kernel-level performance.
- Optimized for LLM InferenceTailored for vLLM traces, with attention to CUDA graph behavior, TTFT (time-to-first-token), and kernel throughput.
- Lightweight and PortableNo server needed. Outputs a self-contained HTML file viewable in any modern browser.

## Prerequisites

- **OS**: Linux 
- **Python**: 3.10 - 3.12

## Environment Setup

### Option 1: Using venv (Recommended)

1. **Create virtual environment:**
   ```bash
   python3 -m venv tracemap
   source tracemap/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install bokeh pandas numpy
   ```

## Usage

### Basic Usage

Run the profiling dashboard generator with default trace files:

```bash
python3 main.py
```

### Custom Trace Files

Specify your own trace files for comparison:

```bash
python3 main.py \
    --trace1 /path/to/first_trace.pt.trace.json.gz \
    --trace2 /path/to/second_trace.pt.trace.json.gz \
    --name1 "Trace_A_Name" \
    --name2 "Trace_B_Name" \
    --output custom_dashboard.html
```

### Command Line Arguments

- `--trace1`: Path to first trace file (default: `./trace_file/examples/trace1.pt.trace.json.gz`)
- `--trace2`: Path to second trace file (default: `./trace_file/examples/trace2.pt.trace.json.gz`)
- `--name1`: Name for first trace (default: `Trace_A`)
- `--name2`: Name for second trace (default: `Trace_B`)
- `--output`: Output HTML file name (default: `gpu_trace_profiling.html`)

## Output

The tool generates an interactive HTML dashboard that includes:
- GPU kernel execution timelines
- Performance comparisons between different traces
- Interactive visualizations for detailed analysis
- Summary statistics and profiling metrics

Open the generated HTML file in your web browser to explore the profiling results.

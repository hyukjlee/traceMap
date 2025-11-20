import argparse
import datetime
from pathlib import Path
from bokeh.plotting import save, output_file
from src.chart import GPUTraceDashboard


# Ensure traceMap outputs land under the shared benchNap directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHNAP_DIR = PROJECT_ROOT / "benchNap"
TRACE_OUTPUT_DIR = BENCHNAP_DIR / "trace_outputs"
TRACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Generate GPU Kernel Profiling Dashboard')
    parser.add_argument('--trace1', default="./trace_file/examples/trace1.pt.trace.json.gz", 
                       help='Path to first trace file')
    parser.add_argument('--trace2', default="./trace_file/examples/trace2.pt.trace.json.gz",
                       help='Path to second trace file')
    parser.add_argument('--name1', default="Trace_A", 
                       help='Name for first trace (default: Trace_A)')
    parser.add_argument('--name2', default="Trace_B", 
                       help='Name for second trace (default: Trace_B)')
    
    parser.add_argument('--output', default="tm.html",
                       help='Output HTML file name (default: tm_{timestamp}.html)')
    parser.add_argument('--csv',
                       help='Optional path for CSV/XLSX export with detailed kernel data')
    parser.add_argument('--layers', type=int,
                       help='Expected number of repeated layers to prioritize (e.g., 36)')
    
    args = parser.parse_args()
    
    # Add timestamp to output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.output)
    output_filename = TRACE_OUTPUT_DIR / f"{base_output.stem}_{timestamp}.html"
    
    # Create and save the visualization
    output_file(str(output_filename), title="GPU Kernel Profiling Dashboard")
    dashboard = GPUTraceDashboard(args.trace1, args.trace2, args.name1, args.name2)
    layout = dashboard.create_visualization()
    save(layout)
    print(f"Dashboard saved to {output_filename}")

    if args.csv:
        csv_basename = Path(args.csv)
        csv_suffix = csv_basename.suffix or ".xlsx"
        csv_output = TRACE_OUTPUT_DIR / f"{csv_basename.stem}_{timestamp}{csv_suffix}"
        csv_path = dashboard.export_csv_report(
            csv_output,
            unique_kernel_file="unique_kernels.txt",
            total_layers=args.layers
        )
        print(f"CSV report saved to {csv_path}")

if __name__ == "__main__":
    main()

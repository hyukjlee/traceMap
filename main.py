import argparse
import pandas as pd
from bokeh.plotting import figure, save, output_file
from src.chart import GPUTraceDashboard

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
    parser.add_argument('--output', default="gpu_trace_profiling.html",
                       help='Output HTML file name (default: gpu_trace_profiling.html)')
    
    args = parser.parse_args()
    
   # Create and save the visualization
    output_file(args.output, title="GPU Kernel Profiling Dashboard")
    dashboard = GPUTraceDashboard(args.trace1, args.trace2, args.name1, args.name2)
    layout = dashboard.create_visualization()
    save(layout)
    print(f"Dashboard saved to {args.output}")

if __name__ == "__main__":
    main() 
import argparse
import bokeh
import gzip
import json
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, DataTable, TableColumn, Slider, 
    CustomJS, Div, Select, NumberFormatter, Tabs, TabPanel,
    Spinner
)
from bokeh.io import curdoc
from collections import Counter

def extract_kernel_data(trace_path):
    with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
        trace_data = json.load(f)
    events = trace_data.get("traceEvents", [])
    kernel_events = []
    for event in events:
        if event.get("ph") == "X" and "kernel" in event.get("cat", "").lower():
            kernel_name = event.get("name", "")
            start = event.get("ts", 0)
            duration = event.get("dur", 0)
            end = start + duration
            kernel_events.append((kernel_name, start, duration, end))
    if kernel_events:
        base_time = kernel_events[0][1]
        parsed = [{
            "Kernel Index": idx,
            "Kernel Name": name,
            "Start (us)": round(start - base_time, 3),
            "Duration (us)": round(duration, 3),
            "End (us)": round(end - base_time, 3),
        } for idx, (name, start, duration, end) in enumerate(kernel_events)]
        return pd.DataFrame(parsed)
    else:
        return pd.DataFrame(columns=["Kernel Index", "Kernel Name", "Start (us)", "Duration (us)", "End (us)"])

def create_top_n_data(df, n=10):
    """Create top N kernels by total latency and counts"""
    kernel_stats = df.groupby('Kernel Name').agg({
        'Duration (us)': ['sum', 'count', 'mean']
    }).round(3)
    kernel_stats.columns = ['Total Duration (us)', 'Count', 'Avg Duration (us)']
    kernel_stats = kernel_stats.sort_values('Total Duration (us)', ascending=False).head(n)
    kernel_stats = kernel_stats.reset_index()
    return kernel_stats

def create_sorted_latency_data(df):
    """Create sorted kernel data by latency for individual kernels"""
    sorted_df = df.sort_values('Duration (us)', ascending=False).reset_index(drop=True)
    return sorted_df

def create_visualization(trace_path1, trace_path2, gpu_name_a="GPU_A", gpu_name_b="GPU_B"):
    # Load traces
    df_gpu_a = extract_kernel_data(trace_path1)
    df_gpu_b = extract_kernel_data(trace_path2)
    
    # Create data sources
    source_gpu_a = ColumnDataSource(df_gpu_a)
    source_gpu_b = ColumnDataSource(df_gpu_b)
    
    # Create filtered sources for sliding window
    default_window_size = 100  # Default window size
    
    source_gpu_a_filtered = ColumnDataSource(df_gpu_a.head(default_window_size))
    source_gpu_b_filtered = ColumnDataSource(df_gpu_b.head(default_window_size))
    
    # Sorted filtered sources - these will be sorted versions of the filtered data
    initial_gpu_a_sorted = df_gpu_a.head(default_window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
    initial_gpu_b_sorted = df_gpu_b.head(default_window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
    
    source_sorted_gpu_a_filtered = ColumnDataSource(initial_gpu_a_sorted)
    source_sorted_gpu_b_filtered = ColumnDataSource(initial_gpu_b_sorted)
    
    # For combined view - create separate filtered sources for side-by-side comparison
    source_gpu_a_combined_filtered = ColumnDataSource(df_gpu_a.head(default_window_size))
    source_gpu_b_combined_filtered = ColumnDataSource(df_gpu_b.head(default_window_size))
    
    # Combined sorted filtered sources
    source_sorted_gpu_a_combined_filtered = ColumnDataSource(initial_gpu_a_sorted)
    source_sorted_gpu_b_combined_filtered = ColumnDataSource(initial_gpu_b_sorted)
    
    # Create top N data sources
    top_n_gpu_a = create_top_n_data(df_gpu_a)
    top_n_gpu_b = create_top_n_data(df_gpu_b)
    top_n_both = create_top_n_data(pd.concat([df_gpu_a, df_gpu_b], ignore_index=True))
    
    source_top_gpu_a = ColumnDataSource(top_n_gpu_a)
    source_top_gpu_b = ColumnDataSource(top_n_gpu_b)
    source_top_both = ColumnDataSource(top_n_both)
    
    # Window size controls
    window_size_spinner = Spinner(title="Window Size:", low=10, high=1000, step=10, value=default_window_size, width=150)
    window_size_spinner_combined = Spinner(title="Window Size:", low=10, high=1000, step=10, value=default_window_size, width=150)
    
    # Create bar charts
    def create_bar_chart(title, source_filtered, color, width=800):
        p = figure(
            title=title,
            x_axis_label="Kernel Index",
            y_axis_label="Duration (us)",
            width=width,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save,tap"
        )
        bars = p.vbar(x='Kernel Index', top='Duration (us)', width=0.8, 
                     source=source_filtered, color=color, alpha=0.7)
        return p, bars
    
    # GPU A Chart
    chart_gpu_a, bars_gpu_a = create_bar_chart(f"{gpu_name_a} Kernel Latency", source_gpu_a_filtered, "blue", width=2000)
    
    # GPU B Chart  
    chart_gpu_b, bars_gpu_b = create_bar_chart(f"{gpu_name_b} Kernel Latency", source_gpu_b_filtered, "red", width=2000)
    
    # Combined Charts - Side by Side
    chart_gpu_a_combined, bars_gpu_a_combined = create_bar_chart(f"{gpu_name_a} Kernel Latency", source_gpu_a_combined_filtered, "blue", width=1000)
    chart_gpu_b_combined, bars_gpu_b_combined = create_bar_chart(f"{gpu_name_b} Kernel Latency", source_gpu_b_combined_filtered, "red", width=1000)
    
    # Create sliders for sliding windows
    slider_gpu_a = Slider(
        start=0, end=max(0, len(df_gpu_a) - default_window_size), value=0, 
        step=default_window_size, title=f"{gpu_name_a} Kernel Index Window (showing {default_window_size} at a time)", width=1000
    )
    
    slider_gpu_b = Slider(
        start=0, end=max(0, len(df_gpu_b) - default_window_size), value=0, 
        step=default_window_size, title=f"{gpu_name_b} Kernel Index Window (showing {default_window_size} at a time)", width=1000
    )
    
    # Combined sliders - separate for each GPU
    slider_gpu_a_combined = Slider(
        start=0, end=max(0, len(df_gpu_a) - default_window_size), value=0, 
        step=default_window_size, title=f"{gpu_name_a} Kernel Index Window (showing {default_window_size} at a time)", width=1000
    )
    
    slider_gpu_b_combined = Slider(
        start=0, end=max(0, len(df_gpu_b) - default_window_size), value=0, 
        step=default_window_size, title=f"{gpu_name_b} Kernel Index Window (showing {default_window_size} at a time)", width=1000
    )
    
    # Create data tables
    columns = [
        TableColumn(field="Kernel Index", title="Index", width=80),
        TableColumn(field="Kernel Name", title="Kernel Name", width=600),
        TableColumn(field="Start (us)", title="Start (μs)", width=100, 
                   formatter=NumberFormatter(format="0,0.000")),
        TableColumn(field="Duration (us)", title="Duration (μs)", width=120, 
                   formatter=NumberFormatter(format="0,0.000")),
        TableColumn(field="End (us)", title="End (μs)", width=100, 
                   formatter=NumberFormatter(format="0,0.000")),
    ]
    
    table_gpu_a = DataTable(source=source_gpu_a_filtered, columns=columns, 
                          width=2000, height=600, index_position=None)
    table_gpu_b = DataTable(source=source_gpu_b_filtered, columns=columns, 
                            width=2000, height=600, index_position=None)
    
    # Sorted latency tables
    sorted_table_gpu_a = DataTable(source=source_sorted_gpu_a_filtered, columns=columns, 
                                  width=2000, height=600, index_position=None)
    sorted_table_gpu_b = DataTable(source=source_sorted_gpu_b_filtered, columns=columns, 
                                   width=2000, height=600, index_position=None)
    
    # Combined tables - side by side
    table_gpu_a_combined = DataTable(source=source_gpu_a_combined_filtered, columns=columns, 
                                   width=1000, height=600, index_position=None)
    table_gpu_b_combined = DataTable(source=source_gpu_b_combined_filtered, columns=columns, 
                                     width=1000, height=600, index_position=None)
    
    # Combined sorted tables
    sorted_table_gpu_a_combined = DataTable(source=source_sorted_gpu_a_combined_filtered, columns=columns, 
                                          width=1000, height=600, index_position=None)
    sorted_table_gpu_b_combined = DataTable(source=source_sorted_gpu_b_combined_filtered, columns=columns, 
                                            width=1000, height=600, index_position=None)
    
    # Create top N tables
    top_columns = [
        TableColumn(field="Kernel Name", title="Kernel Name", width=680),
        TableColumn(field="Total Duration (us)", title="Total Duration (μs)", width=120,
                   formatter=NumberFormatter(format="0,0.000")),
        TableColumn(field="Count", title="Count", width=60),
        TableColumn(field="Avg Duration (us)", title="Avg Duration (μs)", width=120,
                   formatter=NumberFormatter(format="0,0.000")),
    ]
    
    top_table_gpu_a = DataTable(source=source_top_gpu_a, columns=top_columns, 
                              width=2000, height=600, index_position=None)
    top_table_gpu_b = DataTable(source=source_top_gpu_b, columns=top_columns, 
                                width=2000, height=600, index_position=None)
    top_table_both = DataTable(source=source_top_both, columns=top_columns, 
                              width=800, height=300, index_position=None)
    
    # Combined top N tables - side by side
    top_table_gpu_a_combined = DataTable(source=source_top_gpu_a, columns=top_columns, 
                                       width=1000, height=600, index_position=None)
    top_table_gpu_b_combined = DataTable(source=source_top_gpu_b, columns=top_columns, 
                                         width=1000, height=600, index_position=None)
    
    # Helper function to create sorted data from filtered data
    def create_sorted_data_js():
        return """
        function createSortedData(filtered_data) {
            const indices = [];
            const length = filtered_data['Kernel Index'].length;
            for (let i = 0; i < length; i++) {
                indices.push(i);
            }
            
            // Sort indices by Duration (us) in descending order
            indices.sort((a, b) => filtered_data['Duration (us)'][b] - filtered_data['Duration (us)'][a]);
            
            const sorted_data = {};
            for (let key in filtered_data) {
                sorted_data[key] = indices.map(i => filtered_data[key][i]);
            }
            return sorted_data;
        }
        """
    
    # Window size change callback
    window_size_callback = CustomJS(
        args=dict(
            spinner=window_size_spinner,
            slider_gpu_a=slider_gpu_a,
            slider_gpu_b=slider_gpu_b,
            source_gpu_a=source_gpu_a,
            source_gpu_b=source_gpu_b,
            source_gpu_a_filtered=source_gpu_a_filtered,
            source_gpu_b_filtered=source_gpu_b_filtered,
            source_sorted_gpu_a_filtered=source_sorted_gpu_a_filtered,
            source_sorted_gpu_b_filtered=source_sorted_gpu_b_filtered,
            gpu_name_a=gpu_name_a,
            gpu_name_b=gpu_name_b
        ),
        code=create_sorted_data_js() + """
        const window_size = spinner.value;
        
        // Update slider properties
        slider_gpu_a.step = window_size;
        slider_gpu_b.step = window_size;
        slider_gpu_a.end = Math.max(0, source_gpu_a.data['Kernel Index'].length - window_size);
        slider_gpu_b.end = Math.max(0, source_gpu_b.data['Kernel Index'].length - window_size);
        slider_gpu_a.title = `${gpu_name_a} Kernel Index Window (showing ${window_size} at a time)`;
        slider_gpu_b.title = `${gpu_name_b} Kernel Index Window (showing ${window_size} at a time)`;
        
        // Update filtered data
        const start_gpu_a = slider_gpu_a.value;
        const end_gpu_a = Math.min(start_gpu_a + window_size, source_gpu_a.data['Kernel Index'].length);
        
        const start_gpu_b = slider_gpu_b.value;
        const end_gpu_b = Math.min(start_gpu_b + window_size, source_gpu_b.data['Kernel Index'].length);
        
        // Update GPU A filtered data
        const gpu_a_filtered = {};
        for (let key in source_gpu_a.data) {
            gpu_a_filtered[key] = source_gpu_a.data[key].slice(start_gpu_a, end_gpu_a);
        }
        source_gpu_a_filtered.data = gpu_a_filtered;
        
        // Update GPU B filtered data
        const gpu_b_filtered = {};
        for (let key in source_gpu_b.data) {
            gpu_b_filtered[key] = source_gpu_b.data[key].slice(start_gpu_b, end_gpu_b);
        }
        source_gpu_b_filtered.data = gpu_b_filtered;
        
        // Update sorted filtered data by sorting the current window
        source_sorted_gpu_a_filtered.data = createSortedData(gpu_a_filtered);
        source_sorted_gpu_b_filtered.data = createSortedData(gpu_b_filtered);
        
        source_gpu_a_filtered.change.emit();
        source_gpu_b_filtered.change.emit();
        source_sorted_gpu_a_filtered.change.emit();
        source_sorted_gpu_b_filtered.change.emit();
        """
    )
    
    # Callback for combined view
    window_size_callback_combined = CustomJS(
        args=dict(
            spinner=window_size_spinner_combined,
            slider_gpu_a=slider_gpu_a_combined,
            slider_gpu_b=slider_gpu_b_combined,
            source_gpu_a=source_gpu_a,
            source_gpu_b=source_gpu_b,
            source_gpu_a_filtered=source_gpu_a_combined_filtered,
            source_gpu_b_filtered=source_gpu_b_combined_filtered,
            source_sorted_gpu_a_filtered=source_sorted_gpu_a_combined_filtered,
            source_sorted_gpu_b_filtered=source_sorted_gpu_b_combined_filtered,
            gpu_name_a=gpu_name_a,
            gpu_name_b=gpu_name_b
        ),
        code=create_sorted_data_js() + """
        const window_size = spinner.value;
        
        slider_gpu_a.step = window_size;
        slider_gpu_b.step = window_size;
        slider_gpu_a.end = Math.max(0, source_gpu_a.data['Kernel Index'].length - window_size);
        slider_gpu_b.end = Math.max(0, source_gpu_b.data['Kernel Index'].length - window_size);
        slider_gpu_a.title = `${gpu_name_a} Kernel Index Window (showing ${window_size} at a time)`;
        slider_gpu_b.title = `${gpu_name_b} Kernel Index Window (showing ${window_size} at a time)`;
        
        const start_gpu_a = slider_gpu_a.value;
        const end_gpu_a = Math.min(start_gpu_a + window_size, source_gpu_a.data['Kernel Index'].length);
        
        const start_gpu_b = slider_gpu_b.value;
        const end_gpu_b = Math.min(start_gpu_b + window_size, source_gpu_b.data['Kernel Index'].length);
        
        const gpu_a_filtered = {};
        for (let key in source_gpu_a.data) {
            gpu_a_filtered[key] = source_gpu_a.data[key].slice(start_gpu_a, end_gpu_a);
        }
        source_gpu_a_filtered.data = gpu_a_filtered;
        
        const gpu_b_filtered = {};
        for (let key in source_gpu_b.data) {
            gpu_b_filtered[key] = source_gpu_b.data[key].slice(start_gpu_b, end_gpu_b);
        }
        source_gpu_b_filtered.data = gpu_b_filtered;
        
        source_sorted_gpu_a_filtered.data = createSortedData(gpu_a_filtered);
        source_sorted_gpu_b_filtered.data = createSortedData(gpu_b_filtered);
        
        source_gpu_a_filtered.change.emit();
        source_gpu_b_filtered.change.emit();
        source_sorted_gpu_a_filtered.change.emit();
        source_sorted_gpu_b_filtered.change.emit();
        """
    )
    
    # Attach window size callbacks
    window_size_spinner.js_on_change('value', window_size_callback)
    window_size_spinner_combined.js_on_change('value', window_size_callback_combined)
    
    # JavaScript callbacks for sliding windows
    slider_callback_gpu_a = CustomJS(
        args=dict(
            source=source_gpu_a, 
            source_filtered=source_gpu_a_filtered, 
            source_sorted_filtered=source_sorted_gpu_a_filtered,
            slider=slider_gpu_a,
            spinner=window_size_spinner
        ),
        code=create_sorted_data_js() + """
        const start = slider.value;
        const window_size = spinner.value;
        const end = Math.min(start + window_size, source.data['Kernel Index'].length);
        
        const filtered_data = {};
        for (let key in source.data) {
            filtered_data[key] = source.data[key].slice(start, end);
        }
        source_filtered.data = filtered_data;
        
        // Create sorted version of the current window
        source_sorted_filtered.data = createSortedData(filtered_data);
        
        source_filtered.change.emit();
        source_sorted_filtered.change.emit();
        """
    )
    
    slider_callback_gpu_b = CustomJS(
        args=dict(
            source=source_gpu_b, 
            source_filtered=source_gpu_b_filtered, 
            source_sorted_filtered=source_sorted_gpu_b_filtered,
            slider=slider_gpu_b,
            spinner=window_size_spinner
        ),
        code=create_sorted_data_js() + """
        const start = slider.value;
        const window_size = spinner.value;
        const end = Math.min(start + window_size, source.data['Kernel Index'].length);
        
        const filtered_data = {};
        for (let key in source.data) {
            filtered_data[key] = source.data[key].slice(start, end);
        }
        source_filtered.data = filtered_data;
        
        source_sorted_filtered.data = createSortedData(filtered_data);
        
        source_filtered.change.emit();
        source_sorted_filtered.change.emit();
        """
    )
    
    # Combined callbacks
    slider_callback_gpu_a_combined = CustomJS(
        args=dict(
            source=source_gpu_a, 
            source_filtered=source_gpu_a_combined_filtered,
            source_sorted_filtered=source_sorted_gpu_a_combined_filtered,
            slider=slider_gpu_a_combined,
            spinner=window_size_spinner_combined
        ),
        code=create_sorted_data_js() + """
        const start = slider.value;
        const window_size = spinner.value;
        const end = Math.min(start + window_size, source.data['Kernel Index'].length);
        
        const filtered_data = {};
        for (let key in source.data) {
            filtered_data[key] = source.data[key].slice(start, end);
        }
        source_filtered.data = filtered_data;
        
        source_sorted_filtered.data = createSortedData(filtered_data);
        
        source_filtered.change.emit();
        source_sorted_filtered.change.emit();
        """
    )
    
    slider_callback_gpu_b_combined = CustomJS(
        args=dict(
            source=source_gpu_b, 
            source_filtered=source_gpu_b_combined_filtered,
            source_sorted_filtered=source_sorted_gpu_b_combined_filtered,
            slider=slider_gpu_b_combined,
            spinner=window_size_spinner_combined
        ),
        code=create_sorted_data_js() + """
        const start = slider.value;
        const window_size = spinner.value;
        const end = Math.min(start + window_size, source.data['Kernel Index'].length);
        
        const filtered_data = {};
        for (let key in source.data) {
            filtered_data[key] = source.data[key].slice(start, end);
        }
        source_filtered.data = filtered_data;
        
        source_sorted_filtered.data = createSortedData(filtered_data);
        
        source_filtered.change.emit();
        source_sorted_filtered.change.emit();
        """
    )
    
    # Bar click callbacks to highlight table rows 
    tap_callback_gpu_a = CustomJS(
        args=dict(source=source_gpu_a_filtered, table=table_gpu_a, sorted_table=sorted_table_gpu_a),
        code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            table.source.selected.indices = indices;
            sorted_table.source.selected.indices = indices;
            const row_height = 25;
            const scroll_top = indices[0] * row_height;
            table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
            sorted_table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
        }
        """
    )
    
    tap_callback_gpu_b = CustomJS(
        args=dict(source=source_gpu_b_filtered, table=table_gpu_b, sorted_table=sorted_table_gpu_b),
        code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            table.source.selected.indices = indices;
            sorted_table.source.selected.indices = indices;
            const row_height = 25;
            const scroll_top = indices[0] * row_height;
            table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
            sorted_table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
        }
        """
    )
    
    # Combined tap callbacks
    tap_callback_gpu_a_combined = CustomJS(
        args=dict(source=source_gpu_a_combined_filtered, table=table_gpu_a_combined, sorted_table=sorted_table_gpu_a_combined),
        code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            table.source.selected.indices = indices;
            sorted_table.source.selected.indices = indices;
            const row_height = 25;
            const scroll_top = indices[0] * row_height;
            table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
            sorted_table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
        }
        """
    )
    
    tap_callback_gpu_b_combined = CustomJS(
        args=dict(source=source_gpu_b_combined_filtered, table=table_gpu_b_combined, sorted_table=sorted_table_gpu_b_combined),
        code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            table.source.selected.indices = indices;
            sorted_table.source.selected.indices = indices;
            const row_height = 25;
            const scroll_top = indices[0] * row_height;
            table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
            sorted_table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
        }
        """
    )
    
    # Attach callbacks
    slider_gpu_a.js_on_change('value', slider_callback_gpu_a)
    slider_gpu_b.js_on_change('value', slider_callback_gpu_b)
    slider_gpu_a_combined.js_on_change('value', slider_callback_gpu_a_combined)
    slider_gpu_b_combined.js_on_change('value', slider_callback_gpu_b_combined)
    
    bars_gpu_a.data_source.selected.js_on_change('indices', tap_callback_gpu_a)
    bars_gpu_b.data_source.selected.js_on_change('indices', tap_callback_gpu_b)
    bars_gpu_a_combined.data_source.selected.js_on_change('indices', tap_callback_gpu_a_combined)
    bars_gpu_b_combined.data_source.selected.js_on_change('indices', tap_callback_gpu_b_combined)
    
    # Create layouts for each tab
    gpu_a_layout = column(
        Div(text=f"<h2>{gpu_name_a} Kernel Analysis</h2>"),
        row(window_size_spinner, slider_gpu_a),
        chart_gpu_a,
        Div(text="<h3>Kernel Details Table</h3>"),
        table_gpu_a,
        Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"),
        sorted_table_gpu_a,
        Div(text="<h3>Top 10 Kernels by Total Duration</h3>"),
        top_table_gpu_a
    )
    
    gpu_b_layout = column(
        Div(text=f"<h2>{gpu_name_b} Kernel Analysis</h2>"),
        row(window_size_spinner, slider_gpu_b),
        chart_gpu_b,
        Div(text="<h3>Kernel Details Table</h3>"),
        table_gpu_b,
        Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"),
        sorted_table_gpu_b,
        Div(text="<h3>Top 10 Kernels by Total Duration</h3>"),
        top_table_gpu_b
    )
    
    # Updated combined layout with side-by-side comparison
    both_layout = column(
        Div(text="<h2>Side by Side Comparison</h2>"),        
        row(window_size_spinner_combined),
        row(slider_gpu_a_combined, slider_gpu_b_combined),
        row(chart_gpu_a_combined, chart_gpu_b_combined),
        Div(text="<h3>Kernel Details Tables</h3>"),
        row(
            column(Div(text=f"<h4>{gpu_name_a} Kernels</h4>"), table_gpu_a_combined),
            column(Div(text=f"<h4>{gpu_name_b} Kernels</h4>"), table_gpu_b_combined)
        ),
        Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"),
        row(
            column(Div(text=f"<h4>{gpu_name_a} Sorted by Latency</h4>"), sorted_table_gpu_a_combined),
            column(Div(text=f"<h4>{gpu_name_b} Sorted by Latency</h4>"), sorted_table_gpu_b_combined)
        ),
        Div(text="<h3>Top 10 Kernels Comparison</h3>"),
        row(
            column(Div(text=f"<h4>{gpu_name_a} Top Kernels</h4>"), top_table_gpu_a_combined),
            column(Div(text=f"<h4>{gpu_name_b} Top Kernels</h4>"), top_table_gpu_b_combined)
        ),
    )
    
    # Create tabs
    tab1 = TabPanel(child=gpu_a_layout, title=gpu_name_a)
    tab2 = TabPanel(child=gpu_b_layout, title=gpu_name_b) 
    tab3 = TabPanel(child=both_layout, title="Side-by-Side Comparison")
    
    tabs = Tabs(tabs=[tab1, tab2, tab3])
    
    # Final layout
    layout = column(               
        tabs
    )
    
    return layout

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
    layout = create_visualization(args.trace1, args.trace2, args.name1, args.name2)
    save(layout)
    print(f"Dashboard saved to {args.output}")

if __name__ == "__main__":
    main()
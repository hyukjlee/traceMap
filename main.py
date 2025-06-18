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
    Spinner, Button, TextAreaInput, TextInput
)
from bokeh.io import curdoc
from collections import Counter


class TraceAnalyzer:
    """Class to handle trace data extraction and processing"""
    
    def __init__(self, trace_path):
        self.trace_path = trace_path
        self.df = self.extract_kernel_data()
    
    def extract_kernel_data(self):
        """Extract kernel data from trace file"""
        with gzip.open(self.trace_path, 'rt', encoding='utf-8') as f:
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
    
    def create_top_n_data(self, n=10):
        """Create top N kernels by total latency and counts"""
        kernel_stats = self.df.groupby('Kernel Name').agg({
            'Duration (us)': ['sum', 'count', 'mean']
        }).round(3)
        kernel_stats.columns = ['Total Duration (us)', 'Count', 'Avg Duration (us)']
        kernel_stats = kernel_stats.sort_values('Total Duration (us)', ascending=False).head(n)
        return kernel_stats.reset_index()
    
    def create_sorted_latency_data(self):
        """Create sorted kernel data by latency for individual kernels"""
        return self.df.sort_values('Duration (us)', ascending=False).reset_index(drop=True)


class DataSourceManager:
    """Class to manage Bokeh data sources"""
    
    def __init__(self, trace_analyzer_a, trace_analyzer_b, default_window_size=100):
        self.trace_a = trace_analyzer_a
        self.trace_b = trace_analyzer_b
        self.window_size = default_window_size
        self._create_data_sources()
    
    def _create_data_sources(self):
        """Create all necessary data sources"""
        # Main data sources
        self.source_gpu_a = ColumnDataSource(self.trace_a.df)
        self.source_gpu_b = ColumnDataSource(self.trace_b.df)
        
        # Filtered sources for sliding window
        self.source_gpu_a_filtered = ColumnDataSource(self.trace_a.df.head(self.window_size))
        self.source_gpu_b_filtered = ColumnDataSource(self.trace_b.df.head(self.window_size))
        
        # Sorted filtered sources
        initial_gpu_a_sorted = self.trace_a.df.head(self.window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        initial_gpu_b_sorted = self.trace_b.df.head(self.window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        
        self.source_sorted_gpu_a_filtered = ColumnDataSource(initial_gpu_a_sorted)
        self.source_sorted_gpu_b_filtered = ColumnDataSource(initial_gpu_b_sorted)
        
        # Combined view sources
        self.source_gpu_a_combined_filtered = ColumnDataSource(self.trace_a.df.head(self.window_size))
        self.source_gpu_b_combined_filtered = ColumnDataSource(self.trace_b.df.head(self.window_size))
        
        self.source_sorted_gpu_a_combined_filtered = ColumnDataSource(initial_gpu_a_sorted)
        self.source_sorted_gpu_b_combined_filtered = ColumnDataSource(initial_gpu_b_sorted)
        
        # Top N data sources
        self.source_top_gpu_a = ColumnDataSource(self.trace_a.create_top_n_data())
        self.source_top_gpu_b = ColumnDataSource(self.trace_b.create_top_n_data())
        
        combined_df = pd.concat([self.trace_a.df, self.trace_b.df], ignore_index=True)
        combined_analyzer = TraceAnalyzer.__new__(TraceAnalyzer)
        combined_analyzer.df = combined_df
        self.source_top_both = ColumnDataSource(combined_analyzer.create_top_n_data())


class ChartManager:
    """Class to manage chart creation and interactions"""
    
    def __init__(self, data_sources, gpu_name_a="GPU_A", gpu_name_b="GPU_B"):
        self.ds = data_sources
        self.gpu_name_a = gpu_name_a
        self.gpu_name_b = gpu_name_b
        self.charts = {}
        self.bars = {}
        self._create_charts()
    
    def _create_bar_chart(self, title, source_filtered, color, width=800):
        """Create a bar chart"""
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
    
    def _create_charts(self):
        """Create all charts"""
        # Individual GPU charts
        self.charts['gpu_a'], self.bars['gpu_a'] = self._create_bar_chart(
            f"{self.gpu_name_a} Kernel Latency", self.ds.source_gpu_a_filtered, "blue", width=2000)
        
        self.charts['gpu_b'], self.bars['gpu_b'] = self._create_bar_chart(
            f"{self.gpu_name_b} Kernel Latency", self.ds.source_gpu_b_filtered, "red", width=2000)
        
        # Combined view charts
        self.charts['gpu_a_combined'], self.bars['gpu_a_combined'] = self._create_bar_chart(
            f"{self.gpu_name_a} Kernel Latency", self.ds.source_gpu_a_combined_filtered, "blue", width=1000)
        
        self.charts['gpu_b_combined'], self.bars['gpu_b_combined'] = self._create_bar_chart(
            f"{self.gpu_name_b} Kernel Latency", self.ds.source_gpu_b_combined_filtered, "red", width=1000)


class ControlManager:
    """Class to manage UI controls and interactions"""
    
    def __init__(self, data_sources, chart_manager, gpu_name_a="GPU_A", gpu_name_b="GPU_B", default_window_size=100):
        self.ds = data_sources
        self.chart_manager = chart_manager
        self.gpu_name_a = gpu_name_a
        self.gpu_name_b = gpu_name_b
        self.default_window_size = default_window_size
        self._create_controls()
        self._setup_callbacks()
    
    def _create_controls(self):
        """Create UI controls"""
        # Window size controls
        self.window_size_spinner = Spinner(
            title="Window Size:", low=10, high=1000, step=10, 
            value=self.default_window_size, width=150)
        self.window_size_spinner_combined = Spinner(
            title="Window Size:", low=10, high=1000, step=10, 
            value=self.default_window_size, width=150)
        
        # Sliders
        self.slider_gpu_a = Slider(
            start=0, end=max(0, len(self.ds.trace_a.df) - self.default_window_size), value=0, 
            step=self.default_window_size, 
            title=f"{self.gpu_name_a} Kernel Index Window (showing {self.default_window_size} at a time)", width=1000
        )
        
        self.slider_gpu_b = Slider(
            start=0, end=max(0, len(self.ds.trace_b.df) - self.default_window_size), value=0, 
            step=self.default_window_size, 
            title=f"{self.gpu_name_b} Kernel Index Window (showing {self.default_window_size} at a time)", width=1000
        )
        
        # Combined sliders
        self.slider_gpu_a_combined = Slider(
            start=0, end=max(0, len(self.ds.trace_a.df) - self.default_window_size), value=0, 
            step=self.default_window_size, 
            title=f"{self.gpu_name_a} Kernel Index Window (showing {self.default_window_size} at a time)", width=1000
        )
        
        self.slider_gpu_b_combined = Slider(
            start=0, end=max(0, len(self.ds.trace_b.df) - self.default_window_size), value=0, 
            step=self.default_window_size, 
            title=f"{self.gpu_name_b} Kernel Index Window (showing {self.default_window_size} at a time)", width=1000
        )
    
    def _create_sorted_data_js(self):
        """JavaScript function to create sorted data"""
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
    
    def _setup_callbacks(self):
        """Setup JavaScript callbacks"""
        # Window size callbacks
        window_size_callback = CustomJS(
            args=dict(
                spinner=self.window_size_spinner,
                slider_gpu_a=self.slider_gpu_a,
                slider_gpu_b=self.slider_gpu_b,
                source_gpu_a=self.ds.source_gpu_a,
                source_gpu_b=self.ds.source_gpu_b,
                source_gpu_a_filtered=self.ds.source_gpu_a_filtered,
                source_gpu_b_filtered=self.ds.source_gpu_b_filtered,
                source_sorted_gpu_a_filtered=self.ds.source_sorted_gpu_a_filtered,
                source_sorted_gpu_b_filtered=self.ds.source_sorted_gpu_b_filtered,
                gpu_name_a=self.gpu_name_a,
                gpu_name_b=self.gpu_name_b
            ),
            code=self._create_sorted_data_js() + """
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
        
        # Similar callback for combined view
        window_size_callback_combined = CustomJS(
            args=dict(
                spinner=self.window_size_spinner_combined,
                slider_gpu_a=self.slider_gpu_a_combined,
                slider_gpu_b=self.slider_gpu_b_combined,
                source_gpu_a=self.ds.source_gpu_a,
                source_gpu_b=self.ds.source_gpu_b,
                source_gpu_a_filtered=self.ds.source_gpu_a_combined_filtered,
                source_gpu_b_filtered=self.ds.source_gpu_b_combined_filtered,
                source_sorted_gpu_a_filtered=self.ds.source_sorted_gpu_a_combined_filtered,
                source_sorted_gpu_b_filtered=self.ds.source_sorted_gpu_b_combined_filtered,
                gpu_name_a=self.gpu_name_a,
                gpu_name_b=self.gpu_name_b
            ),
            code=self._create_sorted_data_js() + """
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
        self.window_size_spinner.js_on_change('value', window_size_callback)
        self.window_size_spinner_combined.js_on_change('value', window_size_callback_combined)
        
        # Slider callbacks
        self._setup_slider_callbacks()
    
    def _setup_slider_callbacks(self):
        """Setup slider callbacks"""
        slider_callback_gpu_a = CustomJS(
            args=dict(
                source=self.ds.source_gpu_a, 
                source_filtered=self.ds.source_gpu_a_filtered, 
                source_sorted_filtered=self.ds.source_sorted_gpu_a_filtered,
                slider=self.slider_gpu_a,
                spinner=self.window_size_spinner
            ),
            code=self._create_sorted_data_js() + """
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
        
        slider_callback_gpu_b = CustomJS(
            args=dict(
                source=self.ds.source_gpu_b, 
                source_filtered=self.ds.source_gpu_b_filtered, 
                source_sorted_filtered=self.ds.source_sorted_gpu_b_filtered,
                slider=self.slider_gpu_b,
                spinner=self.window_size_spinner
            ),
            code=self._create_sorted_data_js() + """
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
        
        # Combined callbacks (similar pattern)
        slider_callback_gpu_a_combined = CustomJS(
            args=dict(
                source=self.ds.source_gpu_a, 
                source_filtered=self.ds.source_gpu_a_combined_filtered,
                source_sorted_filtered=self.ds.source_sorted_gpu_a_combined_filtered,
                slider=self.slider_gpu_a_combined,
                spinner=self.window_size_spinner_combined
            ),
            code=self._create_sorted_data_js() + """
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
                source=self.ds.source_gpu_b, 
                source_filtered=self.ds.source_gpu_b_combined_filtered,
                source_sorted_filtered=self.ds.source_sorted_gpu_b_combined_filtered,
                slider=self.slider_gpu_b_combined,
                spinner=self.window_size_spinner_combined
            ),
            code=self._create_sorted_data_js() + """
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
        
        # Attach slider callbacks
        self.slider_gpu_a.js_on_change('value', slider_callback_gpu_a)
        self.slider_gpu_b.js_on_change('value', slider_callback_gpu_b)
        self.slider_gpu_a_combined.js_on_change('value', slider_callback_gpu_a_combined)
        self.slider_gpu_b_combined.js_on_change('value', slider_callback_gpu_b_combined)


class KernelDurationCalculator:
    """Class to handle kernel duration calculations"""
    
    def __init__(self, data_sources, gpu_name_a="GPU_A", gpu_name_b="GPU_B"):
        self.ds = data_sources
        self.gpu_name_a = gpu_name_a
        self.gpu_name_b = gpu_name_b
        self._create_calculator_widgets()
        self._setup_calculator_callbacks()
    
    def _create_calculator_widgets(self):
        """Create calculator UI widgets"""
        # Text input for kernel names
        self.kernel_input_a = TextAreaInput(
            value="", 
            title=f"Paste {self.gpu_name_a} Kernel Names (one per line):",
            rows=5, 
            width=500,
            placeholder="Paste kernel names here..."
        )
        
        self.kernel_input_b = TextAreaInput(
            value="", 
            title=f"Paste {self.gpu_name_b} Kernel Names (one per line):",
            rows=5, 
            width=500,
            placeholder="Paste kernel names here..."
        )
        
        # Calculate buttons
        self.calc_button_a = Button(label=f"Calculate {self.gpu_name_a} Total", button_type="success", width=200)
        self.calc_button_b = Button(label=f"Calculate {self.gpu_name_b} Total", button_type="success", width=200)
        
        # Result displays
        self.result_div_a = Div(text=f"<h4>{self.gpu_name_a} Results:</h4><p>Enter kernel names and click calculate</p>", width=500)
        self.result_div_b = Div(text=f"<h4>{self.gpu_name_b} Results:</h4><p>Enter kernel names and click calculate</p>", width=500)
        
        # Combined calculator
        self.kernel_input_combined = TextAreaInput(
            value="", 
            title="Paste Kernel Names (one per line) - will search both traces:",
            rows=5, 
            width=1000,
            placeholder="Paste kernel names here..."
        )
        
        self.calc_button_combined = Button(label="Calculate Both Traces", button_type="primary", width=200)
        self.result_div_combined = Div(text="<h4>Combined Results:</h4><p>Enter kernel names and click calculate</p>", width=1000)
    
    def _setup_calculator_callbacks(self):
        """Setup calculator callbacks"""
        # GPU A calculator
        calc_callback_a = CustomJS(
            args=dict(
                source=self.ds.source_gpu_a,
                input_widget=self.kernel_input_a,
                result_div=self.result_div_a,
                gpu_name=self.gpu_name_a
            ),
            code="""
            const kernel_names = input_widget.value.split('\\n').map(name => name.trim()).filter(name => name.length > 0);
            
            if (kernel_names.length === 0) {
                result_div.text = `<h4>${gpu_name} Results:</h4><p style="color: red;">Please enter at least one kernel name</p>`;
                return;
            }
            
            const data = source.data;
            let total_duration = 0;
            let found_count = 0;
            let results_html = `<h4>${gpu_name} Results:</h4>`;
            
            // Create a map for faster lookup
            const kernel_durations = {};
            for (let i = 0; i < data['Kernel Name'].length; i++) {
                const name = data['Kernel Name'][i];
                const duration = data['Duration (us)'][i];
                if (kernel_durations[name]) {
                    kernel_durations[name] += duration;
                } else {
                    kernel_durations[name] = duration;
                }
            }
            
            results_html += '<table style="border-collapse: collapse; width: 100%;">';
            results_html += '<tr style="background-color: #f0f0f0;"><th style="border: 1px solid #ddd; padding: 8px;">Kernel Name</th><th style="border: 1px solid #ddd; padding: 8px;">Total Duration (μs)</th><th style="border: 1px solid #ddd; padding: 8px;">Status</th></tr>';
            
            for (let kernel_name of kernel_names) {
                if (kernel_durations[kernel_name]) {
                    const duration = kernel_durations[kernel_name].toFixed(3);
                    total_duration += kernel_durations[kernel_name];
                    found_count++;
                    results_html += `<tr><td style="border: 1px solid #ddd; padding: 8px;">${kernel_name}</td><td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${duration}</td><td style="border: 1px solid #ddd; padding: 8px; color: green;">Found</td></tr>`;
                } else {
                    results_html += `<tr><td style="border: 1px solid #ddd; padding: 8px;">${kernel_name}</td><td style="border: 1px solid #ddd; padding: 8px; text-align: right;">0.000</td><td style="border: 1px solid #ddd; padding: 8px; color: red;">Not Found</td></tr>`;
                }
            }
            
            results_html += `<tr style="background-color: #e6f3ff; font-weight: bold;"><td style="border: 1px solid #ddd; padding: 8px;">TOTAL</td><td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${total_duration.toFixed(3)}</td><td style="border: 1px solid #ddd; padding: 8px;">${found_count}/${kernel_names.length} found</td></tr>`;
            results_html += '</table>';
            
            results_html += `<p><strong>Summary:</strong> Found ${found_count} out of ${kernel_names.length} kernels. Total duration: <strong>${total_duration.toFixed(3)} μs</strong></p>`;
            
            result_div.text = results_html;
            """
        )
        
        # GPU B calculator (similar)
        calc_callback_b = CustomJS(
            args=dict(
                source=self.ds.source_gpu_b,
                input_widget=self.kernel_input_b,
                result_div=self.result_div_b,
                gpu_name=self.gpu_name_b
            ),
            code="""
            const kernel_names = input_widget.value.split('\\n').map(name => name.trim()).filter(name => name.length > 0);
            
            if (kernel_names.length === 0) {
                result_div.text = `<h4>${gpu_name} Results:</h4><p style="color: red;">Please enter at least one kernel name</p>`;
                return;
            }
            
            const data = source.data;
            let total_duration = 0;
            let found_count = 0;
            let results_html = `<h4>${gpu_name} Results:</h4>`;
            
            // Create a map for faster lookup
            const kernel_durations = {};
            for (let i = 0; i < data['Kernel Name'].length; i++) {
                const name = data['Kernel Name'][i];
                const duration = data['Duration (us)'][i];
                if (kernel_durations[name]) {
                    kernel_durations[name] += duration;
                } else {
                    kernel_durations[name] = duration;
                }
            }
            
            results_html += '<table style="border-collapse: collapse; width: 100%;">';
            results_html += '<tr style="background-color: #f0f0f0;"><th style="border: 1px solid #ddd; padding: 8px;">Kernel Name</th><th style="border: 1px solid #ddd; padding: 8px;">Total Duration (μs)</th><th style="border: 1px solid #ddd; padding: 8px;">Status</th></tr>';
            
            for (let kernel_name of kernel_names) {
                if (kernel_durations[kernel_name]) {
                    const duration = kernel_durations[kernel_name].toFixed(3);
                    total_duration += kernel_durations[kernel_name];
                    found_count++;
                    results_html += `<tr><td style="border: 1px solid #ddd; padding: 8px;">${kernel_name}</td><td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${duration}</td><td style="border: 1px solid #ddd; padding: 8px; color: green;">Found</td></tr>`;
                } else {
                    results_html += `<tr><td style="border: 1px solid #ddd; padding: 8px;">${kernel_name}</td><td style="border: 1px solid #ddd; padding: 8px; text-align: right;">0.000</td><td style="border: 1px solid #ddd; padding: 8px; color: red;">Not Found</td></tr>`;
                }
            }
            
            results_html += `<tr style="background-color: #e6f3ff; font-weight: bold;"><td style="border: 1px solid #ddd; padding: 8px;">TOTAL</td><td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${total_duration.toFixed(3)}</td><td style="border: 1px solid #ddd; padding: 8px;">${found_count}/${kernel_names.length} found</td></tr>`;
            results_html += '</table>';
            
            results_html += `<p><strong>Summary:</strong> Found ${found_count} out of ${kernel_names.length} kernels. Total duration: <strong>${total_duration.toFixed(3)} μs</strong></p>`;
            
            result_div.text = results_html;
            """
        )
        
        # Combined calculator
        calc_callback_combined = CustomJS(
            args=dict(
                source_a=self.ds.source_gpu_a,
                source_b=self.ds.source_gpu_b,
                input_widget=self.kernel_input_combined,
                result_div=self.result_div_combined,
                gpu_name_a=self.gpu_name_a,
                gpu_name_b=self.gpu_name_b
            ),
            code="""
            const kernel_names = input_widget.value.split('\\n').map(name => name.trim()).filter(name => name.length > 0);
            
            if (kernel_names.length === 0) {
                result_div.text = '<h4>Combined Results:</h4><p style="color: red;">Please enter at least one kernel name</p>';
                return;
            }
            
            // Process GPU A
            const data_a = source_a.data;
            const kernel_durations_a = {};
            for (let i = 0; i < data_a['Kernel Name'].length; i++) {
                const name = data_a['Kernel Name'][i];
                const duration = data_a['Duration (us)'][i];
                if (kernel_durations_a[name]) {
                    kernel_durations_a[name] += duration;
                } else {
                    kernel_durations_a[name] = duration;
                }
            }
            
            // Process GPU B
            const data_b = source_b.data;
            const kernel_durations_b = {};
            for (let i = 0; i < data_b['Kernel Name'].length; i++) {
                const name = data_b['Kernel Name'][i];
                const duration = data_b['Duration (us)'][i];
                if (kernel_durations_b[name]) {
                    kernel_durations_b[name] += duration;
                } else {
                    kernel_durations_b[name] = duration;
                }
            }
            
            let total_duration_a = 0;
            let total_duration_b = 0;
            let found_count_a = 0;
            let found_count_b = 0;
            
            let results_html = '<h4>Combined Results:</h4>';
            results_html += '<table style="border-collapse: collapse; width: 100%;">';
            results_html += `<tr style="background-color: #f0f0f0;"><th style="border: 1px solid #ddd; padding: 8px;">Kernel Name</th><th style="border: 1px solid #ddd; padding: 8px;">${gpu_name_a} Duration (μs)</th><th style="border: 1px solid #ddd; padding: 8px;">${gpu_name_b} Duration (μs)</th><th style="border: 1px solid #ddd; padding: 8px;">Difference (μs)</th><th style="border: 1px solid #ddd; padding: 8px;">Ratio (A/B)</th></tr>`;
            
            for (let kernel_name of kernel_names) {
                const duration_a = kernel_durations_a[kernel_name] || 0;
                const duration_b = kernel_durations_b[kernel_name] || 0;
                const difference = duration_a - duration_b;
                const ratio = duration_b > 0 ? (duration_a / duration_b) : 'N/A';
                
                if (duration_a > 0) {
                    total_duration_a += duration_a;
                    found_count_a++;
                }
                if (duration_b > 0) {
                    total_duration_b += duration_b;
                    found_count_b++;
                }
                
                const ratio_text = ratio === 'N/A' ? 'N/A' : ratio.toFixed(3);
                const difference_color = difference > 0 ? 'color: red;' : difference < 0 ? 'color: green;' : '';
                
                results_html += `<tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">${kernel_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${duration_a.toFixed(3)}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${duration_b.toFixed(3)}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right; ${difference_color}">${difference.toFixed(3)}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${ratio_text}</td>
                </tr>`;
            }
            
            const total_difference = total_duration_a - total_duration_b;
            const total_ratio = total_duration_b > 0 ? (total_duration_a / total_duration_b).toFixed(3) : 'N/A';
            const total_difference_color = total_difference > 0 ? 'color: red;' : total_difference < 0 ? 'color: green;' : '';
            
            results_html += `<tr style="background-color: #e6f3ff; font-weight: bold;">
                <td style="border: 1px solid #ddd; padding: 8px;">TOTAL</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${total_duration_a.toFixed(3)}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${total_duration_b.toFixed(3)}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: right; ${total_difference_color}">${total_difference.toFixed(3)}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">${total_ratio}</td>
            </tr>`;
            results_html += '</table>';
            
            results_html += `<p><strong>Summary:</strong></p>`;
            results_html += `<p>${gpu_name_a}: Found ${found_count_a}/${kernel_names.length} kernels, Total: <strong>${total_duration_a.toFixed(3)} μs</strong></p>`;
            results_html += `<p>${gpu_name_b}: Found ${found_count_b}/${kernel_names.length} kernels, Total: <strong>${total_duration_b.toFixed(3)} μs</strong></p>`;
            results_html += `<p>Difference: <strong style="${total_difference_color}">${total_difference.toFixed(3)} μs</strong></p>`;
            
            result_div.text = results_html;
            """
        )
        
        # Attach callbacks
        self.calc_button_a.js_on_click(calc_callback_a)
        self.calc_button_b.js_on_click(calc_callback_b)
        self.calc_button_combined.js_on_click(calc_callback_combined)


class TableManager:
    """Class to manage data tables"""
    
    def __init__(self, data_sources, control_manager, chart_manager):
        self.ds = data_sources
        self.controls = control_manager
        self.charts = chart_manager
        self.tables = {}
        self.copy_buttons = {}
        self._create_tables()
        self._setup_table_interactions()
        self._create_copy_buttons()
    
    def _create_table_columns(self):
        """Create standard table columns"""
        return [
            TableColumn(field="Kernel Index", title="Index", width=80),
            TableColumn(field="Kernel Name", title="Kernel Name", width=600),
            TableColumn(field="Start (us)", title="Start (μs)", width=100, 
                       formatter=NumberFormatter(format="0,0.000")),
            TableColumn(field="Duration (us)", title="Duration (μs)", width=120, 
                       formatter=NumberFormatter(format="0,0.000")),
            TableColumn(field="End (us)", title="End (μs)", width=100, 
                       formatter=NumberFormatter(format="0,0.000")),
        ]
    
    def _create_top_columns(self):
        """Create top N table columns"""
        return [
            TableColumn(field="Kernel Name", title="Kernel Name", width=680),
            TableColumn(field="Total Duration (us)", title="Total Duration (μs)", width=120,
                       formatter=NumberFormatter(format="0,0.000")),
            TableColumn(field="Count", title="Count", width=60),
            TableColumn(field="Avg Duration (us)", title="Avg Duration (μs)", width=120,
                       formatter=NumberFormatter(format="0,0.000")),
        ]
    
    def _create_tables(self):
        """Create all data tables"""
        columns = self._create_table_columns()
        top_columns = self._create_top_columns()
        
        # Individual view tables
        self.tables['gpu_a'] = DataTable(source=self.ds.source_gpu_a_filtered, columns=columns, 
                                        width=2000, height=600, index_position=None, selectable=True)
        self.tables['gpu_b'] = DataTable(source=self.ds.source_gpu_b_filtered, columns=columns, 
                                        width=2000, height=600, index_position=None, selectable=True)
        
        # Sorted tables
        self.tables['sorted_gpu_a'] = DataTable(source=self.ds.source_sorted_gpu_a_filtered, columns=columns, 
                                               width=2000, height=600, index_position=None, selectable=True)
        self.tables['sorted_gpu_b'] = DataTable(source=self.ds.source_sorted_gpu_b_filtered, columns=columns, 
                                               width=2000, height=600, index_position=None, selectable=True)
        
        # Combined view tables
        self.tables['gpu_a_combined'] = DataTable(source=self.ds.source_gpu_a_combined_filtered, columns=columns, 
                                                 width=1000, height=600, index_position=None, selectable=True)
        self.tables['gpu_b_combined'] = DataTable(source=self.ds.source_gpu_b_combined_filtered, columns=columns, 
                                                 width=1000, height=600, index_position=None, selectable=True)
        
        # Combined sorted tables
        self.tables['sorted_gpu_a_combined'] = DataTable(source=self.ds.source_sorted_gpu_a_combined_filtered, columns=columns, 
                                                        width=1000, height=600, index_position=None, selectable=True)
        self.tables['sorted_gpu_b_combined'] = DataTable(source=self.ds.source_sorted_gpu_b_combined_filtered, columns=columns, 
                                                        width=1000, height=600, index_position=None, selectable=True)
        
        # Top N tables
        self.tables['top_gpu_a'] = DataTable(source=self.ds.source_top_gpu_a, columns=top_columns, 
                                           width=2000, height=600, index_position=None, selectable=True)
        self.tables['top_gpu_b'] = DataTable(source=self.ds.source_top_gpu_b, columns=top_columns, 
                                            width=2000, height=600, index_position=None, selectable=True)
        self.tables['top_both'] = DataTable(source=self.ds.source_top_both, columns=top_columns, 
                                          width=800, height=300, index_position=None, selectable=True)
        
        # Combined top N tables
        self.tables['top_gpu_a_combined'] = DataTable(source=self.ds.source_top_gpu_a, columns=top_columns, 
                                                     width=1000, height=600, index_position=None, selectable=True)
        self.tables['top_gpu_b_combined'] = DataTable(source=self.ds.source_top_gpu_b, columns=top_columns, 
                                                     width=1000, height=600, index_position=None, selectable=True)
    
    def _create_copy_js(self, columns, with_headers=True):
        """Create JavaScript for copying table data to clipboard in Excel-friendly format"""
        if columns == 'top':
            cols = ['Kernel Name', 'Total Duration (us)', 'Count', 'Avg Duration (us)']
        else:
            cols = ['Kernel Index', 'Kernel Name', 'Start (us)', 'Duration (us)', 'End (us)']
        
        header_line = '"' + '"\t"'.join(cols) + '"' if with_headers else '';
        
        return f"""
        const data = source.data;
        const length = data['{cols[0]}'].length;
        let text = '{header_line}';
        
        for (let i = 0; i < length; i++) {{
            if (text && text !== '') text += '\\n';
            const row = [];
            {chr(10).join([f'row.push(String(data["{col}"][i]));' for col in cols])}
            text += '"' + row.join('"\\t"') + '"';
        }}
        
        navigator.clipboard.writeText(text).then(function() {{
            console.log('Table data copied to clipboard');
            button.label = 'Copied!';
            setTimeout(() => {{ button.label = 'Copy Table'; }}, 2000);
        }}).catch(function(err) {{
            console.error('Could not copy data: ', err);
            button.label = 'Copy Failed';
            setTimeout(() => {{ button.label = 'Copy Table'; }}, 2000);
        }});
        """
    
    def _create_copy_buttons(self):
        """Create copy buttons for all tables"""
        # Individual view copy buttons
        self.copy_buttons['gpu_a'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['gpu_a'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_gpu_a_filtered, button=self.copy_buttons['gpu_a']),
            code=self._create_copy_js('standard')
        ))
        
        self.copy_buttons['gpu_b'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['gpu_b'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_gpu_b_filtered, button=self.copy_buttons['gpu_b']),
            code=self._create_copy_js('standard')
        ))
        
        # Sorted tables copy buttons
        self.copy_buttons['sorted_gpu_a'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['sorted_gpu_a'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_sorted_gpu_a_filtered, button=self.copy_buttons['sorted_gpu_a']),
            code=self._create_copy_js('standard')
        ))
        
        self.copy_buttons['sorted_gpu_b'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['sorted_gpu_b'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_sorted_gpu_b_filtered, button=self.copy_buttons['sorted_gpu_b']),
            code=self._create_copy_js('standard')
        ))
        
        # Top N tables copy buttons
        self.copy_buttons['top_gpu_a'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['top_gpu_a'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_top_gpu_a, button=self.copy_buttons['top_gpu_a']),
            code=self._create_copy_js('top')
        ))
        
        self.copy_buttons['top_gpu_b'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['top_gpu_b'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_top_gpu_b, button=self.copy_buttons['top_gpu_b']),
            code=self._create_copy_js('top')
        ))
        
        # Combined view copy buttons
        self.copy_buttons['gpu_a_combined'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['gpu_a_combined'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_gpu_a_combined_filtered, button=self.copy_buttons['gpu_a_combined']),
            code=self._create_copy_js('standard')
        ))
        
        self.copy_buttons['gpu_b_combined'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['gpu_b_combined'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_gpu_b_combined_filtered, button=self.copy_buttons['gpu_b_combined']),
            code=self._create_copy_js('standard')
        ))
        
        self.copy_buttons['sorted_gpu_a_combined'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['sorted_gpu_a_combined'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_sorted_gpu_a_combined_filtered, button=self.copy_buttons['sorted_gpu_a_combined']),
            code=self._create_copy_js('standard')
        ))
        
        self.copy_buttons['sorted_gpu_b_combined'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['sorted_gpu_b_combined'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_sorted_gpu_b_combined_filtered, button=self.copy_buttons['sorted_gpu_b_combined']),
            code=self._create_copy_js('standard')
        ))
        
        self.copy_buttons['top_gpu_a_combined'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['top_gpu_a_combined'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_top_gpu_a, button=self.copy_buttons['top_gpu_a_combined']),
            code=self._create_copy_js('top')
        ))
        
        self.copy_buttons['top_gpu_b_combined'] = Button(label="Copy Table", width=100, button_type="primary")
        self.copy_buttons['top_gpu_b_combined'].js_on_click(CustomJS(
            args=dict(source=self.ds.source_top_gpu_b, button=self.copy_buttons['top_gpu_b_combined']),
            code=self._create_copy_js('top')
        ))
    
    def _setup_table_interactions(self):
        """Setup table interaction callbacks"""
        # Bar click callbacks to highlight table rows
        tap_callback_gpu_a = CustomJS(
            args=dict(source=self.ds.source_gpu_a_filtered, 
                     table=self.tables['gpu_a'], 
                     sorted_table=self.tables['sorted_gpu_a']),
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
            args=dict(source=self.ds.source_gpu_b_filtered, 
                     table=self.tables['gpu_b'], 
                     sorted_table=self.tables['sorted_gpu_b']),
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
            args=dict(source=self.ds.source_gpu_a_combined_filtered, 
                     table=self.tables['gpu_a_combined'], 
                     sorted_table=self.tables['sorted_gpu_a_combined']),
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
            args=dict(source=self.ds.source_gpu_b_combined_filtered, 
                     table=self.tables['gpu_b_combined'], 
                     sorted_table=self.tables['sorted_gpu_b_combined']),
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
        self.charts.bars['gpu_a'].data_source.selected.js_on_change('indices', tap_callback_gpu_a)
        self.charts.bars['gpu_b'].data_source.selected.js_on_change('indices', tap_callback_gpu_b)
        self.charts.bars['gpu_a_combined'].data_source.selected.js_on_change('indices', tap_callback_gpu_a_combined)
        self.charts.bars['gpu_b_combined'].data_source.selected.js_on_change('indices', tap_callback_gpu_b_combined)


class DashboardBuilder:
    """Main class to build the complete dashboard"""
    
    def __init__(self, trace_path1, trace_path2, gpu_name_a="GPU_A", gpu_name_b="GPU_B"):
        self.trace_analyzer_a = TraceAnalyzer(trace_path1)
        self.trace_analyzer_b = TraceAnalyzer(trace_path2)
        self.gpu_name_a = gpu_name_a
        self.gpu_name_b = gpu_name_b
        
        # Initialize components
        self.data_sources = DataSourceManager(self.trace_analyzer_a, self.trace_analyzer_b)
        self.chart_manager = ChartManager(self.data_sources, gpu_name_a, gpu_name_b)
        self.control_manager = ControlManager(self.data_sources, self.chart_manager, gpu_name_a, gpu_name_b)
        self.table_manager = TableManager(self.data_sources, self.control_manager, self.chart_manager)
        self.duration_calculator = KernelDurationCalculator(self.data_sources, gpu_name_a, gpu_name_b)
    
    def create_layout(self):
        """Create the complete dashboard layout"""
        # Create spacer div for left margin
        spacer = Div(text="", width=50, height=10)
        
        # GPU A layout with left spacing
        gpu_a_layout = column(
            row(spacer, Div(text=f"<h2>{self.gpu_name_a} Kernel Analysis</h2>")),
            row(spacer, self.control_manager.window_size_spinner, self.control_manager.slider_gpu_a),
            row(spacer, self.chart_manager.charts['gpu_a']),
            row(spacer, Div(text="<h3>Kernel Details Table</h3>"), self.table_manager.copy_buttons['gpu_a']),
            row(spacer, self.table_manager.tables['gpu_a']),
            row(spacer, Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"), self.table_manager.copy_buttons['sorted_gpu_a']),
            row(spacer, self.table_manager.tables['sorted_gpu_a']),
            row(spacer, Div(text="<h3>Top 10 Kernels by Total Duration</h3>"), self.table_manager.copy_buttons['top_gpu_a']),
            row(spacer, self.table_manager.tables['top_gpu_a']),
            row(spacer, Div(text="<h3>Duration Calculator</h3>")),
            row(spacer, self.duration_calculator.kernel_input_a),
            row(spacer, self.duration_calculator.calc_button_a),
            row(spacer, self.duration_calculator.result_div_a)
        )
        
        # GPU B layout with left spacing
        gpu_b_layout = column(
            row(spacer, Div(text=f"<h2>{self.gpu_name_b} Kernel Analysis</h2>")),
            row(spacer, self.control_manager.window_size_spinner, self.control_manager.slider_gpu_b),
            row(spacer, self.chart_manager.charts['gpu_b']),
            row(spacer, Div(text="<h3>Kernel Details Table</h3>"), self.table_manager.copy_buttons['gpu_b']),
            row(spacer, self.table_manager.tables['gpu_b']),
            row(spacer, Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"), self.table_manager.copy_buttons['sorted_gpu_b']),
            row(spacer, self.table_manager.tables['sorted_gpu_b']),
            row(spacer, Div(text="<h3>Top 10 Kernels by Total Duration</h3>"), self.table_manager.copy_buttons['top_gpu_b']),
            row(spacer, self.table_manager.tables['top_gpu_b']),
            row(spacer, Div(text="<h3>Duration Calculator</h3>")),
            row(spacer, self.duration_calculator.kernel_input_b),
            row(spacer, self.duration_calculator.calc_button_b),
            row(spacer, self.duration_calculator.result_div_b)
        )
        
        # Combined layout with left spacing
        both_layout = column(
            row(spacer, Div(text="<h2>Side by Side Comparison</h2>")),        
            row(spacer, self.control_manager.window_size_spinner_combined),
            row(spacer, self.control_manager.slider_gpu_a_combined, self.control_manager.slider_gpu_b_combined),
            row(spacer, self.chart_manager.charts['gpu_a_combined'], self.chart_manager.charts['gpu_b_combined']),
            row(spacer, Div(text="<h3>Kernel Details Tables</h3>")),
            row(spacer,
                column(Div(text=f"<h4>{self.gpu_name_a} Kernels</h4>"), 
                       row(self.table_manager.copy_buttons['gpu_a_combined']), 
                       self.table_manager.tables['gpu_a_combined']),
                column(Div(text=f"<h4>{self.gpu_name_b} Kernels</h4>"), 
                       row(self.table_manager.copy_buttons['gpu_b_combined']), 
                       self.table_manager.tables['gpu_b_combined'])
            ),
            row(spacer, Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>")),
            row(spacer,
                column(Div(text=f"<h4>{self.gpu_name_a} Sorted by Latency</h4>"), 
                       row(self.table_manager.copy_buttons['sorted_gpu_a_combined']), 
                       self.table_manager.tables['sorted_gpu_a_combined']),
                column(Div(text=f"<h4>{self.gpu_name_b} Sorted by Latency</h4>"), 
                       row(self.table_manager.copy_buttons['sorted_gpu_b_combined']), 
                       self.table_manager.tables['sorted_gpu_b_combined'])
            ),
            row(spacer, Div(text="<h3>Top 10 Kernels Comparison</h3>")),
            row(spacer,
                column(Div(text=f"<h4>{self.gpu_name_a} Top Kernels</h4>"), 
                       row(self.table_manager.copy_buttons['top_gpu_a_combined']), 
                       self.table_manager.tables['top_gpu_a_combined']),
                column(Div(text=f"<h4>{self.gpu_name_b} Top Kernels</h4>"), 
                       row(self.table_manager.copy_buttons['top_gpu_b_combined']), 
                       self.table_manager.tables['top_gpu_b_combined'])
            ),
        )
        
        # Create tabs
        tab1 = TabPanel(child=gpu_a_layout, title=self.gpu_name_a)
        tab2 = TabPanel(child=gpu_b_layout, title=self.gpu_name_b) 
        tab3 = TabPanel(child=both_layout, title="Side-by-Side Comparison")
        
        tabs = Tabs(tabs=[tab1, tab2, tab3])
        
        # Add overall left margin to the entire dashboard
        main_spacer = Div(text="", width=30, height=10)
        
        return row(main_spacer, column(tabs))


def create_visualization(trace_path1, trace_path2, gpu_name_a="GPU_A", gpu_name_b="GPU_B"):
    """Create visualization using the new class-based approach"""
    dashboard = DashboardBuilder(trace_path1, trace_path2, gpu_name_a, gpu_name_b)
    return dashboard.create_layout()


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
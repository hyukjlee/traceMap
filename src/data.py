import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource

class TraceDataProcessor:
    """Handles loading and processing of trace data."""
    
    @staticmethod
    def extract_kernel_data(trace_path):
        """Extract kernel data from a trace file."""
        try:
            with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
                trace_data = json.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(trace_path, 'r', encoding='utf-8') as f:
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
                "TS (us)": round(start, 3),
                "Start (us)": round(start - base_time, 3),
                "Duration (us)": round(duration, 3),
                "End (us)": round(end - base_time, 3),
            } for idx, (name, start, duration, end) in enumerate(kernel_events)]
            return pd.DataFrame(parsed)
        else:
            return pd.DataFrame(columns=["Kernel Index", "Kernel Name", "TS (us)", "Start (us)", "Duration (us)", "End (us)"])

    @staticmethod
    def create_top_n_data(df, n=30):
        """Create top N kernels by total latency and counts."""
        kernel_stats = df.groupby('Kernel Name').agg({
            'Duration (us)': ['sum', 'count', 'mean']
        }).round(3)
        kernel_stats.columns = ['Total Duration (us)', 'Count', 'Avg Duration (us)']
        kernel_stats = kernel_stats.sort_values('Total Duration (us)', ascending=False).head(n)
        kernel_stats = kernel_stats.reset_index()
        return kernel_stats

    @staticmethod
    def create_sorted_latency_data(df):
        """Create sorted kernel data by latency for individual kernels."""
        sorted_df = df.sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        return sorted_df

    @staticmethod
    def _sanitize_sheet_name(name, fallback):
        """Ensure sheet names comply with Excel limitations."""
        invalid_chars = set('[]:*?/\\')
        sanitized = ''.join('_' if ch in invalid_chars else ch for ch in str(name)) or fallback
        sanitized = sanitized[:31]
        return sanitized
    
    @staticmethod
    def _load_kernel_names(unique_kernel_file):
        """Load kernel names from a text file."""
        if not unique_kernel_file:
            return []
        path = Path(unique_kernel_file)
        if not path.is_file():
            return []
        with path.open('r', encoding='utf-8') as handle:
            return [line.strip() for line in handle if line.strip()]

    @staticmethod
    def _sort_for_sheet(df):
        """Sort kernels by absolute timestamp when available."""
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        sort_column = 'TS (us)' if 'TS (us)' in df.columns else 'Start (us)'
        return df.sort_values(sort_column).reset_index(drop=True)

    @staticmethod
    def summarize_trace(df, kernel_order=None):
        """Summarize average duration per unique kernel for a single trace."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["Kernel Name", "Avg Duration (us)"])

        summary = (
            df.groupby("Kernel Name")["Duration (us)"]
            .mean()
            .round(3)
            .reset_index()
            .rename(columns={"Duration (us)": "Avg Duration (us)"})
        )

        if kernel_order:
            order = [kernel for kernel in kernel_order if kernel in summary["Kernel Name"].values]
            if order:
                ordered = summary.set_index("Kernel Name").loc[order]
                remaining = summary.set_index("Kernel Name").drop(order, errors="ignore")
                summary = pd.concat([ordered, remaining], axis=0).reset_index()
            else:
                summary = summary.sort_values("Avg Duration (us)", ascending=False).reset_index(drop=True)
        else:
            summary = summary.sort_values("Avg Duration (us)", ascending=False).reset_index(drop=True)
        return summary

    @staticmethod
    def _encode_kernel_names(names):
        """Map kernel names to integer ids."""
        mapping = {}
        encoded = []
        for name in names:
            if name not in mapping:
                mapping[name] = len(mapping) + 1
            encoded.append(mapping[name])
        return encoded, mapping

    @staticmethod
    def _select_non_overlapping(indices, block_length):
        """Select non-overlapping starting indices."""
        selected = []
        last_end = -1
        for idx in sorted(indices):
            if idx >= last_end:
                selected.append(idx)
                last_end = idx + block_length
        return selected

    @staticmethod
    def find_repeated_block(df, min_block_length=30, max_block_length=60, min_repeats=2, target_occurrences=None):
        """Find the most prominent repeated block of kernels.

        Args:
            df (pd.DataFrame): Trace data frame.
            min_block_length (int): Minimum number of kernels in a block.
            max_block_length (int): Maximum number of kernels in a block.
            min_repeats (int): Minimum number of times a block must repeat.
            target_occurrences (int | None): Expected repeat count for a block (e.g. layer count).
        """
        if df is None or df.empty:
            return None

        names = df["Kernel Name"].tolist()
        encoded, mapping = TraceDataProcessor._encode_kernel_names(names)
        n = len(encoded)

        if n < min_block_length * min_repeats:
            return None

        mod = 1 << 64
        base = 1_000_003
        best = None
        best_priority = None
        max_length = min(max_block_length, n // min_repeats)

        for length in range(max_length, min_block_length - 1, -1):
            pow_base = pow(base, length - 1, mod)
            hashes = {}

            # initial hash
            h = 0
            for i in range(length):
                h = (h * base + encoded[i]) % mod
            hashes.setdefault(h, []).append(0)

            for start in range(1, n - length + 1):
                left_val = encoded[start - 1]
                right_val = encoded[start + length - 1]
                h = (h - (left_val * pow_base) % mod) % mod
                h = (h * base + right_val) % mod
                hashes.setdefault(h, []).append(start)

            found_for_length = False
            for hash_value, idxs in hashes.items():
                if len(idxs) < min_repeats:
                    continue
                sequences = {}
                for idx in idxs:
                    seq = tuple(encoded[idx: idx + length])
                    sequences.setdefault(seq, []).append(idx)

                for seq_ids, occurrences in sequences.items():
                    non_overlapping = TraceDataProcessor._select_non_overlapping(occurrences, length)
                    if len(non_overlapping) < min_repeats:
                        continue

                    score = length * len(non_overlapping)
                    occurrence_count = len(non_overlapping)
                    if target_occurrences is not None:
                        diff = abs(occurrence_count - target_occurrences)
                        priority = (-diff, occurrence_count, length, score)
                    else:
                        priority = (score, length, occurrence_count)

                    if not best or priority > best_priority:
                        kernel_sequence = [names[idx] for idx in range(non_overlapping[0], non_overlapping[0] + length)]
                        best = {
                            "length": length,
                            "occurrences": non_overlapping,
                            "kernel_sequence": kernel_sequence,
                            "score": score,
                            "occurrence_count": occurrence_count,
                            "target_occurrences": target_occurrences,
                            "occurrence_diff": diff if target_occurrences is not None else None,
                        }
                        best_priority = priority
                        found_for_length = True

        return best

    @staticmethod
    def summarize_block(df, block_info):
        """Summarize duration statistics for a repeated block."""
        if not block_info:
            return pd.DataFrame(columns=["Position", "Kernel Name", "Avg Duration (us)", "Median Duration (us)",
                                         "Min Duration (us)", "Max Duration (us)", "Occurrences"])

        length = block_info["length"]
        starts = block_info["occurrences"]
        data = []

        for position in range(length):
            durations = [
                df.iloc[start + position]["Duration (us)"]
                for start in starts
                if start + position < len(df)
            ]
            kernel_name = block_info["kernel_sequence"][position]
            if not durations:
                continue
            stats = {
                "Position": position,
                "Kernel Name": kernel_name,
                "Avg Duration (us)": round(float(np.mean(durations)), 3),
                "Median Duration (us)": round(float(np.median(durations)), 3),
                "Min Duration (us)": round(float(np.min(durations)), 3),
                "Max Duration (us)": round(float(np.max(durations)), 3),
                "Occurrences": len(durations),
            }
            data.append(stats)

        return pd.DataFrame(data)

    @staticmethod
    def block_metadata(df, block_info, trace_name):
        """Create metadata summary for a repeated block."""
        if not block_info:
            return pd.DataFrame({"Metric": ["Status"], "Value": [f"No repeated block (length >= 10) found for {trace_name}"]})

        length = block_info["length"]
        starts = block_info["occurrences"]
        occurrence_count = block_info.get("occurrence_count", len(starts))
        block_durations = []
        for start in starts:
            slice_df = df.iloc[start: start + length]
            total = slice_df["Duration (us)"].sum()
            block_durations.append(total)

        block_durations = np.array(block_durations, dtype=float)
        std_val = float(block_durations.std(ddof=1)) if len(block_durations) > 1 else 0.0
        metrics = [
            ("Trace Name", trace_name),
            ("Block Length (kernels)", length),
            ("Occurrences", occurrence_count),
            ("First Kernel Index", starts[0]),
            ("Score", block_info["score"]),
            ("Mean Block Duration (us)", round(float(block_durations.mean()), 3)),
            ("Std Block Duration (us)", round(std_val, 3)),
            ("Min Block Duration (us)", round(float(block_durations.min()), 3)),
            ("Max Block Duration (us)", round(float(block_durations.max()), 3)),
        ]

        target_occurrences = block_info.get("target_occurrences")
        if target_occurrences is not None:
            metrics.append(("Target Occurrences", target_occurrences))
            metrics.append(("Occurrence Delta", abs(occurrence_count - target_occurrences)))

        return pd.DataFrame(metrics, columns=["Metric", "Value"])

    @staticmethod
    def export_kernel_report(
        df_gpu_a,
        df_gpu_b,
        name_a,
        name_b,
        output_path,
        unique_kernel_file=None,
        total_layers=None,
    ):
        """Export kernel summaries to a multi-sheet Excel workbook."""
        if not output_path:
            raise ValueError("output_path is required for export.")

        output_path = Path(output_path)
        if output_path.suffix.lower() == ".csv":
            output_path = output_path.with_suffix(".xlsx")

        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        sheet_name_a = TraceDataProcessor._sanitize_sheet_name(name_a, "Trace_A")
        sheet_name_b = TraceDataProcessor._sanitize_sheet_name(name_b, "Trace_B")

        kernel_order = TraceDataProcessor._load_kernel_names(unique_kernel_file)
        summary_a = TraceDataProcessor.summarize_trace(df_gpu_a, kernel_order)
        summary_b = TraceDataProcessor.summarize_trace(df_gpu_b, kernel_order)
        block_info_a = TraceDataProcessor.find_repeated_block(df_gpu_a, target_occurrences=total_layers)
        block_info_b = TraceDataProcessor.find_repeated_block(df_gpu_b, target_occurrences=total_layers)
        df_a_sorted = TraceDataProcessor._sort_for_sheet(df_gpu_a)
        df_b_sorted = TraceDataProcessor._sort_for_sheet(df_gpu_b)

        engine = None
        for candidate in ("openpyxl", "xlsxwriter"):
            try:
                __import__(candidate)
                engine = "openpyxl" if candidate == "openpyxl" else "xlsxwriter"
                break
            except ImportError:
                continue

        if engine is None:
            raise ModuleNotFoundError(
                "CSV export requires either 'openpyxl' or 'xlsxwriter'. "
                "Install one of these packages to enable --csv output."
            )

        with pd.ExcelWriter(output_path, engine=engine) as writer:
            df_a_sorted.to_excel(writer, sheet_name=sheet_name_a, index=False)
            df_b_sorted.to_excel(writer, sheet_name=sheet_name_b, index=False)

            summaries_sheet = "Summaries"
            summary_a.to_excel(writer, sheet_name=summaries_sheet, index=False, startrow=1, startcol=0)
            start_col_b = summary_a.shape[1] + 2
            summary_b.to_excel(writer, sheet_name=summaries_sheet, index=False, startrow=1, startcol=start_col_b)

            worksheet = writer.sheets[summaries_sheet]
            header_a = f"{name_a} Unique Kernels"
            header_b = f"{name_b} Unique Kernels"
            if engine == "xlsxwriter":
                worksheet.write(0, 0, header_a)
                worksheet.write(0, start_col_b, header_b)
            else:
                worksheet.cell(row=1, column=1, value=header_a)
                worksheet.cell(row=1, column=start_col_b + 1, value=header_b)

            # Side-by-side detailed traces
            combined_sheet = "Trace Comparison"
            df_a_sorted.to_excel(writer, sheet_name=combined_sheet, index=False, startrow=1, startcol=0)
            start_col_trace_b = df_a_sorted.shape[1] + 2
            df_b_sorted.to_excel(writer, sheet_name=combined_sheet, index=False, startrow=1, startcol=start_col_trace_b)

            combined_ws = writer.sheets[combined_sheet]
            combined_header_a = f"{name_a} Full Trace"
            combined_header_b = f"{name_b} Full Trace"
            if engine == "xlsxwriter":
                combined_ws.write(0, 0, combined_header_a)
                combined_ws.write(0, start_col_trace_b, combined_header_b)
            else:
                combined_ws.cell(row=1, column=1, value=combined_header_a)
                combined_ws.cell(row=1, column=start_col_trace_b + 1, value=combined_header_b)

            block_sheet = "Repeated Blocks"
            block_infos = [
                (name_a, df_gpu_a, block_info_a),
                (name_b, df_gpu_b, block_info_b),
            ]
            block_column_offset = 9
            for index, (trace_name, df_trace, block_info) in enumerate(block_infos):
                start_col = index * block_column_offset
                metadata_df = TraceDataProcessor.block_metadata(df_trace, block_info, trace_name)
                metadata_df.to_excel(writer, sheet_name=block_sheet, index=False, startrow=0, startcol=start_col)

                if block_info:
                    summary_df = TraceDataProcessor.summarize_block(df_trace, block_info)
                    summary_row_start = len(metadata_df) + 2
                    summary_df.to_excel(
                        writer,
                        sheet_name=block_sheet,
                        index=False,
                        startrow=summary_row_start,
                        startcol=start_col
                    )
                else:
                    continue

        return output_path

class DataSourceManager:
    """Manages Bokeh data sources for charts and tables."""
    
    def __init__(self, df_gpu_a, df_gpu_b, default_window_size=100):
        self.df_gpu_a = df_gpu_a
        self.df_gpu_b = df_gpu_b
        self.default_window_size = default_window_size
        self._create_all_sources()
    
    def _create_all_sources(self):
        """Create all data sources needed for the visualization."""
        # Main data sources
        self.source_gpu_a = ColumnDataSource(self.df_gpu_a)
        self.source_gpu_b = ColumnDataSource(self.df_gpu_b)
        
        # Filtered sources for sliding window
        self.source_gpu_a_filtered = ColumnDataSource(self.df_gpu_a.head(self.default_window_size))
        self.source_gpu_b_filtered = ColumnDataSource(self.df_gpu_b.head(self.default_window_size))
        
        # Sorted filtered sources
        initial_gpu_a_sorted = self.df_gpu_a.head(self.default_window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        initial_gpu_b_sorted = self.df_gpu_b.head(self.default_window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        
        self.source_sorted_gpu_a_filtered = ColumnDataSource(initial_gpu_a_sorted)
        self.source_sorted_gpu_b_filtered = ColumnDataSource(initial_gpu_b_sorted)
        
        # Combined view sources
        self.source_gpu_a_combined_filtered = ColumnDataSource(self.df_gpu_a.head(self.default_window_size))
        self.source_gpu_b_combined_filtered = ColumnDataSource(self.df_gpu_b.head(self.default_window_size))
        
        self.source_sorted_gpu_a_combined_filtered = ColumnDataSource(initial_gpu_a_sorted)
        self.source_sorted_gpu_b_combined_filtered = ColumnDataSource(initial_gpu_b_sorted)
        
        # Top N data sources
        self._create_top_n_sources()
    
    def _create_top_n_sources(self):
        """Create top N data sources."""
        top_n_gpu_a = TraceDataProcessor.create_top_n_data(self.df_gpu_a)
        top_n_gpu_b = TraceDataProcessor.create_top_n_data(self.df_gpu_b)
        top_n_both = TraceDataProcessor.create_top_n_data(pd.concat([self.df_gpu_a, self.df_gpu_b], ignore_index=True))
        
        self.source_top_gpu_a = ColumnDataSource(top_n_gpu_a)
        self.source_top_gpu_b = ColumnDataSource(top_n_gpu_b)
        self.source_top_both = ColumnDataSource(top_n_both)

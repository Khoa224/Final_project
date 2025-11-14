import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from collections import defaultdict


class Visualizer:
    def __init__(self, max_nets=None, max_edges_per_net=None, grid_resolution=None):
        """
        Initialize the Visualizer with optimized settings for millions of connections.

        Args:
            max_nets: Maximum number of nets to parse (None = unlimited)
            max_edges_per_net: Maximum number of edges per net (None = unlimited)
            grid_resolution: Grid resolution for heatmap (None = use grid from .gr file)
        """
        self.max_nets = max_nets
        self.max_edges_per_net = max_edges_per_net
        self.grid_resolution = grid_resolution
        self.df = None
        self.heatmap_data = None
        self.grid_bounds = None
        self.grid_config = None  # Will store grid info from .gr file
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def parse_gr_file_header(self, gr_file_path):
        """
        Parse the header of a .gr file to extract grid configuration.

        Args:
            gr_file_path: Path to the .gr input file

        Returns:
            Dictionary with grid configuration
        """
        if not os.path.exists(gr_file_path):
            print(f"Warning: .gr file not found: {gr_file_path}")
            return None

        try:
            with open(gr_file_path, 'r') as f:
                lines = f.readlines()

            grid_config = {}

            for line in lines[:10]:  # Grid info is in first few lines
                line = line.strip()
                if line.startswith('grid'):
                    parts = line.split()
                    if len(parts) >= 4:
                        grid_config['x_grids'] = int(parts[1])
                        grid_config['y_grids'] = int(parts[2])
                        grid_config['num_layers'] = int(parts[3])
                elif line.startswith('vertical capacity'):
                    parts = line.split()
                    grid_config['vertical_capacity'] = [int(parts[2]), int(parts[3])]
                elif line.startswith('horizontal capacity'):
                    parts = line.split()
                    grid_config['horizontal_capacity'] = [int(parts[2]), int(parts[3])]
                elif not line.startswith(('minimum', 'via', 'num')):
                    # Check if it's the boundary line (4 numbers)
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            grid_config['lower_left_x'] = int(parts[0])
                            grid_config['lower_left_y'] = int(parts[1])
                            grid_config['tile_width'] = int(parts[2])
                            grid_config['tile_height'] = int(parts[3])
                        except ValueError:
                            pass

            if 'x_grids' in grid_config and 'y_grids' in grid_config:
                print(f"Grid configuration from .gr file:")
                print(f"  Grid size: {grid_config['x_grids']} x {grid_config['y_grids']}")
                print(f"  Layers: {grid_config.get('num_layers', 'unknown')}")
                if 'tile_width' in grid_config:
                    print(f"  Lower-left: ({grid_config['lower_left_x']}, {grid_config['lower_left_y']})")
                    print(f"  Tile size: {grid_config['tile_width']} x {grid_config['tile_height']}")

                self.grid_config = grid_config

                # Set grid resolution based on .gr file if not specified
                if self.grid_resolution is None:
                    self.grid_resolution = (grid_config['x_grids'], grid_config['y_grids'])

                return grid_config
            else:
                print("Warning: Could not parse grid configuration from .gr file")
                return None

        except Exception as e:
            print(f"Error parsing .gr file: {e}")
            return None

    def parse_nthu_output(self, path):
        """
        Parse NTHU Router output file and extract routing edges.
        Optimized for large files.

        Args:
            path: Path to the routing output file

        Returns:
            DataFrame with routing edge information
        """
        edges = []
        print(f"Parsing file: {path}")

        with open(path) as f:
            lines = f.readlines()

        i = net_cnt = 0
        total_edges = 0

        while i < len(lines):
            if self.max_nets and net_cnt >= self.max_nets:
                break

            line = lines[i].strip()
            if line.startswith('n') and '!' not in line:
                parts = re.split(r'\s+', line)
                if len(parts) < 3:
                    i += 1
                    continue
                try:
                    net_id = int(parts[1])
                    num_edges = int(parts[2])
                except:
                    i += 1
                    continue

                i += 1
                edge_cnt = 0

                while i < len(lines):
                    if self.max_edges_per_net and edge_cnt >= self.max_edges_per_net:
                        # Skip remaining edges for this net
                        while i < len(lines) and lines[i].strip() != '!':
                            i += 1
                        break

                    el = lines[i].strip()
                    if el == '!':
                        break

                    m = re.match(r'\((\d+),(\d+),(\d+)\)-\((\d+),(\d+),(\d+)\)', el)
                    if m:
                        x1, y1, z1, x2, y2, z2 = map(int, m.groups())
                        layer = (z1 + z2) // 2
                        edges.append({
                            'net_id': net_id,
                            'x1': x1, 'y1': y1, 'z1': z1,
                            'x2': x2, 'y2': y2, 'z2': z2,
                            'layer': layer
                        })
                        edge_cnt += 1
                        total_edges += 1
                    i += 1

                i += 1
                net_cnt += 1

                # Progress update
                if net_cnt % 1000 == 0:
                    print(f"  Parsed {net_cnt} nets, {total_edges} edges...")
            else:
                i += 1

        print(f"Total parsed: {net_cnt} nets, {total_edges} edges")

        df = pd.DataFrame(edges)
        if df.empty:
            print("No edges parsed. Returning empty DataFrame.")
            return df

        df['layer'] = df['layer'].astype('int32')
        self.df = df
        return df

    def build_heatmap(self, df=None, resolution=None):
        """
        Build routing density heatmap for each layer.
        This is shown when zoomed out.

        Args:
            df: DataFrame to build heatmap from (uses self.df if None)
            resolution: Grid resolution (uses self.grid_resolution if None, or grid from .gr file)

        Returns:
            Dictionary of heatmaps per layer
        """
        if df is None:
            df = self.df
        if resolution is None:
            resolution = self.grid_resolution

        if df is None or df.empty:
            print("No data for heatmap")
            return {}

        # Find grid bounds from actual routing data (physical coordinates)
        x_min = min(df['x1'].min(), df['x2'].min())
        x_max = max(df['x1'].max(), df['x2'].max())
        y_min = min(df['y1'].min(), df['y2'].min())
        y_max = max(df['y1'].max(), df['y2'].max())

        print(f"Routing coordinate bounds: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")

        self.grid_bounds = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max
        }

        # Determine resolution for heatmap
        if isinstance(resolution, tuple):
            x_resolution, y_resolution = resolution
        elif resolution is not None:
            x_resolution = y_resolution = resolution
        elif self.grid_config:
            # Use actual grid resolution from .gr file
            x_resolution = self.grid_config['x_grids']
            y_resolution = self.grid_config['y_grids']
        else:
            x_resolution = y_resolution = 100

        print(f"Heatmap resolution: {x_resolution} x {y_resolution}")

        # Get capacity information from grid config
        has_capacity = False
        if self.grid_config and 'vertical_capacity' in self.grid_config and 'horizontal_capacity' in self.grid_config:
            has_capacity = True
            print(f"Using capacity info from .gr file:")
            print(f"  Vertical capacity: {self.grid_config['vertical_capacity']}")
            print(f"  Horizontal capacity: {self.grid_config['horizontal_capacity']}")

        # Create heatmap for each layer
        layers = df['layer'].unique()
        heatmaps = {}

        for layer in layers:
            layer_df = df[df['layer'] == layer]

            # Create 2D histograms for horizontal and vertical routing
            heatmap_h = np.zeros((y_resolution, x_resolution))  # Horizontal routing
            heatmap_v = np.zeros((y_resolution, x_resolution))  # Vertical routing

            # Use physical coordinate bins
            x_bins = np.linspace(x_min, x_max + 1, x_resolution + 1)
            y_bins = np.linspace(y_min, y_max + 1, y_resolution + 1)

            # Count routing segments in each bin, separating horizontal and vertical
            for _, row in layer_df.iterrows():
                # Bin both endpoints
                x1_bin = np.searchsorted(x_bins, row['x1']) - 1
                y1_bin = np.searchsorted(y_bins, row['y1']) - 1
                x2_bin = np.searchsorted(x_bins, row['x2']) - 1
                y2_bin = np.searchsorted(y_bins, row['y2']) - 1

                # Clip to valid range
                x1_bin = np.clip(x1_bin, 0, x_resolution - 1)
                y1_bin = np.clip(y1_bin, 0, y_resolution - 1)
                x2_bin = np.clip(x2_bin, 0, x_resolution - 1)
                y2_bin = np.clip(y2_bin, 0, y_resolution - 1)

                # Determine if segment is horizontal or vertical
                is_horizontal = row['y1'] == row['y2']
                is_vertical = row['x1'] == row['x2']

                if is_horizontal:
                    heatmap_h[y1_bin, x1_bin] += 1
                    heatmap_h[y2_bin, x2_bin] += 1
                elif is_vertical:
                    heatmap_v[y1_bin, x1_bin] += 1
                    heatmap_v[y2_bin, x2_bin] += 1
                else:
                    # Diagonal or multi-segment - count in both
                    heatmap_h[y1_bin, x1_bin] += 0.5
                    heatmap_h[y2_bin, x2_bin] += 0.5
                    heatmap_v[y1_bin, x1_bin] += 0.5
                    heatmap_v[y2_bin, x2_bin] += 0.5

            # Calculate utilization if capacity is available
            if has_capacity:
                # Get capacity for this layer (layer index from 0)
                layer_idx = int(layer)
                if layer_idx < len(self.grid_config['horizontal_capacity']):
                    h_cap = self.grid_config['horizontal_capacity'][layer_idx]
                    v_cap = self.grid_config['vertical_capacity'][layer_idx]
                else:
                    # Use last capacity if layer index is out of range
                    h_cap = self.grid_config['horizontal_capacity'][-1]
                    v_cap = self.grid_config['vertical_capacity'][-1]

                # Calculate utilization percentage
                if h_cap > 0:
                    util_h = (heatmap_h / h_cap) * 100
                else:
                    util_h = np.zeros_like(heatmap_h)

                if v_cap > 0:
                    util_v = (heatmap_v / v_cap) * 100
                else:
                    util_v = np.zeros_like(heatmap_v)

                # Combined utilization (max of horizontal and vertical)
                utilization = np.maximum(util_h, util_v)

                heatmaps[layer] = {
                    'data': utilization,
                    'x_bins': x_bins,
                    'y_bins': y_bins,
                    'is_utilization': True,
                    'h_capacity': h_cap,
                    'v_capacity': v_cap
                }

                print(f"  Layer {layer}: max utilization = {utilization.max():.1f}% (H_cap={h_cap}, V_cap={v_cap})")
            else:
                # No capacity info - use raw counts
                heatmap_total = heatmap_h + heatmap_v

                heatmaps[layer] = {
                    'data': heatmap_total,
                    'x_bins': x_bins,
                    'y_bins': y_bins,
                    'is_utilization': False
                }

                print(f"  Layer {layer}: max density = {heatmap_total.max()}")

        self.heatmap_data = heatmaps
        return heatmaps

    def create_adaptive_plot(self):
        """
        Create an adaptive visualization that shows:
        - Heatmap when zoomed out (overview)
        - Sampled routing lines at medium zoom
        - Detailed routing when zoomed in

        Returns:
            Plotly Figure with interactive zoom-based rendering
        """
        if self.df is None or self.df.empty:
            print("No data to visualize")
            return go.Figure().update_layout(title="No routing data")

        if self.heatmap_data is None:
            print("Building heatmap...")
            self.build_heatmap()

        # Get unique layers
        layers = sorted(self.df['layer'].unique())

        # Create subplots for each layer
        n_layers = len(layers)

        fig = make_subplots(
            rows=1, cols=n_layers,
            subplot_titles=[f'Layer M{layer}' for layer in layers],
            horizontal_spacing=0.05
        )

        # Check if we're showing utilization or raw counts
        is_utilization = self.heatmap_data[layers[0]].get('is_utilization', False)

        # Add heatmap for each layer
        for idx, layer in enumerate(layers, 1):
            if layer in self.heatmap_data:
                hm = self.heatmap_data[layer]

                if is_utilization:
                    hover_template = 'X: %{x}<br>Y: %{y}<br>Utilization: %{z:.1f}%<extra></extra>'
                    colorbar_title = "Utilization %"
                else:
                    hover_template = 'X: %{x}<br>Y: %{y}<br>Density: %{z}<extra></extra>'
                    colorbar_title = "Edge Count"

                fig.add_trace(
                    go.Heatmap(
                        z=hm['data'],
                        x=hm['x_bins'][:-1],
                        y=hm['y_bins'][:-1],
                        colorscale='RdYlGn_r',  # Red = high congestion, Green = low
                        name=f'Layer {layer}',
                        hovertemplate=hover_template,
                        showscale=(idx == 1),
                        colorbar=dict(title=colorbar_title) if idx == 1 else None,
                        zmin=0,
                        zmax=100 if is_utilization else None
                    ),
                    row=1, col=idx
                )

        title = "Global Routing Utilization Heatmap" if is_utilization else "Global Routing Density Heatmap"

        fig.update_layout(
            title=title,
            height=600,
            width=300 * n_layers,
            hovermode='closest'
        )

        # Update axes
        for idx in range(1, n_layers + 1):
            fig.update_xaxes(title_text="X (G-cell)", row=1, col=idx)
            if idx == 1:
                fig.update_yaxes(title_text="Y (G-cell)", row=1, col=idx)

        return fig

    def create_detailed_plot_for_layer(self, layer, sample_rate=10):
        """
        Create a detailed routing plot for a specific layer.

        Args:
            layer: Layer number to visualize
            sample_rate: Show every Nth edge (1 = all edges, 10 = every 10th edge)

        Returns:
            Plotly Figure
        """
        if self.df is None or self.df.empty:
            return go.Figure().update_layout(title="No data")

        layer_df = self.df[self.df['layer'] == layer]

        if layer_df.empty:
            return go.Figure().update_layout(title=f"No data for layer {layer}")

        # Sample the data
        sampled = layer_df.iloc[::sample_rate]

        print(f"Rendering layer {layer}: {len(sampled)} / {len(layer_df)} edges")

        # Build line segments
        xs, ys = [], []
        for _, row in sampled.iterrows():
            xs.extend([row['x1'], row['x2'], None])
            ys.extend([row['y1'], row['y2'], None])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=self.palette[layer % len(self.palette)], width=1),
            name=f'Layer M{layer}',
            hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Layer M{layer} Routing (Sample rate: 1/{sample_rate})",
            xaxis_title="X (G-cell)",
            yaxis_title="Y (G-cell)",
            width=900,
            height=700,
            hovermode='closest'
        )

        return fig

    def create_multi_lod_plot(self):
        """
        Create a plot with multiple LOD levels that can be switched via buttons.
        - Heatmap (overview)
        - Coarse sampling (1/100 edges)
        - Medium sampling (1/10 edges)
        - Fine sampling (1/2 edges)

        Returns:
            Plotly Figure with LOD switching buttons
        """
        if self.df is None or self.df.empty:
            return go.Figure().update_layout(title="No data")

        if self.heatmap_data is None:
            self.build_heatmap()

        layers = sorted(self.df['layer'].unique())
        fig = go.Figure()

        # Track which traces belong to which LOD level
        lod_traces = defaultdict(list)

        # Check if we're showing utilization or raw counts
        is_utilization = self.heatmap_data[layers[0]].get('is_utilization', False)

        # LOD 0: Heatmap (all layers combined)
        if self.heatmap_data:
            combined_heatmap = None
            for layer in layers:
                if layer in self.heatmap_data:
                    if combined_heatmap is None:
                        combined_heatmap = self.heatmap_data[layer]['data'].copy()
                    else:
                        combined_heatmap += self.heatmap_data[layer]['data']

            if combined_heatmap is not None:
                hm = self.heatmap_data[layers[0]]  # Use first layer's bins

                if is_utilization:
                    # For utilization, average across layers instead of sum
                    combined_heatmap = combined_heatmap / len(layers)
                    hover_template = 'X: %{x}<br>Y: %{y}<br>Avg Utilization: %{z:.1f}%<extra></extra>'
                    colorbar_title = "Utilization %"
                    zmax = 100
                else:
                    hover_template = 'X: %{x}<br>Y: %{y}<br>Total Density: %{z:.0f}<extra></extra>'
                    colorbar_title = "Edge Count"
                    zmax = None

                trace = go.Heatmap(
                    z=combined_heatmap,
                    x=hm['x_bins'][:-1],
                    y=hm['y_bins'][:-1],
                    colorscale='RdYlGn_r',  # Red = high congestion, Green = low
                    name='Routing Congestion',
                    hovertemplate=hover_template,
                    visible=True,
                    colorbar=dict(title=colorbar_title),
                    zmin=0,
                    zmax=zmax
                )
                fig.add_trace(trace)
                lod_traces[0].append(len(fig.data) - 1)

        # LOD 1-3: Sampled routing lines
        sampling_rates = [100, 10, 2]

        for lod_idx, sample_rate in enumerate(sampling_rates, 1):
            for layer in layers:
                layer_df = self.df[self.df['layer'] == layer]
                sampled = layer_df.iloc[::sample_rate]

                xs, ys = [], []
                for _, row in sampled.iterrows():
                    xs.extend([row['x1'], row['x2'], None])
                    ys.extend([row['y1'], row['y2'], None])

                trace = go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    line=dict(color=self.palette[layer % len(self.palette)], width=1),
                    name=f'M{layer}',
                    visible=False,
                    hovertemplate=f'Layer {layer}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>'
                )
                fig.add_trace(trace)
                lod_traces[lod_idx].append(len(fig.data) - 1)

        # Add grid lines overlay if grid config is available
        grid_trace_indices = []
        if self.grid_config and self.grid_bounds:
            # Calculate grid line spacing in physical coordinates
            if 'tile_width' in self.grid_config and 'tile_height' in self.grid_config:
                # Grid is defined in physical coordinates
                grid_x_spacing = self.grid_config['tile_width']
                grid_y_spacing = self.grid_config['tile_height']
                grid_x_start = self.grid_config.get('lower_left_x', 0)
                grid_y_start = self.grid_config.get('lower_left_y', 0)
                
                # Calculate grid boundaries
                grid_x_end = grid_x_start + self.grid_config['x_grids'] * grid_x_spacing
                grid_y_end = grid_y_start + self.grid_config['y_grids'] * grid_y_spacing

                # Calculate number of grid lines to show (limit to avoid too many lines)
                num_x_lines = min(50, self.grid_config['x_grids'])
                num_y_lines = min(50, self.grid_config['y_grids'])
                
                # Calculate step to show approximately that many lines
                x_step = max(1, self.grid_config['x_grids'] // num_x_lines)
                y_step = max(1, self.grid_config['y_grids'] // num_y_lines)
                
                # Add vertical grid lines (at actual grid tile boundaries)
                for i in range(0, self.grid_config['x_grids'] + 1, x_step):
                    x_coord = grid_x_start + i * grid_x_spacing
                    trace = go.Scatter(
                        x=[x_coord, x_coord],
                        y=[grid_y_start, grid_y_end],
                        mode='lines',
                        line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5),
                        showlegend=False,
                        visible=False,
                        hoverinfo='skip'
                    )
                    fig.add_trace(trace)
                    grid_trace_indices.append(len(fig.data) - 1)
                
                # Add horizontal grid lines (at actual grid tile boundaries)
                for i in range(0, self.grid_config['y_grids'] + 1, y_step):
                    y_coord = grid_y_start + i * grid_y_spacing
                    trace = go.Scatter(
                        x=[grid_x_start, grid_x_end],
                        y=[y_coord, y_coord],
                        mode='lines',
                        line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5),
                        showlegend=False,
                        visible=False,
                        hoverinfo='skip'
                    )
                    fig.add_trace(trace)
                    grid_trace_indices.append(len(fig.data) - 1)
                
                print(f"Added grid overlay: {len(grid_trace_indices)} lines")
                print(f"  Grid spacing: {grid_x_spacing} x {grid_y_spacing}")
                print(f"  Grid bounds: X=[{grid_x_start}, {grid_x_end}], Y=[{grid_y_start}, {grid_y_end}]")
            else:
                print("Warning: Grid tile size not found, cannot add grid overlay")

        # Create buttons for LOD switching
        buttons = []

        # Heatmap button
        visible = [False] * len(fig.data)
        for idx in lod_traces[0]:
            visible[idx] = True

        heatmap_title = "Global Routing Utilization Heatmap" if is_utilization else "Global Routing Density Heatmap"

        buttons.append(dict(
            label="Heatmap (Overview)",
            method="update",
            args=[{"visible": visible},
                  {"title": heatmap_title}]
        ))

        # Sampling buttons
        labels = ["Coarse (1/100)", "Medium (1/10)", "Fine (1/2)"]
        for lod_idx in range(1, 4):
            visible = [False] * len(fig.data)
            for idx in lod_traces[lod_idx]:
                visible[idx] = True
            buttons.append(dict(
                label=labels[lod_idx - 1],
                method="update",
                args=[{"visible": visible},
                      {"title": f"Global Routing - {labels[lod_idx - 1]} Detail"}]
            ))
        
        # Add button to show grid overlay (on top of coarse detail)
        if grid_trace_indices:
            visible = [False] * len(fig.data)
            # Show coarse routing
            for idx in lod_traces[1]:  # Use coarse (1/100) sampling
                visible[idx] = True
            # Show grid lines
            for idx in grid_trace_indices:
                visible[idx] = True
            buttons.append(dict(
                label="Grid + Routing",
                method="update",
                args=[{"visible": visible},
                      {"title": "Global Routing Grid Overlay"}]
            ))

        fig.update_layout(
            title=heatmap_title,
            xaxis_title="X (Physical Coordinates)",
            yaxis_title="Y (Physical Coordinates)",
            width=1000,
            height=800,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            hovermode='closest',
            legend=dict(title="Layer")
        )

        return fig

    def visualize(self, output_file, gr_file=None, output_html="routing_visualization.html",
                  show_browser=True, mode="multi_lod"):
        """
        Complete workflow: parse, build visualizations, and save.

        Args:
            output_file: Path to NTHU Router output file
            gr_file: Path to .gr input file (optional, for grid configuration)
            output_html: Path to save HTML visualization
            show_browser: Whether to open in browser
            mode: Visualization mode:
                  - "multi_lod": Multi-LOD plot with heatmap and sampling (default)
                  - "adaptive": Separate heatmap per layer
                  - "detailed": Detailed plot for first layer only

        Returns:
            Plotly Figure object or None if failed
        """
        # Check if file exists
        if not os.path.exists(output_file):
            print(f"Error: File '{output_file}' not found.")
            return None

        # Step 0: Parse grid configuration from .gr file if provided
        if gr_file:
            print(f"Parsing grid configuration from '{gr_file}'...")
            self.parse_gr_file_header(gr_file)
        else:
            # Try to infer .gr file path from output file
            # Replace _routing_output.txt with _routing_input.gr
            if '_routing_output' in output_file:
                inferred_gr = output_file.replace('_routing_output.txt', '_routing_input.gr')
                if os.path.exists(inferred_gr):
                    print(f"Found .gr file: {inferred_gr}")
                    self.parse_gr_file_header(inferred_gr)

        # Step 1: Parse the routing output
        print(f"\nParsing routing data from '{output_file}'...")
        df = self.parse_nthu_output(output_file)

        if df.empty:
            print("No routing data found.")
            return None

        print(f"Parsed {len(df)} edges from {df['net_id'].nunique()} nets.")

        # Step 2: Build heatmap
        print("\nBuilding routing density heatmap...")
        self.build_heatmap(df)

        # Step 3: Create visualization based on mode
        print(f"\nGenerating {mode} visualization...")

        if mode == "multi_lod":
            fig = self.create_multi_lod_plot()
        elif mode == "adaptive":
            fig = self.create_adaptive_plot()
        elif mode == "detailed":
            first_layer = df['layer'].iloc[0]
            fig = self.create_detailed_plot_for_layer(first_layer, sample_rate=10)
        else:
            print(f"Unknown mode: {mode}")
            return None

        # Step 4: Save and display
        if show_browser:
            fig.show()

        fig.write_html(output_html)
        print(f"\n✓ Visualization saved to '{output_html}'")
        print(f"  Total edges: {len(self.df):,}")
        print(f"  Layers: {sorted(self.df['layer'].unique())}")

        if self.grid_config:
            print(f"  Grid: {self.grid_config['x_grids']} x {self.grid_config['y_grids']}")

        return fig


if __name__ == '__main__':
    # Create visualizer instance (no limits - handles millions of edges)
    visualizer = Visualizer(max_nets=None, max_edges_per_net=None, grid_resolution=None)

    # Run complete visualization workflow
    output_file = r"D:\work\IDE\PyCharm\FinalProject\test_result\adaptec1_routing_output.txt"
    gr_file = r"D:\work\IDE\PyCharm\FinalProject\test_result\adaptec1_routing_input.gr"

    # Use multi_lod mode for best performance with large datasets
    fig = visualizer.visualize(
        output_file,
        gr_file=gr_file,  # Provide .gr file for accurate grid configuration
        output_html="routing_visualization.html",
        show_browser=True,
        mode="multi_lod"  # Switch between heatmap and detailed views
    )

    if fig is None:
        exit(1)

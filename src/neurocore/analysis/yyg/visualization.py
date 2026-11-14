from ._features import ChannelSpectrum, get_scaling_array
from .eeg_report.config import MAX_DISPLAY_FREQ
import numpy as np


def _build_3d_fft_chart(ch_a: ChannelSpectrum, ch_b: ChannelSpectrum,
                        n_seconds: int, filepath: str,
                        include_plotlyjs=True) -> str:
    """Build 3D FFT visualization chart, reusing V10 logic."""
    import colorsys
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path

    # [已停用 2026-11-04] 本地表原供 _get_scale() 使用；缩放已改由上游
    # load_and_process*() 依官方增益表写入 spectrum_scaled，此表注释保留备查：
    # BANDS_3D = [
    #     ("Eye",     0,  2,  0.40),
    #     ("Delta",   2,  4,  0.31),
    #     ("Theta",   4,  8,  0.24),
    #     ("Alpha",   8, 12,  0.21),
    #     ("LowBeta", 12, 20,  0.20),
    #     ("HighBeta",20, 30,  0.15),
    # ]
    FREQ_TICKVALS_3D = [0, 3, 7, 12, 18, 30]
    FREQ_TICKTEXT_3D = ["0", "3", "7", "12", "18", "30"]
    GAMMA_3D = 2.0
    CMAX_3D = 40
    ZMAX_3D = 50
    FREQ_INTERP = 4

    BASE_CS = [
        [0.0,   'rgb(0, 0, 80)'],
        [0.08,  'rgb(0, 0, 180)'],
        [0.16,  'rgb(0, 50, 255)'],
        [0.24,  'rgb(0, 150, 255)'],
        [0.40,  'rgb(0, 220, 220)'],
        [0.6,   'rgb(150, 220, 0)'],
        [0.7,   'rgb(235, 235, 50)'],
        [1.0,   'rgb(255, 0, 255)'],
    ]

    def _parse_rgb(s):
        inner = s.replace('rgb(', '').replace(')', '')
        return tuple(int(p.strip()) for p in inner.split(','))

    def _rgb_str(r, g, b):
        r, g, b = [int(max(0, min(255, round(x)))) for x in (r, g, b)]
        return f'rgb({r},{g},{b})'

    def _apply_gamma(colorscale, gamma):
        result = []
        for pos, color in colorscale:
            result.append([pos ** gamma, color])
        return result

    colorscale_3d = _apply_gamma(BASE_CS, GAMMA_3D)

    def _color_for_height(h, cmax, cs):
        if h <= 0:
            return cs[0][1]
        if h >= cmax:
            return cs[-1][1]
        t = h / cmax
        for i in range(len(cs) - 1):
            t0, c0 = cs[i]
            t1, c1 = cs[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r0, g0, b0 = _parse_rgb(c0)
                r1, g1, b1 = _parse_rgb(c1)
                return _rgb_str(r0 + frac * (r1 - r0),
                                g0 + frac * (g1 - g0),
                                b0 + frac * (b1 - b0))
        return cs[-1][1]

    # [已停用 2026-11-04] 见上方 BANDS_3D 说明，注释保留备查：
    # def _get_scale(freq):
    #     scale = np.zeros_like(freq, dtype=float)
    #     for name, fmin, fmax, factor in BANDS_3D:
    #         scale[(freq >= fmin) & (freq < fmax)] = factor
    #     scale[freq >= MAX_DISPLAY_FREQ] = 0.0
    #     return scale

    def _interp_freq(spectrum, freq_axis, factor):
        n_frames, n_freqs = spectrum.shape
        new_n = n_freqs * factor - (factor - 1)
        freq_interp = np.linspace(freq_axis[0], freq_axis[-1], new_n)
        spec_interp = np.zeros((n_frames, new_n))
        for i in range(n_freqs - 1):
            lo = i * factor
            for k in range(factor + 1):
                col = lo + k
                if col >= new_n:
                    break
                frac = k / factor
                spec_interp[:, col] = (spectrum[:, i] * (1 - frac) +
                                        spectrum[:, i + 1] * frac)
        return spec_interp, freq_interp

    def _build_walls(z_data, freq_axis, time_axis, cmax, cs):
        x_list, y_list, z_list = [], [], []
        i_list, j_list, k_list = [], [], []
        face_colors = []
        vo = 0
        n_t, n_f = z_data.shape

        def add_quad(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, fc):
            nonlocal vo
            x_list.extend([x0, x1, x2, x3])
            y_list.extend([y0, y1, y2, y3])
            z_list.extend([z0, z1, z2, z3])
            i_list.extend([vo, vo])
            j_list.extend([vo + 1, vo + 2])
            k_list.extend([vo + 2, vo + 3])
            face_colors.extend([fc, fc])
            vo += 4

        for fi in range(n_f - 1):
            fc = _color_for_height(max(z_data[0, fi], z_data[0, fi + 1]), cmax, cs)
            add_quad(freq_axis[fi], time_axis[0], 0,
                     freq_axis[fi+1], time_axis[0], 0,
                     freq_axis[fi+1], time_axis[0], z_data[0, fi + 1],
                     freq_axis[fi], time_axis[0], z_data[0, fi], fc)
        for fi in range(n_f - 1):
            fc = _color_for_height(max(z_data[-1, fi], z_data[-1, fi + 1]), cmax, cs)
            add_quad(freq_axis[fi], time_axis[-1], 0,
                     freq_axis[fi+1], time_axis[-1], 0,
                     freq_axis[fi+1], time_axis[-1], z_data[-1, fi + 1],
                     freq_axis[fi], time_axis[-1], z_data[-1, fi], fc)
        for ti in range(n_t - 1):
            fc = _color_for_height(max(z_data[ti, 0], z_data[ti + 1, 0]), cmax, cs)
            add_quad(freq_axis[0], time_axis[ti], 0,
                     freq_axis[0], time_axis[ti+1], 0,
                     freq_axis[0], time_axis[ti+1], z_data[ti + 1, 0],
                     freq_axis[0], time_axis[ti], z_data[ti, 0], fc)
        for ti in range(n_t - 1):
            fc = _color_for_height(max(z_data[ti, -1], z_data[ti + 1, -1]), cmax, cs)
            add_quad(freq_axis[-1], time_axis[ti], 0,
                     freq_axis[-1], time_axis[ti+1], 0,
                     freq_axis[-1], time_axis[ti+1], z_data[ti + 1, -1],
                     freq_axis[-1], time_axis[ti], z_data[ti, -1], fc)
        floor_c = cs[0][1]
        add_quad(freq_axis[0], time_axis[0], 0,
                 freq_axis[-1], time_axis[0], 0,
                 freq_axis[-1], time_axis[-1], 0,
                 freq_axis[0], time_axis[-1], 0, floor_c)
        return go.Mesh3d(
            x=x_list, y=y_list, z=z_list,
            i=i_list, j=j_list, k=k_list,
            facecolor=face_colors,
            opacity=1.0, hoverinfo='skip', showscale=False,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1, roughness=0.9),
        )

    freq = ch_a.freq_axis
    # [已停用 2026-11-04] scale/A_orig/B_orig 为历史死代码，注释保留备查：
    # scale = _get_scale(freq)
    A_scaled = ch_a.spectrum_scaled
    B_scaled = ch_b.spectrum_scaled
    # A_orig = ch_a.spectrum
    # B_orig = ch_b.spectrum

    A_surf, freq_surf = _interp_freq(A_scaled, freq, FREQ_INTERP)
    B_surf, _ = _interp_freq(B_scaled, freq, FREQ_INTERP)
    A_wall, B_wall = A_scaled, B_scaled
    freq_wall = freq

    time_axis = np.arange(1, n_seconds + 1)
    filename = Path(filepath).name

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'surface'}], [{'type': 'surface'}]],
        subplot_titles=("<b>左前额叶</b>", "<b>右前额叶</b>"),
        vertical_spacing=0.08,
    )

    walls_a = _build_walls(A_wall, freq_wall, time_axis, CMAX_3D, colorscale_3d)
    fig.add_trace(walls_a, row=1, col=1)
    surf_a = go.Surface(
        z=A_surf, x=freq_surf, y=time_axis,
        colorscale=colorscale_3d, cmin=0, cmax=CMAX_3D,
        showscale=True,
        colorbar=dict(title="mV", title_side="top", orientation="h", x=0.5, y=-0.08, len=0.5, thickness=12),
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.15, roughness=0.85),
        lightposition=dict(x=100, y=-100, z=200),
        name="左前额叶",
                hovertemplate="<b>左前额叶</b><br>频率: %{x:.1f} Hz<br>时间: %{y:.0f} s<br>波幅: %{z:.2f} mV<extra></extra>",
    )
    fig.add_trace(surf_a, row=1, col=1)

    walls_b = _build_walls(B_wall, freq_wall, time_axis, CMAX_3D, colorscale_3d)
    fig.add_trace(walls_b, row=2, col=1)
    surf_b = go.Surface(
        z=B_surf, x=freq_surf, y=time_axis,
        colorscale=colorscale_3d, cmin=0, cmax=CMAX_3D,
        showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.15, roughness=0.85),
        lightposition=dict(x=100, y=-100, z=200),
        name="右前额叶",
                hovertemplate="<b>右前额叶</b><br>频率: %{x:.1f} Hz<br>时间: %{y:.0f} s<br>波幅: %{z:.2f} mV<extra></extra>",
    )
    fig.add_trace(surf_b, row=2, col=1)

    camera = dict(
        eye=dict(x=2.087, y=-1.205, z=1.033),
        center=dict(x=-0.07, y=-0.282, z=-0.257),
        up=dict(x=-0.443, y=0.19, z=0.876),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="频率 (Hz)", range=[0, MAX_DISPLAY_FREQ],
                       tickvals=FREQ_TICKVALS_3D, ticktext=FREQ_TICKTEXT_3D,
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            yaxis=dict(title="时间 (s)", autorange="reversed",
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            zaxis=dict(title="波幅 (mV)", range=[0, ZMAX_3D], dtick=10,
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            camera=camera,
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.2, z=0.9),
            dragmode="turntable",
        ),
        scene2=dict(
            xaxis=dict(title="频率 (Hz)", range=[0, MAX_DISPLAY_FREQ],
                       tickvals=FREQ_TICKVALS_3D, ticktext=FREQ_TICKTEXT_3D,
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            yaxis=dict(title="时间 (s)", autorange="reversed",
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            zaxis=dict(title="波幅 (mV)", range=[0, ZMAX_3D], dtick=10,
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            camera=camera,
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.2, z=0.9),
            dragmode="turntable",
        ),
        autosize=True, height=1000,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Arial, sans-serif", size=11),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

def _build_3d_fft_chart_single(ch: ChannelSpectrum, n_seconds: int,
                                filepath: str, show_colorbar: bool = True,
                                include_plotlyjs: bool = False,
                                traditional: bool = False,
                                config: dict = None,
                                height: int = 370) -> str:
    """Build a single 3D surface chart for one channel.

    traditional=True 时坐标轴标题与悬停提示改用繁体中文（前后测等
    繁体报告用）；默认 False 保持简体，单次报告不受影响。
    config 透传 plotly to_html（plotly 默认注入 responsive=True；前后测
    报告传 {"responsive": False}，见 eeg_pre_post 列印版面註解）。
    """
    import plotly.graph_objects as go

    # [已停用 2026-11-04] 本函数从未使用此表；缩放已由上游依官方增益表
    # 写入 spectrum_scaled，注释保留备查：
    # BANDS_3D = [
    #     ("Eye",     0,  2,  0.40),
    #     ("Delta",   2,  4,  0.31),
    #     ("Theta",   4,  8,  0.24),
    #     ("Alpha",   8, 12,  0.21),
    #     ("LowBeta", 12, 20,  0.20),
    #     ("HighBeta",20, 30,  0.15),
    # ]
    FREQ_TICKVALS_3D = [0, 3, 7, 12, 18, 30]
    FREQ_TICKTEXT_3D = ["0", "3", "7", "12", "18", "30"]
    GAMMA_3D = 2.0
    CMAX_3D = 40
    ZMAX_3D = 50
    FREQ_INTERP = 4

    BASE_CS = [
        [0.0,   'rgb(0, 0, 80)'],
        [0.08,  'rgb(0, 0, 180)'],
        [0.16,  'rgb(0, 50, 255)'],
        [0.24,  'rgb(0, 150, 255)'],
        [0.40,  'rgb(0, 220, 220)'],
        [0.6,   'rgb(150, 220, 0)'],
        [0.7,   'rgb(235, 235, 50)'],
        [1.0,   'rgb(255, 0, 255)'],
    ]

    def _parse_rgb(s):
        inner = s.replace('rgb(', '').replace(')', '')
        return tuple(int(p.strip()) for p in inner.split(','))

    def _rgb_str(r, g, b):
        r, g, b = [int(max(0, min(255, round(x)))) for x in (r, g, b)]
        return f'rgb({r},{g},{b})'

    def _apply_gamma(colorscale, gamma):
        result = []
        for pos, color in colorscale:
            result.append([pos ** gamma, color])
        return result

    colorscale_3d = _apply_gamma(BASE_CS, GAMMA_3D)

    def _color_for_height(h, cmax, cs):
        if h <= 0:
            return cs[0][1]
        if h >= cmax:
            return cs[-1][1]
        t = h / cmax
        for i in range(len(cs) - 1):
            t0, c0 = cs[i]
            t1, c1 = cs[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r0, g0, b0 = _parse_rgb(c0)
                r1, g1, b1 = _parse_rgb(c1)
                return _rgb_str(r0 + frac * (r1 - r0),
                                g0 + frac * (g1 - g0),
                                b0 + frac * (b1 - b0))
        return cs[-1][1]

    def _interp_freq(spectrum, freq_axis, factor):
        n_frames, n_freqs = spectrum.shape
        new_n = n_freqs * factor - (factor - 1)
        freq_interp = np.linspace(freq_axis[0], freq_axis[-1], new_n)
        spec_interp = np.zeros((n_frames, new_n))
        for i in range(n_freqs - 1):
            lo = i * factor
            for k in range(factor + 1):
                col = lo + k
                if col >= new_n:
                    break
                frac = k / factor
                spec_interp[:, col] = (spectrum[:, i] * (1 - frac) +
                                        spectrum[:, i + 1] * frac)
        return spec_interp, freq_interp

    def _build_walls(z_data, freq_axis, time_axis, cmax, cs):
        x_list, y_list, z_list = [], [], []
        i_list, j_list, k_list = [], [], []
        face_colors = []
        vo = 0
        n_t, n_f = z_data.shape

        def add_quad(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, fc):
            nonlocal vo
            x_list.extend([x0, x1, x2, x3])
            y_list.extend([y0, y1, y2, y3])
            z_list.extend([z0, z1, z2, z3])
            i_list.extend([vo, vo])
            j_list.extend([vo + 1, vo + 2])
            k_list.extend([vo + 2, vo + 3])
            face_colors.extend([fc, fc])
            vo += 4

        for fi in range(n_f - 1):
            fc = _color_for_height(max(z_data[0, fi], z_data[0, fi + 1]), cmax, cs)
            add_quad(freq_axis[fi], time_axis[0], 0,
                     freq_axis[fi+1], time_axis[0], 0,
                     freq_axis[fi+1], time_axis[0], z_data[0, fi + 1],
                     freq_axis[fi], time_axis[0], z_data[0, fi], fc)
        for fi in range(n_f - 1):
            fc = _color_for_height(max(z_data[-1, fi], z_data[-1, fi + 1]), cmax, cs)
            add_quad(freq_axis[fi], time_axis[-1], 0,
                     freq_axis[fi+1], time_axis[-1], 0,
                     freq_axis[fi+1], time_axis[-1], z_data[-1, fi + 1],
                     freq_axis[fi], time_axis[-1], z_data[-1, fi], fc)
        for ti in range(n_t - 1):
            fc = _color_for_height(max(z_data[ti, 0], z_data[ti + 1, 0]), cmax, cs)
            add_quad(freq_axis[0], time_axis[ti], 0,
                     freq_axis[0], time_axis[ti+1], 0,
                     freq_axis[0], time_axis[ti+1], z_data[ti + 1, 0],
                     freq_axis[0], time_axis[ti], z_data[ti, 0], fc)
        for ti in range(n_t - 1):
            fc = _color_for_height(max(z_data[ti, -1], z_data[ti + 1, -1]), cmax, cs)
            add_quad(freq_axis[-1], time_axis[ti], 0,
                     freq_axis[-1], time_axis[ti+1], 0,
                     freq_axis[-1], time_axis[ti+1], z_data[ti + 1, -1],
                     freq_axis[-1], time_axis[ti], z_data[ti, -1], fc)
        floor_c = cs[0][1]
        add_quad(freq_axis[0], time_axis[0], 0,
                 freq_axis[-1], time_axis[0], 0,
                 freq_axis[-1], time_axis[-1], 0,
                 freq_axis[0], time_axis[-1], 0, floor_c)
        return go.Mesh3d(
            x=x_list, y=y_list, z=z_list,
            i=i_list, j=j_list, k=k_list,
            facecolor=face_colors,
            opacity=1.0, hoverinfo='skip', showscale=False,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1, roughness=0.9),
        )

    freq = ch.freq_axis
    scaled = ch.spectrum_scaled
    surf_data, freq_surf = _interp_freq(scaled, freq, FREQ_INTERP)
    wall_data = scaled
    freq_wall = freq
    time_axis = np.arange(1, n_seconds + 1)

    fig = go.Figure()
    walls = _build_walls(wall_data, freq_wall, time_axis, CMAX_3D, colorscale_3d)
    fig.add_trace(walls)
    surf = go.Surface(
        z=surf_data, x=freq_surf, y=time_axis,
        colorscale=colorscale_3d, cmin=0, cmax=CMAX_3D,
        showscale=show_colorbar,
        colorbar=dict(title="mV", title_side="top", orientation="h", x=0.5, y=-0.08, len=0.5, thickness=12) if show_colorbar else None,
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.15, roughness=0.85),
        lightposition=dict(x=100, y=-100, z=200),
        hovertemplate=("頻率: %{x:.1f} Hz<br>時間: %{y:.0f} s<br>波幅: %{z:.2f} mV<extra></extra>"
                       if traditional else
                       "频率: %{x:.1f} Hz<br>时间: %{y:.0f} s<br>波幅: %{z:.2f} mV<extra></extra>"),
    )
    fig.add_trace(surf)

    camera = dict(
        eye=dict(x=2.087, y=-1.205, z=1.033),
        center=dict(x=-0.07, y=-0.282, z=-0.257),
        up=dict(x=-0.443, y=0.19, z=0.876),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="頻率 (Hz)" if traditional else "频率 (Hz)", range=[0, MAX_DISPLAY_FREQ],
                       tickvals=FREQ_TICKVALS_3D, ticktext=FREQ_TICKTEXT_3D,
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            yaxis=dict(title="時間 (s)" if traditional else "时间 (s)", autorange="reversed",
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            zaxis=dict(title="波幅 (mV)", range=[0, ZMAX_3D], dtick=10,
                       backgroundcolor="white", gridcolor="rgb(200,200,200)",
                       showbackground=True),
            camera=camera,
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.2, z=0.9),
            dragmode="turntable",
        ),
        autosize=True, height=height,
        margin=dict(l=20, r=20, t=10, b=20),
        font=dict(family="Arial, sans-serif", size=11),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs,
                       config=config)

build_3d_fft_chart = _build_3d_fft_chart
build_3d_fft_chart_single = _build_3d_fft_chart_single

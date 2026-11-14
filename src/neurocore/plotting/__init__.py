"""Optional Plotly rendering; importing neurocore does not require Plotly."""
import numpy as np
from ..data import SpectrumGrid


def surface_figure(grid, max_frequency=30, interpolation=4, title=None):
    """Render an explicit spectrum profile, without changing its amplitude.

    Styling is a portable renderer, not a pixel-perfect official UI replica.
    Frequency interpolation affects geometry only, not spectral resolution.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('Install neurocore-eeg[viz] to render figures') from None
    if not isinstance(grid, SpectrumGrid):
        raise TypeError('Expected SpectrumGrid with explicit profile metadata')
    if type(interpolation) is not int or interpolation < 1:
        raise ValueError('interpolation must be a positive integer')
    z, f, t = np.asarray(grid.amplitude), np.asarray(grid.frequencies), np.asarray(grid.times)
    if z.ndim != 2 or z.shape != (len(t), len(f)) or len(t) == 0 or len(f) < 2:
        raise ValueError('Invalid grid dimensions')
    if not np.isfinite(z).all() or not np.isfinite(f).all() or not np.isfinite(t).all() or (np.diff(f) <= 0).any():
        raise ValueError('Grid must be finite and frequencies strictly increasing')
    mask = f <= max_frequency
    f, z = f[mask], z[:, mask]
    if len(f) < 2:
        raise ValueError('Frequency range contains fewer than two bins')
    fine = np.linspace(f[0], f[-1], (len(f)-1)*interpolation+1)
    dense = np.array([np.interp(fine, f, row) for row in z])
    # A visible first-frame tile is display geometry, not a second acquired frame.
    if len(t) == 1:
        dense = np.repeat(dense, 2, axis=0)
        t = np.array([t[0]-1, t[0]])
    fig = go.Figure(go.Surface(x=fine, y=t, z=dense, colorscale='Turbo',
                              colorbar=dict(title='Display units')))
    fig.update_layout(title=title or grid.profile,
                      meta=dict(profile=grid.profile, evidence=grid.evidence, units=grid.units),
                      scene=dict(xaxis_title='Frequency (Hz)', yaxis_title='Time (s)',
                                 zaxis_title=grid.units, yaxis=dict(range=[float(t[-1]), 0])))
    return fig


def write_html(grid, path, **kwargs):
    """Write a standalone interactive HTML with bundled Plotly (no CDN required)."""
    surface_figure(grid, **kwargs).write_html(str(path), include_plotlyjs=True, full_html=True)

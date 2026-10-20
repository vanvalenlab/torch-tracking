import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from scipy import ndimage

class PositionAnimationBuilder:
    """Build a scatter-point animation incrementally by adding frames during iteration.

    Parameters
    ----------
    marker_size : int, optional
        Size of the scatter markers. Default is 50.
    title : str, optional
        Plot title. Default is ``"Position Animation"``.
    xlim : tuple, optional
        ``(min, max)`` for the x-axis. Auto-determined from the added
        frames if None.
    ylim : tuple, optional
        ``(min, max)`` for the y-axis. Auto-determined from the added
        frames if None.
    figsize : tuple, optional
        Figure size ``(width, height)``. Default is ``(8, 6)``.

    Returns
    -------
    PositionAnimationBuilder
        An initialized ``PositionAnimationBuilder`` object.
    """

    def __init__(self, marker_size=50, title="Position Animation",
                 xlim=None, ylim=None, figsize=(8, 6)):
        self.frames = []  # Store (x_positions, y_positions) for each frame
        self.images = []
        self.marker_size = marker_size
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
    
    def add_frame(self, x_positions, y_positions, image=None):
        """Add a new frame of point positions to the animation.

        Points where ``x_positions`` equals 0 are treated as invalid and
        dropped before storing.

        Parameters
        ----------
        x_positions : array-like
            X coordinates for this timestep.
        y_positions : array-like
            Y coordinates for this timestep.
        image : array-like, optional
            Optional per-point image stack, indexed the same way as
            ``x_positions``/``y_positions``, stored alongside the frame.

        Returns
        -------
        None
            Appends the filtered positions (and optional images) to
            ``self.frames`` in place.
        """
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        valid = (x_positions!=0)

        x_positions = x_positions[valid]
        y_positions = y_positions[valid]
        self.frames.append((x_positions.copy(), y_positions.copy()))

        if image is not None:
            self.images.append(np.array(image[valid, :, :].squeeze()))
        
    
    def show(self, interval=50):
        """Generate and display the animation.

        Parameters
        ----------
        interval : int, optional
            Delay between frames in milliseconds. Default is 50.

        Returns
        -------
        IPython.display.HTML or None
            An HTML5-video embedding of the animation, or None if no
            frames have been added.
        """
        if not self.frames:
            print("No frames to animate!")
            return None
        
        # Determine axis limits if not provided
        if self.xlim is None:
            all_x = np.concatenate([frame[0] for frame in self.frames])
            x_min, x_max = all_x.min(), all_x.max()
            padding = 0.1 * (x_max - x_min) if x_max != x_min else 1
            xlim = (x_min - padding, x_max + padding)
        else:
            xlim = self.xlim
        
        if self.ylim is None:
            all_y = np.concatenate([frame[1] for frame in self.frames])
            y_min, y_max = all_y.min(), all_y.max()
            padding = 0.1 * (y_max - y_min) if y_max != y_min else 1
            ylim = (y_min - padding, y_max + padding)
        else:
            ylim = self.ylim
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(self.title)
        
        # Initialize scatter plot
        scatter = ax.scatter([], [], s=self.marker_size, c='k')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return scatter, time_text
        
        def update(frame_idx):

            x_pos, y_pos = self.frames[frame_idx]
            positions = np.column_stack((x_pos, y_pos))
            scatter.set_offsets(positions)
            time_text.set_text(f'Frame: {frame_idx} ({len(x_pos)} points)')
            return scatter, time_text
        
        # Create animation
        anim = FuncAnimation(fig, update, init_func=init,
                            frames=len(self.frames), interval=interval,
                            blit=True, repeat=True)
        
        plt.close()  # Prevent static plot from displaying
        
        # Return as HTML5 video for inline display
        return HTML(anim.to_html5_video())

class CellAnimationBuilder:
    """Build a cell-position animation incrementally by adding frames during iteration.

    Parameters
    ----------
    marker_size : int, optional
        Size of the scatter markers. Default is 50.
    title : str, optional
        Plot title. Default is ``"Position Animation"``.
    xlim : tuple, optional
        ``(min, max)`` for the x-axis. Auto-determined from the added
        frames if None.
    ylim : tuple, optional
        ``(min, max)`` for the y-axis. Auto-determined from the added
        frames if None.
    figsize : tuple, optional
        Figure size ``(width, height)``. Default is ``(8, 6)``.

    Returns
    -------
    CellAnimationBuilder
        An initialized ``CellAnimationBuilder`` object.
    """

    def __init__(self, marker_size=50, title="Position Animation",
                 xlim=None, ylim=None, figsize=(8, 6)):
        self.frames = []  # Store (x_positions, y_positions) for each frame
        self.images = []
        self.marker_size = marker_size
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
    
    def add_frame(self, x_positions, y_positions, image=None):
        """Add a new frame of point positions to the animation.

        Points where ``x_positions`` equals 0 are treated as invalid and
        dropped before storing.

        Parameters
        ----------
        x_positions : array-like
            X coordinates for this timestep.
        y_positions : array-like
            Y coordinates for this timestep.
        image : array-like, optional
            Optional per-point image stack, indexed the same way as
            ``x_positions``/``y_positions``, stored alongside the frame.

        Returns
        -------
        None
            Appends the filtered positions (and optional images) to
            ``self.frames`` in place.
        """
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        valid = (x_positions!=0)

        x_positions = x_positions[valid]
        y_positions = y_positions[valid]
        self.frames.append((x_positions.copy(), y_positions.copy()))

        if image is not None:
            self.images.append(np.array(image[valid, :, :].squeeze()))
        
    
    def show(self, interval=50):
        """Generate and display the animation.

        Parameters
        ----------
        interval : int, optional
            Delay between frames in milliseconds. Default is 50.

        Returns
        -------
        IPython.display.HTML or None
            An HTML5-video embedding of the animation, or None if no
            frames have been added.
        """
        if not self.frames:
            print("No frames to animate!")
            return None
        
        # Determine axis limits if not provided
        if self.xlim is None:
            all_x = np.concatenate([frame[0] for frame in self.frames])
            x_min, x_max = all_x.min(), all_x.max()
            padding = 0.1 * (x_max - x_min) if x_max != x_min else 1
            xlim = (x_min - padding, x_max + padding)
        else:
            xlim = self.xlim
        
        if self.ylim is None:
            all_y = np.concatenate([frame[1] for frame in self.frames])
            y_min, y_max = all_y.min(), all_y.max()
            padding = 0.1 * (y_max - y_min) if y_max != y_min else 1
            ylim = (y_min - padding, y_max + padding)
        else:
            ylim = self.ylim
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(self.title)
        
        # Initialize scatter plot
        scatter = ax.scatter([], [], s=self.marker_size, c='k')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return scatter, time_text
        
        def update(frame_idx):

            x_pos, y_pos = self.frames[frame_idx]
            positions = np.column_stack((x_pos, y_pos))
            scatter.set_offsets(positions)
            time_text.set_text(f'Frame: {frame_idx} ({len(x_pos)} points)')
            return scatter, time_text
        
        # Create animation
        anim = FuncAnimation(fig, update, init_func=init,
                            frames=len(self.frames), interval=interval,
                            blit=True, repeat=True)
        
        plt.close()  # Prevent static plot from displaying
        
        # Return as HTML5 video for inline display
        return HTML(anim.to_html5_video())
    


def get_cell_centroids(label_frame):
    """Compute the (row, col) centroid for each labeled cell in a 2D label image.

    A label image has integer pixel values: 0 is background, 1, 2, 3, ...
    are cell IDs. ``scipy.ndimage.center_of_mass`` computes the weighted
    average position of all pixels belonging to each label.

    Parameters
    ----------
    label_frame : ndarray
        Label image of shape ``(H, W)`` or ``(H, W, C)``. If 3D, only the
        first channel is used.

    Returns
    -------
    dict
        Mapping of ``{cell_label: (row, col)}`` for each non-background
        label found in ``label_frame``. Empty if no labeled cells are
        present.
    """
    # label_frame may be (H, W, C) — take the first channel which holds labels
    if label_frame.ndim == 3:
        lf = label_frame[..., 0]
    else:
        lf = label_frame
    
    cell_ids = np.unique(lf)
    cell_ids = cell_ids[cell_ids != 0]  # remove background
    
    if len(cell_ids) == 0:
        return {}
    
    # center_of_mass returns (row, col) for each label
    centroids_list = ndimage.center_of_mass(np.ones_like(lf), lf, cell_ids)
    return {int(cid): centroid for cid, centroid in zip(cell_ids, centroids_list)}


def classify_cells_in_frame(frame_idx, lineage):
    """Classify every cell present in a frame as dividing, a daughter, or normal.

    A cell is ``'dividing'`` if its last tracked frame is ``frame_idx`` and
    it has daughters (i.e. it divides in the next frame). A cell is
    ``'daughter'`` if its first tracked frame is ``frame_idx`` and it has a
    parent (i.e. it was just born). All other cells present in the frame
    are ``'normal'``.

    Parameters
    ----------
    frame_idx : int
        Index of the frame to classify cells in.
    lineage : dict
        Lineage dict keyed by cell label, where each value is a dict with
        ``'frames'`` (list of int), ``'daughters'`` (list of int), and
        ``'parent'`` (int or None).

    Returns
    -------
    dict
        Mapping of ``{cell_label: 'dividing' | 'daughter' | 'normal'}`` for
        every cell present in ``frame_idx``.
    """
    classifications = {}
    
    for cell_label, info in lineage.items():
        cell_label = int(cell_label)
        frames = info.get('frames', [])
        daughters = info.get('daughters', [])
        parent = info.get('parent', None)
        
        if not frames:
            continue
        
        if frame_idx not in frames:
            continue  # cell not present in this frame
        
        # Is this cell dividing? Its last frame is the current frame and it has daughters.
        if frames[-1] == frame_idx and len(daughters) > 0:
            classifications[cell_label] = 'dividing'
        # Is this a newly born daughter? Its first frame is the current frame and it has a parent.
        elif frames[0] == frame_idx and parent is not None:
            classifications[cell_label] = 'daughter'
        else:
            classifications[cell_label] = 'normal'
    
    return classifications


def build_division_arrows(frame_idx, lineage, centroids_curr, centroids_next):
    """Build mother-to-daughter centroid pairs for drawing division arrows.

    Arrows point from the mother cell's position in ``frame_idx`` to each
    daughter cell's position in ``frame_idx + 1``, visually showing the
    split happening between frames.

    Parameters
    ----------
    frame_idx : int
        Index of the frame in which the division occurs.
    lineage : dict
        Lineage dict keyed by cell label, where each value is a dict with
        ``'frames'`` (list of int) and ``'daughters'`` (list of int).
    centroids_curr : dict
        Mapping of ``{cell_label: (row, col)}`` for ``frame_idx``.
    centroids_next : dict
        Mapping of ``{cell_label: (row, col)}`` for ``frame_idx + 1``.

    Returns
    -------
    list of tuple
        A list of ``(mother_centroid, daughter_centroid)`` pairs, where
        each centroid is a ``(row, col)`` tuple.
    """
    arrows = []
    
    for cell_label, info in lineage.items():
        cell_label = int(cell_label)
        frames = info.get('frames', [])
        daughters = info.get('daughters', [])
        
        # This cell divides at the end of frame_idx
        if frames and frames[-1] == frame_idx and len(daughters) > 0:
            mother_pos = centroids_curr.get(cell_label)
            if mother_pos is None:
                continue
            
            for daughter_label in daughters:
                daughter_pos = centroids_next.get(int(daughter_label))
                if daughter_pos is not None:
                    arrows.append((mother_pos, daughter_pos))
    
    return arrows


def draw_lineage_overlay(ax, label_frame, classifications, arrows,
                         dividing_color='orange', daughter_color='cyan',
                         arrow_color='yellow', alpha=0.4):
    """Draw colored cell overlays and division arrows on an existing imshow axis.

    Paints a semi-transparent colored mask over dividing and daughter
    cells by building an RGBA overlay image (same ``H x W`` as
    ``label_frame``) and coloring it wherever the cell label matches, then
    draws arrows from mother centroids to daughter centroids.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that already has an image plotted on it via ``imshow``.
    label_frame : ndarray
        Integer label array of shape ``(H, W)`` or ``(H, W, C)``. If 3D,
        only the first channel is used.
    classifications : dict
        Mapping of ``{cell_label: 'dividing' | 'daughter' | 'normal'}``,
        as produced by :func:`classify_cells_in_frame`.
    arrows : list of tuple
        A list of ``((r0, c0), (r1, c1))`` centroid pairs, as produced by
        :func:`build_division_arrows`.
    dividing_color : str, optional
        Overlay color for dividing cells. Default is ``'orange'``.
    daughter_color : str, optional
        Overlay color for daughter cells. Default is ``'cyan'``.
    arrow_color : str, optional
        Color of the division arrows. Default is ``'yellow'``.
    alpha : float, optional
        Transparency of the cell color overlays, from 0 (invisible) to 1
        (opaque). Default is 0.4.

    Returns
    -------
    tuple
        A 2-tuple of ``(patch, arrow_artists)`` where ``patch`` is the
        ``AxesImage`` for the color overlay and ``arrow_artists`` is a
        list of the ``Annotation`` artists drawn for each arrow.
    """
    # Extract 2D label map
    if label_frame.ndim == 3:
        lf = label_frame[..., 0]
    else:
        lf = label_frame
    
    H, W = lf.shape
    
    # Build an RGBA overlay: starts fully transparent
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    
    color_map = {
        'dividing': (*plt.cm.colors.to_rgb(dividing_color), alpha),
        'daughter': (*plt.cm.colors.to_rgb(daughter_color), alpha),
    }
    
    for cell_label, cell_class in classifications.items():
        if cell_class in color_map:
            mask = lf == cell_label
            overlay[mask] = color_map[cell_class]
    
    # Draw the overlay on top of the imshow (zorder ensures it's above)
    patch = ax.imshow(overlay, zorder=2, interpolation='nearest')
    
    # Draw arrows: matplotlib uses (x, y) = (col, row), so we swap
    arrow_artists = []
    for (r0, c0), (r1, c1) in arrows:
        arr = ax.annotate(
            '', 
            xy=(c1, r1),        # tip of arrow (daughter position)
            xytext=(c0, r0),    # tail of arrow (mother position)
            arrowprops=dict(
                arrowstyle='->', 
                color=arrow_color, 
                lw=2,
                mutation_scale=15
            ),
            zorder=3
        )
        arrow_artists.append(arr)
    
    return patch, arrow_artists


def create_timelapse_gif_with_lineage(
    im1, im2, 
    lineage1=None, lineage2=None,
    output_path='timelapse.gif', fps=5,
    titles=('Predicted', 'True'),
    cmap='gray',
    dividing_color='orange',
    daughter_color='cyan',
    arrow_color='yellow',
    overlay_alpha=0.4
):
    """Create a side-by-side timelapse GIF of two label movies with lineage overlays.

    For each frame, cells about to divide are highlighted in
    ``dividing_color``, newly born daughter cells are highlighted in
    ``daughter_color``, and arrows are drawn from mother centroids to
    daughter centroids at division frames. The GIF is written to
    ``output_path``.

    Parameters
    ----------
    im1 : numpy.ndarray
        Time-lapse label image of shape ``(T, H, W, C)`` for the left
        panel. Pixel values are integer cell labels (0 = background).
    im2 : numpy.ndarray
        Time-lapse label image of shape ``(T, H, W, C)`` for the right
        panel, same format as ``im1``.
    lineage1 : dict or None, optional
        Lineage dict for ``im1``, keyed by cell label with values
        ``{'frames': [...], 'daughters': [...], 'parent': int or None}``.
        If None, no lineage overlay is drawn for that panel.
    lineage2 : dict or None, optional
        Lineage dict for ``im2``, same format as ``lineage1``. If None, no
        lineage overlay is drawn for that panel.
    output_path : str, optional
        Path to save the output GIF. Default is ``'timelapse.gif'``.
    fps : int, optional
        Frames per second for the GIF. Default is 5.
    titles : tuple, optional
        Titles for the two subplots. Default is ``('Predicted', 'True')``.
    cmap : str, optional
        Colormap for displaying the images. Default is ``'gray'``.
    dividing_color : str, optional
        Color highlight for cells that are about to divide. Default is
        ``'orange'``.
    daughter_color : str, optional
        Color highlight for cells that were just born from division.
        Default is ``'cyan'``.
    arrow_color : str, optional
        Color of arrows drawn from mother to daughter. Default is
        ``'yellow'``.
    overlay_alpha : float, optional
        Transparency of the cell color overlays, from 0 (invisible) to 1
        (opaque). Default is 0.4.

    Returns
    -------
    None
        Writes the animation to ``output_path`` as a side effect.
    """

    T, H, W, C = im1.shape
    
    # Pre-compute centroids for all frames (both panels)
    # This is done outside the animation loop for efficiency.
    # centroids[t] = {cell_label: (row, col)}
    def precompute_centroids(im, lineage):
        if lineage is None:
            return [{}] * T
        return [get_cell_centroids(im[t]) for t in range(T)]
    
    centroids1 = precompute_centroids(im1, lineage1)
    centroids2 = precompute_centroids(im2, lineage2)
    
    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im1_plot = axes[0].imshow(im1[0], cmap=cmap, zorder=1)
    im2_plot = axes[1].imshow(im2[0], cmap=cmap, zorder=1)
    
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[0].axis('off')
    axes[1].axis('off')
    
    plt.colorbar(im1_plot, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2_plot, ax=axes[1], fraction=0.046, pad=0.04)
    
    frame_text = fig.text(0.5, 0.02, f'Frame: 0/{T-1}', ha='center', fontsize=12)
    
    # Add a legend explaining the colors
    if lineage1 is not None or lineage2 is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=dividing_color, alpha=overlay_alpha, label='Dividing cell'),
            Patch(facecolor=daughter_color, alpha=overlay_alpha, label='Daughter cell'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                   ncol=2, bbox_to_anchor=(0.5, 0.06), fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # make room for legend + frame counter
    
    # These will hold references to overlay artists so we can remove them each frame
    # (matplotlib doesn't have a clean "remove all overlays" — we track them manually)
    overlay_artists = []
    
    def clear_overlays():
        for artist in overlay_artists:
            artist.remove()
        overlay_artists.clear()
    
    def add_overlays(frame):
        """Draw lineage overlays for both panels at the given frame."""
        label_frame1 = im1[frame, ..., 0] if C > 0 else im1[frame]
        label_frame2 = im2[frame, ..., 0] if C > 0 else im2[frame]
        
        for ax, label_frame, lineage, centroids in [
            (axes[0], label_frame1, lineage1, centroids1),
            (axes[1], label_frame2, lineage2, centroids2),
        ]:
            if lineage is None:
                continue
            
            # Classify cells in this frame
            classifications = classify_cells_in_frame(frame, lineage)
            
            # Build arrows: only meaningful if there's a next frame
            curr_cents = centroids[frame]
            next_cents = centroids[frame + 1] if frame + 1 < T else {}
            arrows = build_division_arrows(frame, lineage, curr_cents, next_cents)
            
            patch, arrow_artists = draw_lineage_overlay(
                ax, label_frame, classifications, arrows,
                dividing_color=dividing_color,
                daughter_color=daughter_color,
                arrow_color=arrow_color,
                alpha=overlay_alpha
            )
            overlay_artists.append(patch)
            overlay_artists.extend(arrow_artists)
    
    def update(frame):
        """Update function called by FuncAnimation for each frame."""
        # 1. Update the raw image data
        im1_plot.set_data(im1[frame])
        im2_plot.set_data(im2[frame])
        frame_text.set_text(f'Frame: {frame}/{T-1}')
        
        # 2. Remove old overlays and draw new ones
        #    We can't use blit=True easily here because overlays are new artists
        #    each frame — so we redraw manually instead.
        clear_overlays()
        add_overlays(frame)
        
        # Return all changed artists (for blit, though we disable it here)
        return [im1_plot, im2_plot, frame_text]
    
    # Draw overlays for the first frame before animation starts
    add_overlays(0)
    
    # Note: blit=False because we're adding/removing overlay artists dynamically.
    # blit=True only works cleanly when the same set of artists is reused each frame.
    anim = animation.FuncAnimation(fig, update, frames=T,
                                   interval=1000/fps, blit=False)
    
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"GIF saved to {output_path}")
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class PositionAnimationBuilder:
    """
    Build an animation incrementally by adding frames during iteration.
    """
    
    def __init__(self, marker_size=50, title="Position Animation", 
                 xlim=None, ylim=None, figsize=(8, 6)):
        """
        Initialize the animation builder.
        
        Parameters:
        -----------
        marker_size : int
            Size of scatter markers
        title : str
            Plot title
        xlim : tuple, optional
            (min, max) for x-axis. Auto-determined if None
        ylim : tuple, optional
            (min, max) for y-axis. Auto-determined if None
        figsize : tuple
            Figure size (width, height)
        """
        self.frames = []  # Store (x_positions, y_positions) for each frame
        self.images = []
        self.marker_size = marker_size
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
    
    def add_frame(self, x_positions, y_positions, image=None):
        """
        Add a new frame to the animation.
        
        Parameters:
        -----------
        x_positions : array-like
            X coordinates for this timestep
        y_positions : array-like
            Y coordinates for this timestep
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
        """
        Generate and display the animation.
        
        Parameters:
        -----------
        interval : int
            Delay between frames in milliseconds
        
        Returns:
        --------
        HTML object with embedded animation
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
    """
    Build an animation incrementally by adding frames during iteration.
    """
    
    def __init__(self, marker_size=50, title="Position Animation", 
                 xlim=None, ylim=None, figsize=(8, 6)):
        """
        Initialize the animation builder.
        
        Parameters:
        -----------
        marker_size : int
            Size of scatter markers
        title : str
            Plot title
        xlim : tuple, optional
            (min, max) for x-axis. Auto-determined if None
        ylim : tuple, optional
            (min, max) for y-axis. Auto-determined if None
        figsize : tuple
            Figure size (width, height)
        """
        self.frames = []  # Store (x_positions, y_positions) for each frame
        self.images = []
        self.marker_size = marker_size
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
    
    def add_frame(self, x_positions, y_positions, image=None):
        """
        Add a new frame to the animation.
        
        Parameters:
        -----------
        x_positions : array-like
            X coordinates for this timestep
        y_positions : array-like
            Y coordinates for this timestep
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
        """
        Generate and display the animation.
        
        Parameters:
        -----------
        interval : int
            Delay between frames in milliseconds
        
        Returns:
        --------
        HTML object with embedded animation
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
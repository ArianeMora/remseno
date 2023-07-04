###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

"""
Just a class to make all the plots asthetic and pub quality.
"""
import io
from sciviso import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


class Vis:

    def __init__(self, sciutil=None, cmap='Purples', dpi=300, ax_on_img=False,
                 style='ticks', palette='pastel', opacity=0.8, default_colour="teal", figsize=(3, 3),
                 title_font_size=12, label_font_size=8, title_font_weight="bold", text_font_weight="bold"):
        self.cmap = ListedColormap(sns.color_palette(cmap))
        self.figsize = figsize
        self.opacity = opacity
        self.dpi = dpi
        self.palette = palette
        self.style = style
        self.default_colour = default_colour
        self.label_font_size = label_font_size
        self.title_font_size = title_font_size
        self.title_font_weight = title_font_weight
        self.text_font_weight = text_font_weight
        self.axis_font_size = label_font_size
        self.axis_line_width = 1.0
        self.ax_on_img = ax_on_img
        self.palette = palette
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.cmap_str = cmap
        self.labels = None
        self.axis_line_width = 0.5
        self.bins, self.cluster_rows, self.cluster_cols, self.line_width = None, None, None, None
        self.col_colours, self.row_colours, self.vmin, self.vmax, self.x_tick_labels = None, None, None, None, None
        self.min_x, self.min_y, self.max_x, self.max_y = None, None, None, None
        self.add_legend, self.zlabel, self.colour = None, None, None
        self.hue, self.add_dots, self.add_stats, self.stat_method = None, None, None, None
        self.palette = palette if palette else ['#AAC7E2', '#FFC107', '#016957', '#9785C0',
                                                '#D09139', '#338A03', '#FF69A1', '#5930B1', '#FFE884', '#35B567',
                                                '#1E88E5',
                                                '#ACAD60', '#A2FFB4', '#B68F5', '#854A9C']
        plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text
        plt.rcParams['figure.figsize'] = self.figsize
        sns.set(rc={'figure.figsize': self.figsize, 'font.size': label_font_size}, style=self.style)

    def plot_coords(self, df, x, y, colour_col, show_plot=False, title=None, ax=None):
        sc = Scatterplot(df, x, y, title=title, xlabel=x, ylabel=y, colour=colour_col)
        ax = sc.plot(ax)
        if show_plot:
            plt.show()
        return ax

    def plot_img(self, band, show_plot=False, title=None, ax=None, cmap=None):
        # Plot the first band
        if ax is None:
            fig, ax = plt.subplots()
        cmap = cmap if cmap else self.cmap_str
        ax.imshow(band, cmap=cmap)
        ax.set_title(f'{title}')
        if not self.ax_on_img:
            self.remove_axes()
        if show_plot:
            plt.show()
        return ax

    def load_style(self, style_dict):
        """ Load a style from a dict. """
        self.ax_on_img = style_dict.get('ax_on_img') or self.ax_on_img
        self.default_colour = style_dict.get('default_colour') or self.default_colour
        self.label_font_size = style_dict.get('label_font_size') or self.label_font_size
        self.axis_font_size = style_dict.get('axis_font_size') or self.axis_font_size
        self.title_font_size = style_dict.get('title_font_size') or self.title_font_size
        self.title_font_weight = style_dict.get('title_font_weight') or self.title_font_weight
        self.text_font_weight = style_dict.get('text_font_weight') or self.text_font_weight
        self.palette = style_dict.get('palette') or self.palette
        self.figsize = style_dict.get('figsize') or self.figsize
        plt.rcParams['figure.figsize'] = self.figsize
        sns.set(rc={'figure.figsize': self.figsize, 'font.size': self.label_font_size}, style=self.style)
        self.cmap_str = style_dict.get('cmap') or self.cmap_str
        self.style = style_dict.get('style') or self.style
        self.cmap = ListedColormap(sns.color_palette(self.cmap_str))
        self.opacity = style_dict.get('opacity') or self.opacity
        self.labels = style_dict.get('labels') or self.labels
        self.bins = style_dict.get('bins') or self.bins
        self.cluster_rows = style_dict.get('cluster_rows') or self.cluster_rows
        self.cluster_cols = style_dict.get('cluster_cols') or self.cluster_cols
        self.line_width = style_dict.get('line_width') or self.line_width
        self.vmin = style_dict.get('vmin') or self.vmin
        self.vmax = style_dict.get('vmax') or self.vmax
        self.col_colours = style_dict.get('col_colours') or self.col_colours
        self.row_colours = style_dict.get('row_colours') or self.row_colours
        self.x_tick_labels = style_dict.get('x_tick_labels') or self.x_tick_labels
        self.min_x = style_dict.get('min_x') or self.min_x
        self.min_y = style_dict.get('min_y') or self.min_y
        self.max_x = style_dict.get('max_x') or self.max_x
        self.max_y = style_dict.get('max_y') or self.max_y
        self.add_legend = style_dict.get('add_legend') or self.add_legend
        self.zlabel = style_dict.get('zlabel') or self.zlabel
        self.colour = style_dict.get('colour') or self.colour
        self.add_dots = style_dict.get('add_dots') or self.add_dots
        self.add_stats = style_dict.get('add_stats') or self.add_stats
        self.stat_method = style_dict.get('stat_method') or self.stat_method
        self.hue = style_dict.get('hue') or self.hue
        self.s = style_dict.get('s') or 10
        self.axis_line_width = style_dict.get('axis_line_width') or 0.5

    def add_labels(self, title=True, x=True, y=True, ax=None):
        if ax is not None:
            if x:
                ax.set(xlabel=self.xlabel)
            if y:
                ax.set(ylabel=self.ylabel)
            if title:
                ax.title.set_text(self.title)
        else:
            if x:
                plt.xlabel(self.xlabel, fontsize=self.label_font_size, fontweight=self.text_font_weight)
            if y:
                plt.ylabel(self.ylabel, fontsize=self.label_font_size, fontweight=self.text_font_weight)
            if title:
                plt.title(self.title, fontsize=self.title_font_size, fontweight=self.title_font_weight)
        return ax

    @staticmethod
    def apply_limits(axis, max_v: float, min_v=None):
        min_v = 0 if min_v is None else min_v
        if axis == 'x' and max_v is not None:
            plt.xlim(min_v, max_v)
        elif axis == 'y' and max_v is not None:
            plt.ylim(min_v, max_v)

    @staticmethod
    def remove_axes():
        """
        Removes axis good for images.
        :param ax:
        :return:
        """
        plt.axis('off')

    def set_ax_params(self, ax):
        ax.tick_params(direction='out', length=2, width=self.axis_line_width)
        ax.spines['bottom'].set_linewidth(self.axis_line_width)
        ax.spines['top'].set_linewidth(0)
        ax.spines['left'].set_linewidth(self.axis_line_width)
        ax.spines['right'].set_linewidth(0)
        ax.tick_params(labelsize=self.axis_font_size)
        ax.tick_params(axis='x', which='major', pad=2.0)
        ax.tick_params(axis='y', which='major', pad=2.0)
        return ax

    @staticmethod
    def return_svg() -> str:
        """
        Returns the svg string.
        Parameters
        ----------
        plot

        Returns
        -------

        """
        file_io = io.StringIO()
        plt.savefig(file_io, format="svg")
        return file_io.getvalue()

    def save_svg(self, filename: str) -> None:
        plt.savefig(filename, bbox_inches='tight')

    def save_png(self, filename: str) -> None:
        plt.savefig(plt, filename, dpi=300, bbox_inches='tight')
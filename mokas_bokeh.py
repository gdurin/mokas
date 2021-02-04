import bokeh.plotting as plk
from bokeh.models import ColumnDataSource, Slider
from bokeh.io import curdoc
import matplotlib.colors as mpl_colors


def plot_errorbar(x, y, yerr, labels, color, size=5, fig=None, source=None):
    if fig is None:
        fig = plk.figure(title=title)
    # plot the points
    x_label, y_label, l_label = labels
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label
    l_label = str(l_label)
    if len(color) == 3:
        color = mpl_colors.to_hex(color)
    fig.circle(x,y, source=source, color=color, size=size, line_alpha=0, legend_label=l_label)
    # create the coordinates for the errorbars
    # plot them
    err_xs = []
    err_ys = []
    for _x, _y, _yerr in zip(x, y, yerr):
        err_xs.append((_x, _x))
        err_ys.append((_y - _yerr, _y + _yerr))
    fig.multi_line(err_xs, err_ys, color=color)
    return fig

def plot_velocity(df, Bxs):
    if fig is None:
        fig = plk.figure(plot_height=400, plot_width=400, title="Velocity at different angles",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[-180,180])
    df.reset_index(inplace=True)
    df.rename(columns = {'index': 'angle'}, inplace=True)
    source = ColumnDataSource(df)
    for bx in Bxs:
        c0 = str(bx)
        c1 = "%ierr" % bx
        fig.circle(x='angle', y=c0, source=source)
    angle = Slider(title="angle", value=0.0, start=-180.0, end=180, step=15)

def update_velocity(attr, old, new):

    # Get the current slider values
    a = angle.value
    
    # Generate the new curve
    
    source.data = dict(y=str(bx))

for w in [angle]:
    w.on_change('value', update_velocity)   
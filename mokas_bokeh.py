import bokeh.plotting as plk
from bokeh.models import ColumnDataSource, Slider, Range1d
from bokeh.io import push_notebook
from bokeh.layouts import column
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
    fig.circle(x,y, color=color, size=size, line_alpha=0, legend_label=l_label)
    # create the coordinates for the errorbars
    # plot them
    err_xs = []
    err_ys = []
    for _x, _y, _yerr in zip(x, y, yerr):
        err_xs.append((_x, _x))
        err_ys.append((_y - _yerr, _y + _yerr))
    fig.multi_line(err_xs, err_ys, color=color)
    return fig




def plot_velocity(df, fig=None):
    if fig is None:
        fig = plk.figure(plot_height=400, plot_width=400, title="Velocity at different angles",
              tools="crosshair,pan,box_zoom,save,reset",
              x_range=[-180,180])
    df.reset_index(inplace=True)
    df.rename(columns = {'index': 'Bx'}, inplace=True)
    source = ColumnDataSource(data=df)
    source1 = ColumnDataSource(data=df)
    fig.circle(x='Bx', y='0', source=source, color='blue')
    fig.circle(x='Bx', y='-180', source=source1, color='red')
    fig.x_range=Range1d(-180, 180)
    angle = Slider(title="angle", value=0, start=-180, end=180, step=15)
    angle.on_change('value', update_velocity)
    out = column(fig, angle)
    return out

def update_velocity(attr, old, new):

    # Get the current slider values
    a = angle.value
    if a < 0:
        b = 180 - a
        a, b = b, a
    else:
        b = -180 + a

    a = int(a)
    b = int(a)

    # Generate the new curve

    source.data = dict(y=a)
    source1.data = dict(y=b)
    push_notebook()


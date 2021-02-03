import bokeh.plotting as plk
import matplotlib.colors as mpl_colors


def plot_errorbar(x, y, yerr, labels, color, size=5, fig=None):
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
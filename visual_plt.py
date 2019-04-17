import matplotlib
matplotlib.use('Agg')
# above 2 lines set the matplotlib backend to 'Agg', which
#  enables matplotlib-plots to also be generated if no X-server
#  is defined (e.g., when running in basic Docker-container)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.utils import make_grid
import numpy as np


def open_pdf(full_path):
    return PdfPages(full_path)


def plot_images_from_tensor(image_tensor, pdf=None, nrow=8, title=None):
    '''Plot images in [image_tensor] as a grid with [nrow] into [pdf].

    [image_tensor]      <tensor> [batch_size]x[channels]x[width]x[height]'''

    image_grid = make_grid(image_tensor, nrow=nrow, pad_value=1)  # pad_value=0 would give black borders
    plt.imshow(np.transpose(image_grid.numpy(), (1,2,0)))
    if title:
        plt.title(title)
    if pdf is not None:
        pdf.savefig()


def plot_scatter_groups(x, y, colors=None, ylabel=None, xlabel=None, title=None, top_title=None, names=None,
                        xlim=None, ylim=None, markers=None, figsize=None):
    '''Generate a figure containing a scatter-plot.'''

    # if needed, generate default group-names
    if names == None:
        n_groups = len(y)
        names = ["group " + str(id) for id in range(n_groups)]

    # make plot
    f, axarr = plt.subplots(1, 1, figsize=(12, 7) if figsize is None else figsize)
    for i,name in enumerate(names):
        # plot individual points
        axarr.scatter(x=x[i], y=y[i], color=None if (colors is None) else colors[i],
                      marker="o" if markers is None else markers[i], s=40, alpha=0.5)
        # plot group means
        axarr.scatter(x=np.mean(x[i]), y=np.mean(y[i]), color=None if (colors is None) else colors[i], label=name,
                      marker="*" if markers is None else markers[i], s=160)

    # finish layout
    # -set y/x-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    if xlim is not None:
        axarr.set_xlim(xlim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -add legend
    if names is not None:
        axarr.legend()

    # return the figure
    return f


def plot_bar(numbers, names=None, colors=None, ylabel=None, title=None, top_title=None, ylim=None, figsize=None,
             yerr=None):
    '''Generate a figure containing a bar-graph.'''

    # number of bars
    n_bars = len(numbers)

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)
    axarr.bar(x=range(n_bars), height=numbers, color=colors, yerr=yerr)

    # finish layout
    axarr.set_xticks(range(n_bars))
    if names is not None:
        axarr.set_xticklabels(names, rotation=-20)
        axarr.legend()
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)

    # return the figure
    return f


def plot_lines(list_with_lines, x_axes=None, line_names=None, colors=None, title=None,
               title_top=None, xlabel=None, ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded",
               x_log=False, with_dots=False, h_line=None, h_label=None, h_error=None,
               h_lines=None, h_colors=None, h_labels=None, h_errors=None):
    '''Generates a figure containing multiple lines in one plot.

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :param colors:          <list> containing the colors of each line
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>
    '''

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = list(range(n_obs))

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)

    # add error-lines / shaded areas
    if list_with_errors is not None:
        for task_id, name in enumerate(line_names):
            if errors=="shaded":
                axarr.fill_between(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])),
                                   list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])),
                                   color=None if (colors is None) else colors[task_id], alpha=0.25)
            else:
                axarr.plot(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])), label=None,
                           color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')
                axarr.plot(x_axes, list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])), label=None,
                           color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')

    # mean lines
    for task_id, name in enumerate(line_names):
        axarr.plot(x_axes, list_with_lines[task_id], label=name,
                   color=None if (colors is None) else colors[task_id],
                   linewidth=2, marker='o' if with_dots else None)

    # add horizontal line
    if h_line is not None:
        axarr.axhline(y=h_line, label=h_label, color="grey")
        if h_error is not None:
            if errors == "shaded":
                axarr.fill_between([x_axes[0], x_axes[-1]],
                                   [h_line + h_error, h_line + h_error], [h_line - h_error, h_line - h_error],
                                   color="grey", alpha=0.25)
            else:
                axarr.axhline(y=h_line + h_error, label=None, color="grey", linewidth=1, linestyle='dashed')
                axarr.axhline(y=h_line - h_error, label=None, color="grey", linewidth=1, linestyle='dashed')

    # add horizontal lines
    if h_lines is not None:
        h_colors = colors if h_colors is None else h_colors
        for task_id, new_h_line in enumerate(h_lines):
            axarr.axhline(y=new_h_line, label=None if h_labels is None else h_labels[task_id],
                          color=None if (h_colors is None) else h_colors[task_id])
            if h_errors is not None:
                if errors == "shaded":
                    axarr.fill_between([x_axes[0], x_axes[-1]],
                                       [new_h_line + h_errors[task_id], new_h_line+h_errors[task_id]],
                                       [new_h_line - h_errors[task_id], new_h_line - h_errors[task_id]],
                                       color=None if (h_colors is None) else h_colors[task_id], alpha=0.25)
                else:
                    axarr.axhline(y=new_h_line+h_errors[task_id], label=None,
                                  color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                                  linestyle='dashed')
                    axarr.axhline(y=new_h_line-h_errors[task_id], label=None,
                                  color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                                  linestyle='dashed')

    # finish layout
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend
    if line_names is not None:
        axarr.legend()

    # -set x-axis to log-scale
    if x_log:
        axarr.set_xscale('log')

    # return the figure
    return f


def plot_bars(number_list, names=None, colors=None, ylabel=None, title_list=None, top_title=None, ylim=None,
              figsize=None, yerr=None):
    '''Generate a figure containing multiple bar-graphs.

    [number_list]   <list> with <lists> of numbers to plot in each sub-graph
    [names]         <list> (with <lists>) of names for axis
    [colors]        <list> (with <lists>) of colors'''

    # number of plots
    n_plots = len(number_list)

    # number of bars per plot
    n_bars = []
    for i in range(n_plots):
        n_bars.append(len(number_list[i]))

    # decide on scale y-axis
    y_max = np.max(number_list)+0.07*np.max(number_list)

    # make figure
    size = (16,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, n_plots, figsize=size)

    # make all plots
    for i in range(n_plots):
        axarr[i].bar(x=range(n_bars[i]), height=number_list[i], color=colors[i] if type(colors[0])==list else colors,
                     yerr=yerr[i] if yerr is not None else None)

        # finish layout for this plot
        if ylim is None:
            axarr[i].set_ylim(0, y_max)
        else:
            axarr[i].set_ylim(ylim)
        axarr[i].set_xticks(range(n_bars[i]))
        if names is not None:
            axarr[i].set_xticklabels(names[i] if type(names[0])==list else names, rotation=-20)
            axarr[i].legend()
        if i==0 and ylabel is not None:
            axarr[i].set_ylabel(ylabel)
        if title_list is not None:
            axarr[i].set_title(title_list[i])

    # finish global layout
    if top_title is not None:
        f.suptitle(top_title)

    # return the figure
    return f
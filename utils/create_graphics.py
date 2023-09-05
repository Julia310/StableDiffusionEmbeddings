import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_utils import make_dir
import numpy as np
import matplotlib.gridspec as gridspec


def plot_histogram(data, path, prompt):
    sns.histplot(data, kde=True)
    make_dir(path)
    plt.savefig(f'{path}/{prompt[0:30]}.png')


"""def create_boxplot(values, prompt_aesthetics=None, filename=None):
    # Create the figure and boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(values)

    # Add the mean as an empty dot to the plot
    if prompt_aesthetics is not None:
        ax.scatter([1], [prompt_aesthetics], marker='o', color='blue', s=50)

    # Save the figure to disk if a filename is provided
    if filename is not None:
        print(f'save {filename}')
        plt.savefig(filename)

    return fig"""


def create_boxplots(data, labels, x_label='Index', y_label='Values'):
    # Create a new figure
    plt.figure(figsize=(8.5, 6))

    # Font sizes
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 18

    # Create the boxplot
    bp = plt.boxplot(data, labels=labels, patch_artist=True)

    # Coloring the boxes white
    for box in bp['boxes']:
        box.set_facecolor('white')

    # Add a title and labels with defined font sizes
    plt.title(f'{y_label.replace(" x 10", "")} optimization', fontsize=title_fontsize)
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)

    # Adjusting the tick font sizes
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Show the plot with a tight layout
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'./output/{y_label.replace(" x 10", "").replace(" ", "_")}_boxplot.pdf', format='pdf')

    # Display the plot
    plt.show()


import matplotlib.pyplot as plt

def create_combined_boxplots(data1, data2, labels, x_label='Index', y_label1='Metric 1', y_label2='Metric 2'):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 30), sharex=True)

    # Font sizes
    title_fontsize = 18
    label_fontsize = 18
    tick_fontsize = 18

    # First metric on ax1
    bp1 = ax1.boxplot(data1, patch_artist=True, labels=labels)
    ax1.set_ylabel(y_label1, fontsize=label_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize)

    # Color the boxes white for the first metric
    for box in bp1['boxes']:
        box.set_facecolor('white')

    # Second metric on ax2
    bp2 = ax2.boxplot(data2, patch_artist=True, labels=labels)
    ax2.set_ylabel(y_label2, fontsize=label_fontsize)
    ax2.set_xlabel(x_label, fontsize=label_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    ax2.tick_params(axis='x', labelsize=tick_fontsize)

    # Color the boxes white for the second metric
    for box in bp2['boxes']:
        box.set_facecolor('white')

    # Set both y-axes to the same scale based on the combined range of both datasets
    combined_min = min(min(map(min, data1)), min(map(min, data2)))
    combined_max = max(max(map(max, data1)), max(map(max, data2)))
    ax1.set_ylim(combined_min, combined_max)
    ax2.set_ylim(combined_min, combined_max)

    # Adjust the layout
    fig.tight_layout()

    # Save the combined plot
    plt.savefig('./output/separated_boxplot.pdf', format='pdf')

    # Display the plot
    plt.show()

# Sample usage
# data1 = [[1,2,3], [2,3,4]]
# data2 = [[4,5,6], [5,6,7]]
# labels = ["Metric A", "Metric B"]
# create_combined_boxplots(data1, data2, labels, "Shared X-Label", "Value A", "Value B")


"""def create_boxplots(data_progression, labels_progression, data_delta, labels_delta, x_label='Index', y_label='Values'):
    fig = plt.figure(figsize=(9.2, 6))  # Increase the total canvas width, e.g., to 11 inches

    # Adjust the width ratio to leave more space on the right
    gs = gridspec.GridSpec(1, 3, width_ratios=[7, 2, 0.2])  # Add more whitespace ratio to the right

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharey=ax1)

    # Font sizes
    title_fontsize = 18
    label_fontsize = 18
    tick_fontsize = 18

    # Calculate box width scaling factor
    original_fig_width = 9  # This was your original width
    scaling_factor = original_fig_width / fig.get_size_inches()[0]

    # Create the boxplot for progression
    bp1 = ax1.boxplot(data_progression, labels=labels_progression, patch_artist=True)  # Adjust width here
    faded_blue = "#D0E8FF"
    for box in bp1['boxes']:
        box.set_facecolor('white')

    ax1.set_xlabel(x_label, fontsize=label_fontsize)
    ax1.set_ylabel(y_label, fontsize=label_fontsize)
    ax1.set_title(f'{y_label.replace(" x 10", "")} optimization', fontsize=title_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Create the boxplot for delta
    bp2 = ax2.boxplot(data_delta, patch_artist=True, widths=0.35 * scaling_factor)  # Adjust width here
    faded_green = "#B2D8B2"
    for box in bp2['boxes']:
        box.set_facecolor('white')

    ax2.set_title(f'{y_label.replace(" x 10", "")} delta', fontsize=title_fontsize)
    ax2.set_xticks([])

    # Prevent y-ticks from displaying on the second subplot
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'./output/{y_label.replace(" x 10", "").replace(" ", "_")}_combined_boxplot.pdf', format='pdf')

    # Show the plot
    plt.show()"""


#def create_boxplots(data_progression, labels_progression, data_delta, labels_delta, x_label='Index', y_label='Values'):
#    fig = plt.figure(figsize=(9, 6))  # Set the total width (10 + 4) and height of 6
#
#    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 2])  # This sets the relative width of the two subplots
#
#    ax1 = plt.subplot(gs[0])
#    ax2 = plt.subplot(gs[1], sharey=ax1)  # This ensures the y-axis is shared
#
#    # Font sizes
#    title_fontsize = 18
#    label_fontsize = 18
#    tick_fontsize = 18
#
#    # Create the boxplot for progression
#    bp1 = ax1.boxplot(data_progression, labels=labels_progression, patch_artist=True)
#    faded_blue = "#D0E8FF"  # Light faded blue
#    for box in bp1['boxes']:
#        # set face color for progression boxes to white
#        box.set_facecolor('white')
#
#    ax1.set_xlabel(x_label, fontsize=label_fontsize)
#    #ax1.set_ylabel(y_label, fontsize=label_fontsize)
#    ax1.set_title(f'{y_label} optimization', fontsize=title_fontsize)
#    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
#
#    # Create the boxplot for delta
#    bp2 = ax2.boxplot(data_delta, patch_artist=True, widths=0.35)
#    faded_green = "#B2D8B2"  # Light faded green
#    for box in bp2['boxes']:
#        # set face color for delta boxes to white
#        box.set_facecolor('white')
#
#    ax2.set_title(f'{y_label} delta', fontsize=title_fontsize)
#    ax2.set_xticks([])
#
#    ax2.yaxis.tick_right()  # Move the ticks to the right side for the second subplot
#    ax2.tick_params(axis='y', which='major', labelsize=tick_fontsize)
#
#    # Save the combined plot
#    plt.tight_layout()
#    #plt.savefig(f'./output/{y_label.replace(" ", "_")}_combined_boxplot.png')
#    plt.savefig(f'./output/{y_label.replace(" ", "_")}_combined_boxplot.pdf', format='pdf')
#
#    # Show the plot
#    plt.show()


def plot_scores(scores, save_dir, y_label='Scores', window_size=5, num_dots=5):
    """
    Create a line plot of scores and save the image to the specified directory.

    Parameters:
    scores (list): A list of scores to plot.
    save_dir (str): The directory to save the image to.
    """

    smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')

    plt.plot(range(window_size - 1, len(scores)), smoothed_scores)
    plt.scatter(np.argpartition(scores, -num_dots)[-num_dots:] + window_size - 1,
                np.partition(scores, -num_dots)[-num_dots:], color='red', zorder=10)
    plt.ylabel(y_label)
    plt.xlabel('Iterations')
    plt.savefig(save_dir)
    plt.close()
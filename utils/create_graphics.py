import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_utils import make_dir
import numpy as np


def plot_histogram(data, path, prompt):
    sns.histplot(data, kde=True)
    make_dir(path)
    plt.savefig(f'{path}/{prompt[0:30]}.png')


def create_boxplot(values, prompt_aesthetics=None, filename=None):
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

    return fig


def create_boxplots(data, labels):
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    plt.boxplot(data, labels=labels, patch_artist=True)

    # Add a title and labels
    plt.title('Boxplots for Different Indices')
    plt.xlabel('Index')
    plt.ylabel('Values')

    # Show the plot
    plt.savefig('boxplot.png')




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
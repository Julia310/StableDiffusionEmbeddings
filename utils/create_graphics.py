import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_utils import make_dir


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


def plot_scores(scores, file_path, y_label='Score', x_label='Image'):
    """
    Create a line plot of aesthetic scores and save the image to the specified directory.

    Parameters:
    scores (list): A list of aesthetic scores to plot.
    save_dir (str): The directory to save the image to.
    """
    plt.plot(scores)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_path)
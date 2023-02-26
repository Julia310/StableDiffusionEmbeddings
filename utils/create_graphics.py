import matplotlib.pyplot as plt

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


def plot_aesthetic_scores(scores, save_dir):
    """
    Create a line plot of aesthetic scores and save the image to the specified directory.

    Parameters:
    scores (list): A list of aesthetic scores to plot.
    save_dir (str): The directory to save the image to.
    """
    plt.plot(scores)
    plt.ylabel('Aesthetic Score')
    plt.xlabel('Image')
    plt.savefig(save_dir)
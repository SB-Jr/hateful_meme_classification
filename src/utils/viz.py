import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(dataset, num_samples, randomize=True, output="Notebook"):
    """
    visualize the hate meme classification dataset
    inputs:
        output: use either "Notebook" to show output in notebook or a path with filename to store the output at that location
    """
    fig, axs = plt.subplots(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), figsize=(10, 10))
    axs = axs.flatten()
    samples_memes = np.random.randint(0, 100, num_samples)
    for i in range(num_samples):
        sample_memeID = samples_memes[i]
        img, label = dataset[sample_memeID]

        # Show input image.
        axs[i].imshow(img)
        axs[i].axis(False)
        axs[i].set_title(label)

    plt.tight_layout()
    if output == "Notebook":
        plt.show()
    else:
        plt.savefig(output)
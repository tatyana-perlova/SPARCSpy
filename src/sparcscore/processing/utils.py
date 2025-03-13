import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb
import os
import h5py
from matplotlib.colors import LinearSegmentedColormap, to_rgba


from sparcscore.processing.preprocessing import percentile_normalization


def plot_image(array, size = (10,10), save_name="", cmap="magma", **kwargs):
    """
    Visualize and optionally save an input array as an image.

    This function displays a 2D array using matplotlib and can save
    the resulting image as a PNG file.

    Parameters
    ----------
    array : np.array
        Input 2D numpy array to be plotted.
    size : tuple of int, optional
        Figure size in inches, by default (10, 10).
    save_name : str, optional
        Name of the output file, without extension. If not provided, image will not be saved, by default "".
    cmap : str, optional
        Color map used to display the array, by default "magma".
    **kwargs : dict
        Additional keyword arguments to be passed to `ax.imshow`.

    Example
    -------
    >>> array = np.random.rand(10, 10)
    >>> plot_image(array, size=(5, 5))
    """

    fig = plt.figure(frameon=False)
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap=cmap, **kwargs)
    
    if save_name != "":
        plt.savefig(save_name + ".png")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


def visualize_class(class_ids, seg_map, background ,*args, **kwargs):
    """
    Visualize specific classes in a segmentation map by highlighting them on top of a background image.

    This function takes in class IDs and a segmentation map, and creates an output visualization
    where the specified classes are highlighted on top of the provided background image.

    Parameters
    ----------
    class_ids : array-like
        A list or array of integers representing the class IDs to be highlighted.

    seg_map : 2D array-like
        A 2D array representing the segmentation map, where each value corresponds to a class ID.

    background : 2D/3D array-like
        Background image (2D or 3D) on which the classes will be highlighted. Its size should match
        that of `seg_map`.

    *args : additional positional arguments
        Any additional positional arguments that are passed to the underlying plotting functions.

    **kwargs : additional keyword arguments
        Any additional keyword arguments that are passed underlying plotting functions.

    Returns
    -------
    None
        The function will display the highlighted image but does not return any values.

    Example
    -------
    >>> class_ids = [1, 2]
    >>> seg_map = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 1]])
    >>> background = np.random.random((3, 3)) * 255
    >>> visualize_class(class_ids, seg_map, background)
    """
    # Remove class ID 0 from the seg_map, as it is used to represent the background
    index = np.argwhere(class_ids==0.)
    class_ids_no_zero = np.delete(class_ids, index)

    # Set the values in the output map to 2 for the specified class IDs
    outmap_map = np.where(np.isin(seg_map, class_ids_no_zero), 2, seg_map)

    # Set the values in the output map to 1 for all class IDs other than the specified classes
    outmap_map = np.where(np.isin(seg_map, class_ids, invert=True), 1,outmap_map)
    
    # Convert the output map to an RGB image with class areas highlighted
    image = label2rgb(outmap_map,background/np.max(background),alpha=0.4, bg_label=0)
    
    # Display the resulting image
    plot_image(image, **kwargs)
    
def download_testimage(folder):
    """
    Download a set of test images to a provided folder path.
    
    Parameters
    ----------
    folder : string
        The path of the folder where the test images will be saved.

    Returns
    -------
    returns : list
        A list containing the local file paths of the downloaded images.

    Example
    -------
    >>> folder = "test_images"
    >>> downloaded_images = download_testimage(folder)
    Successfully downloaded testimage_dapi.tiff from https://zenodo.org/record/5701474/files/testimage_dapi.tiff?download=1
    Successfully downloaded testimage_wga.tiff from https://zenodo.org/record/5701474/files/testimage_wga.tiff?download=1
    >>> print(downloaded_images)
    ['test_images/testimage_dapi.tiff', 'test_images/testimage_wga.tiff']
    """
    
    # Define the test images' names and URLs
    images = [("testimage_dapi.tiff","https://zenodo.org/record/5701474/files/testimage_dapi.tiff?download=1"),
             ("testimage_wga.tiff","https://zenodo.org/record/5701474/files/testimage_wga.tiff?download=1")]
    
    import urllib.request
    
    returns = []
    for name, url in images:
        # Construct the local file path for the current test image
        path = os.path.join(folder, name)
        
        # Open the local file and write the contents of the test image URL
        f = open(path,'wb')
        f.write(urllib.request.urlopen(url).read())
        f.close()

        # Print a message confirming the download and add the local file path to the output list
        print(f"Successfully downloaded {name} from {url}")
        returns.append(path)
    return returns

def flatten(l):
    """
    Flatten a list of lists into a single list.

    This function takes in a list of lists (nested lists) and returns a single list
    containing all the elements from the input lists.

    Parameters
    ----------
    l : list of lists
        A list containing one or more lists as its elements.

    Returns
    -------
    flattened_list : list
        A single list containing all elements from the input lists.

    Example
    -------
    >>> nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> flatten(nested_list)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    # Flatten the input list using list comprehension
    return [item for sublist in l for item in sublist]
       
def load_data_from_h5(project_path, image_id=0, crop_size=None):
    """ Loads channel and segmentation data from an HDF5 file. """
    seg_path = f"{project_path}/segmentation/input_segmentation.h5"
    with h5py.File(seg_path, "r") as hf:
        if crop_size is None:
            channels = hf['input_images'][image_id, :, :, :]
            segmentation = hf['segmentation'][image_id, :, :, :]
        else:
            channels = hf['input_images'][image_id, :, :crop_size, :crop_size]
            segmentation = hf['segmentation'][image_id, :, :crop_size, :crop_size]

    return channels, segmentation


def create_rgb_image(channels, channel_color_map, normalization_percentiles):
    """ Processes channels data to create an RGB image based on specified mappings and normalization percentiles. """
    crop_size = channels.shape[1]  # Assuming all channels are of the same dimensions
    rgb_image = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
    color_idx_map = {'red': 0, 'green': 1, 'blue': 2}

    # Normalize and assign channels to specified colors in the RGB image
    for channel_index, color in channel_color_map.items():
        if color in color_idx_map:
            low_perc, high_perc = normalization_percentiles.get(channel_index, (0.1, 0.99))
            normalized_channel = percentile_normalization(channels[channel_index, :, :], 
                                                          low_perc, high_perc)
            rgb_image[..., color_idx_map[color]] = normalized_channel

    return rgb_image

def show_segmentation(rgb_image, segmentation, mask_colors={0: 'g', 1: 'r'}):
    # If return_fig is true, create and return a figure displaying the RGB image
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(rgb_image)  # Show the RGB image
    # Draw contours for each segmentation mask
    for i, mask in enumerate(segmentation):
        ax.contour(mask, levels=[0.5], 
                   colors=mask_colors.get(i), 
                   linewidths=1)  # White color for visibility
    plt.axis("off")
    return fig

def plot_seg_overlay_timecourse(project_path, channel_color_map={0: 'red', 1: 'green'}, 
                                selection=None, return_fig=False, crop_size=None, 
                                normalization_percentiles={0: (0.1, 0.98), 1: (0.05, .96)},
                                mask_colors={0: 'g', 1: 'r'},
                                image_id=0):
    """
    Wrapper function to load data and create an RGB image.
    """
    channels, segmentation = load_data_from_h5(project_path, image_id, crop_size)  # Ignoring segmentation

    rgb_image = create_rgb_image(channels, channel_color_map, normalization_percentiles)

    fig = show_segmentation(rgb_image, segmentation, mask_colors)
    if return_fig:
        return(fig)
    
    
def create_custom_colormap(base_color_name):
    """
    Creates colormap for image visualization, copied from Anatoly's function
    """
    base_color = to_rgba(base_color_name)
    start_color = to_rgba("black")
    colors = [start_color, base_color]
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=65536)
    return cmap

def visualize_single_cells_by_id(project_path, 
                                 cell_ids_to_plot,
                                colors, **kwargs):
    """
    Create plots of invidual channels in all the channels
    """
    cells_path = f"{project_path}/extraction/data/single_cells.h5"
    n_cells = len(cell_ids_to_plot)
    
    with h5py.File(cells_path, "r") as hf:
        cells = hf.get("single_cell_data")
        cell_ids = hf.get('single_cell_index')[...]
        n_channels = cells.shape[1]
        
        cell_ids = pd.DataFrame(cell_ids, columns=['index', 'cell_id'])

        fig, axs = plt.subplots(n_cells, n_channels, figsize = (n_channels*1, 
                                                                n_cells*1))
        
        cells_to_return = []
        for i, cell_id in enumerate(cell_ids_to_plot):
            index = cell_ids[cell_ids.cell_id == cell_id].index.values[0]
            image = cells[index]
            cells_to_return.append(cells[index])
            for n in range(n_channels):
                image_n = create_custom_colormap(colors[n])(
                    percentile_normalization(image[n], **kwargs)
                )
                axs[i, n].imshow(image_n)
                axs[i, n].axis("off")
    return(cells_to_return)
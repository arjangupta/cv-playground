import numpy as np

def simple_example():
    # Original array
    array = np.array([[1, 2], [3, 4]])

    # Pad the array with a constant value of 0 at both ends
    padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=(0, 0))

    print(padded_array)

def tuple_args_example():
    # Example array with shape (4, 3, 3, 2)
    images = np.random.rand(4, 3, 3, 2)

    # Pad the array with a width of 3
    width = 3
    padded_images = np.pad(images, pad_width=((0, 0), (width, width), (width, width), (0, 0)), mode='constant', constant_values=0)

    print(padded_images.shape)
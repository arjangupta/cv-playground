import numpy as np

def simple_example():
    # Print simple example message
    print("--- Simple Syntax Example ---")
    # Original array
    array = np.array([[1, 2], [3, 4]])

    # Pad the array with a constant value of 0 at both ends
    padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=(0, 0))

    print(padded_array)

def tuple_args_example():
    # Print tuple args example message
    print("--- Tuple Args Example ---")

    # Example array with shape 3x4x4 (3 channels, 4x4 image)
    images = np.random.randint(0, 255, size=(3, 4, 4))
    # Show the shape of the array
    print(f"Original array shape: {images.shape}")
    # Show the array
    print(images)

    # Pad the array with a width variable
    width = 2
    padded_images = np.pad(images, pad_width=((0, 0), (width, width), (width, width)), mode='constant', constant_values=(0, 0))

    print(f"After padding with width {width}")
    print(f"New array shape: {padded_images.shape}")
    print(padded_images)

if __name__ == '__main__':
    simple_example()
    tuple_args_example()
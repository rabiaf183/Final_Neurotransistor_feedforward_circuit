# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 08:38:01 2025

@author: risc915d
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import gzip

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def rescale_mnist_images(images, target_size):
    """
    Rescales MNIST images to the specified target size and normalizes pixel values to [0, 1].

    Parameters:
        images (numpy.ndarray): Array of MNIST images with shape (num_images, height, width).
        target_size (tuple): Desired target size as (new_height, new_width).

    Returns:
        numpy.ndarray: Rescaled and normalized MNIST images.
    """
    # Ensure images are reshaped to 28x28 if necessary
    if len(images.shape) == 2:  # If flat array
        num_images = images.shape[0]
        images = images.reshape(num_images, 28, 28)

    # Rescale each image to the target size
    rescaled_images = np.array([
        zoom(image, (target_size[0] / image.shape[0], target_size[1] / image.shape[1]))
        for image in images
    ])

    # Normalize pixel values to [0, 1]
    rescaled_images = rescaled_images / 255.0

    return rescaled_images


def generate_pulse_train(image_row, pulse_voltage, pulse_width, pulse_slope):
    """
    Generate time and voltage arrays for a pulse train based on an image row.

    Parameters:
    - image_row: Array of pixel values (0 to 1) for one row of the image
    - pulse_voltage: Maximum voltage of each pulse
    - pulse_width: Width of each pulse
    - pulse_slope: Rise and fall time of each pulse

    Returns:
    - time: Array of time points
    - voltage: Array of voltage values corresponding to the time points
    """
    num_pixels = len(image_row)

    # Define pulse shape (only two points per transition)
    pulse_points = np.array([0, pulse_slope])
    
    # Generate time array
    time = np.zeros(num_pixels * 2 + 1)
    for i in range(num_pixels):
        time[2*i:2*i+2] = i * pulse_width + pulse_points
    time[-1] = num_pixels * pulse_width  # Add final point
    
    # Generate voltage array
    voltage = np.zeros(len(time))
    for i, pixel_value in enumerate(image_row):
        current_voltage = pixel_value * pulse_voltage
        voltage[2*i:2*i+2] = [voltage[2*i-1] if i > 0 else 0, current_voltage]
    voltage[-1] = voltage[-2]  # Set final point to last voltage

    return time, voltage


def generate_pulse_trains(digit_image, pulse_voltage, pulse_width, pulse_slope):
    """
    Generate time and voltage arrays for a pulse train based on an image row.

    Parameters:
    - pulse_voltage: Maximum voltage of each pulse
    - pulse_width: Width of each pulse
    - pulse_slope: Rise and fall time of each pulse

    Returns:
    - times: Array of time points
    - voltages: Array of voltage values corresponding to the time points
    """
    rows = digit_image.shape[1]
    # Initialize lists for this digit
    times = []
    voltages = []
    # create pulse train     
    for row in range(rows):
        time, voltage = generate_pulse_train(digit_image[row], pulse_voltage, pulse_width, pulse_slope)
        times.append(time)
        voltages.append(voltage)
    
    return times, voltages




def get_MNIST_pulse_train(size, pulse_voltage, pulse_width, pulse_slope, selected_digits=None, do_plot=None, specific_image_index=None):
    # Load MNIST dataset
    X_train, y_train = load_mnist('raw', kind='train')
    
    # Reshape images to 28x28
    X_train = X_train.reshape(-1, 28, 28)
    
    # Parameters
    target_size = (size, size)
    
    # Scale images to target size
    X_train_scaled = np.array([zoom(x, (target_size[0] / x.shape[0], target_size[1] / x.shape[1])) for x in X_train])
    
    # Normalize pixel values to [0, 1]
    X_train_scaled = X_train_scaled / 255.0
    
    # Create a dictionary to store pulse trains for each digit
    pulse_trains = {}
    
    # If no digits are specified, process all digits
    if selected_digits is None:
        selected_digits = range(10)
    
    # Create pulse trains for selected digits
    for digit in selected_digits:
        # Get all occurrences of the digit
        digit_images = X_train_scaled[y_train == digit]
        
        # If a specific image index is provided, use only that image
        if specific_image_index is not None:
            digit_images = [digit_images[specific_image_index]]
        
        for idx, digit_image in enumerate(digit_images):
            # Initialize lists for this digit
            times = []
            voltages = []
        
            # create pulse train     
            for row in range(size):
                time, voltage = generate_pulse_train(digit_image[row], pulse_voltage, pulse_width, pulse_slope)
                times.append(time)
                voltages.append(voltage)
            
            # Store the pulse trains for this digit
            if specific_image_index is not None:
                pulse_trains[digit] = {'times': times, 'voltages': voltages}
            else:
                if digit not in pulse_trains:
                    pulse_trains[digit] = []
                pulse_trains[digit].append({'times': times, 'voltages': voltages})
            
            if do_plot is not None:        
                # plot    
                plt.figure(figsize=(15, 10))
                plt.suptitle(f"Pulse Trains for Digit {digit}, Image {idx}")
                
                for row in range(size):
                    plt.subplot(size, 1, row + 1)
                    plt.plot(times[row], voltages[row],'.-')
                    plt.title(f"Row {row + 1}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Voltage (V)")
                    plt.ylim(-0.1, pulse_voltage + 0.1)
        
                plt.tight_layout()
                plt.show()
            
                # Display the scaled digit image
                plt.figure(figsize=(size, size))
                plt.imshow(digit_image, cmap='gray')
                plt.title(f"Scaled Digit {digit}, Image {idx}")
                plt.colorbar()
                plt.show()
    
    return pulse_trains


def create_MNIST_pulse_train(X_train, y_train, size, pulse_voltage, pulse_width, pulse_slope, selected_digits=None, do_plot=None, specific_image_index=None):
    # Load MNIST dataset
    #X_train, y_train = load_mnist('raw', kind='train')
    
    # Reshape images to 28x28
    X_train = X_train.reshape(-1, 28, 28)
    
    # Parameters
    target_size = (size, size)
    
    # Scale images to target size
    X_train_scaled = np.array([zoom(x, (target_size[0] / x.shape[0], target_size[1] / x.shape[1])) for x in X_train])
    
    # Normalize pixel values to [0, 1]
    X_train_scaled = X_train_scaled / 255.0
    
    # Create a dictionary to store pulse trains for each digit
    pulse_trains = {}
    
    # If no digits are specified, process all digits
    if selected_digits is None:
        selected_digits = range(10)
    
    # Create pulse trains for selected digits
    for digit in selected_digits:
        # Get all occurrences of the digit
        digit_images = X_train_scaled[y_train == digit]
        
        # If a specific image index is provided, use only that image
        if specific_image_index is not None:
            digit_images = [digit_images[specific_image_index]]
        
        for idx, digit_image in enumerate(digit_images):
            # Initialize lists for this digit
            times = []
            voltages = []
        
            # create pulse train     
            for row in range(size):
                time, voltage = generate_pulse_train(digit_image[row], pulse_voltage, pulse_width, pulse_slope)
                times.append(time)
                voltages.append(voltage)
            
            # Store the pulse trains for this digit
            if specific_image_index is not None:
                pulse_trains[digit] = {'times': times, 'voltages': voltages}
            else:
                if digit not in pulse_trains:
                    pulse_trains[digit] = []
                pulse_trains[digit].append({'times': times, 'voltages': voltages})
            
            if do_plot is not None:        
                # plot    
                plt.figure(figsize=(15, 10))
                plt.suptitle(f"Pulse Trains for Digit {digit}, Image {idx}")
                
                for row in range(size):
                    plt.subplot(size, 1, row + 1)
                    plt.plot(times[row], voltages[row],'.-')
                    plt.title(f"Row {row + 1}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Voltage (V)")
                    plt.ylim(-0.1, pulse_voltage + 0.1)
        
                plt.tight_layout()
                plt.show()
            
                # Display the scaled digit image
                plt.figure(figsize=(size, size))
                plt.imshow(digit_image, cmap='gray')
                plt.title(f"Scaled Digit {digit}, Image {idx}")
                plt.colorbar()
                plt.show()

    return pulse_trains,digit_image




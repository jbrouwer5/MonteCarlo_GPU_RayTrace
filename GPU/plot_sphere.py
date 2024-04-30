import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Path to the file
file_path = 'output.txt'

# Read the data from the file
def read_data(file_path, size=1000):
    with open(file_path, 'r') as file:
        # Assuming all values are in a single line
        line = file.readline()
        # Split the line by comma, convert each to float, then create a numpy array
        data = np.array([float(val) for val in line.split(',')], dtype=float)
        # Reshape the array into a 2D array (size x size)
        data = data.reshape(size, size)
    return data

# Plot the 2D list
def plot_data(data):
    plt.imshow(data, cmap='gray', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title('Sphere')
    plt.savefig("Sphere.png") 
    plt.close()

# Main
if __name__ == '__main__':
    # Assuming the square grid is 1000x1000, if it's different, adjust the size parameter accordingly
    data = read_data(file_path, size=1000)
    plot_data(data)
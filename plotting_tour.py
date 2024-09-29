import matplotlib.pyplot as plt

# Function to read the input data (coordinates only) from a file
def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        # Skip the first line (either EUCLIDEAN or NON-EUCLIDEAN)
        file.readline()
        
        # Read the second line to get the number of cities
        N = int(file.readline().strip())
        
        # Read the next N lines to get the coordinates of the cities
        coordinates = []
        for _ in range(N):
            line = file.readline().strip()
            coordinates.append(tuple(map(float, line.split())))
            
    return coordinates

# Function to plot the tour
def plot_tour(coordinates, best_tour, best_cost):
    # Create a figure for the plot
    plt.figure(figsize=(10, 8))
    
    # Extract the x and y coordinates
    x_coords = [coordinates[city][0] for city in best_tour]
    y_coords = [coordinates[city][1] for city in best_tour]
    
    # Add the first city to the end to complete the cycle
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    # Plot the cities
    plt.scatter(x_coords, y_coords, color='blue', zorder=5)
    
    # Annotate cities with their index
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=12, ha='right', color='red')
    
    # Plot the path (tour)
    plt.plot(x_coords, y_coords, color='green', zorder=1)
    
    # Add title and labels
    plt.title(f"TSP Tour (Cost: {best_cost})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Show the plot
    plt.grid(True)
    plt.show()

# Main function to read data, plot the tour, and display the result
def main(file_path, best_tour, best_cost):
    # Read the coordinates from the input file
    coordinates = read_coordinates(file_path)
    
    # Plot the best tour
    plot_tour(coordinates, best_tour, best_cost)

# Example usage:
# You can pass this script the path to your input file, the best tour, and its cost.
# For example:
if __name__ == "__main__":
    # Example tour and cost (replace with actual values)
    file_path = 'Assignment 2\data_set_100_NON-EUCLIDEAN'  # Replace this with your actual file path
    best_tour = [52, 21, 78, 14, 25, 56, 37, 76, 99, 19, 7, 54, 82, 74, 58, 10, 95, 73, 90, 35, 18, 11, 65, 86, 85, 87, 55, 23, 4, 51, 29, 26, 68, 98, 48, 67, 36, 94, 66, 28, 22, 83, 16, 46, 33, 72, 1, 27, 9, 89, 88, 61, 77, 42, 62, 75, 32, 30, 41, 47, 96, 59, 69, 84, 45, 13, 60, 44, 6, 8, 97, 20, 49, 12, 50, 43, 31, 24, 17, 15, 53, 0, 2, 93, 91, 80, 63, 92, 81, 71, 39, 40, 79, 64, 34, 5, 3, 70, 57, 38] # Replace with the actual best tour found
    best_cost = 5282  # Replace with the actual best cost found
    
    # Call the main function to plot the tour
    main(file_path, best_tour, best_cost)

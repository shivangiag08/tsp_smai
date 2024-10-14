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
    file_path = 'Assignment 2/tsp_smai\data_set_100_EUCLIDEAN'  # Replace this with your actual file path
    best_tour =  [82, 32, 93, 20, 91, 56, 42, 27, 62, 79, 71, 98, 59, 52, 8, 74, 77, 3, 85, 13, 44, 90, 84, 26, 63, 64, 36, 87, 29, 38, 35, 31, 24, 86, 61, 30, 89, 10, 34, 78, 16, 22, 11, 70, 53, 47, 28, 40, 69, 4, 18, 68, 41, 23, 6, 39, 97, 49, 92, 14, 9, 45, 57, 94, 75, 72, 55, 1, 96, 7, 65, 58, 19, 88, 48, 60, 73, 25, 83, 2, 0, 17, 21, 37, 76, 54, 46, 51, 81, 50, 66, 99, 5, 33, 12, 43, 80, 95, 15, 67]  # Replace with the actual best tour found
    best_cost = 2394  # Replace with the actual best cost found
    
    # Call the main function to plot the tour
    main(file_path, best_tour, best_cost)

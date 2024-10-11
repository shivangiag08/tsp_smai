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
    file_path = '.\data_set_200_EUCLIDEAN'  # Replace this with your actual file path
    best_tour =  [51, 95, 146, 157, 112, 171, 46, 3, 193, 43, 167, 85, 184, 88, 29, 154, 104, 128, 86, 175, 34, 140, 108, 162, 84, 64, 17, 28, 176, 7, 159, 62, 25, 38, 125, 39, 93, 91, 180, 126, 199, 13, 179, 0, 65, 121, 198, 178, 109, 68, 1, 98, 2, 73, 60, 66, 170, 18, 151, 12, 79, 57, 186, 55, 163, 9, 16, 31, 113, 92, 105, 71, 81, 134, 35, 131, 94, 8, 119, 4, 165, 148, 20, 155, 52, 5, 75, 122, 197, 133, 106, 136, 53, 101, 30, 182, 33, 116, 83, 82, 168, 6, 22, 70, 195, 90, 141, 191, 74, 14, 61, 173, 192, 87, 24, 149, 56, 36, 100, 150, 69, 11, 89, 42, 63, 111, 166, 181, 177, 144, 67, 142, 32, 169, 124, 183, 45, 187, 26, 78, 47, 185, 164, 156, 138, 50, 97, 19, 49, 27, 40, 103, 76, 190, 137, 114, 123, 188, 196, 161, 132, 44, 21, 23, 189, 107, 117, 160, 158, 96, 48, 172, 110, 153, 194, 80, 54, 58, 135, 41, 127, 102, 99, 72, 115, 147, 77, 143, 10, 59, 120, 152, 174, 15, 118, 145, 130, 37, 129, 139]    # Replace with the actual best tour found
    best_cost = 1140.1695617285568  # Replace with the actual best cost found
    
    # Call the main function to plot the tour
    main(file_path, best_tour, best_cost)

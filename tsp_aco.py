import time
start_time = time.time()

def read_input(file_path):
    with open(file_path, 'r') as file:
        # Read the first line to determine if it's Euclidean or Non-Euclidean
        distance_type = file.readline().strip()

        # Read the second line to get the number of cities
        N = int(file.readline().strip())

        # Read the next N lines to get the coordinates of the cities
        coordinates = []
        for _ in range(N):
            line = file.readline().strip()
            coordinates.append(tuple(map(float, line.split())))

        # Read the next N lines to get the distance matrix
        distance_matrix = []
        for _ in range(N):
            line = file.readline().strip()
            distance_matrix.append(list(map(float, line.split())))

    # Return the parsed data
    return distance_type, N, coordinates, distance_matrix

import numpy as np
import random

# Function to calculate the total cost of a given tour using the distance matrix
def calculate_tour_cost(tour, distance_matrix):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += distance_matrix[tour[i]][tour[i + 1]]
    total_cost += distance_matrix[tour[-1]][tour[0]]  # Return to the starting city
    return total_cost

# Ant Colony Optimization algorithm for TSP
class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, num_iterations, alpha=1, beta=5, evaporation_rate=0.5, pheromone_deposit=100):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Distance importance (heuristic factor)
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.num_cities = len(distance_matrix)
        self.pheromones = np.ones((self.num_cities, self.num_cities))  # Initial pheromone levels

    # Heuristic information: inverse of the distance (1/d)
    def heuristic(self, i, j):
        return 1 / (self.distance_matrix[i][j] + 1e-10)  # Adding a small constant to avoid division by zero

    # Construct a solution for an ant (tour) based on the pheromone and heuristic information
    def construct_solution(self):
        ant_tour = []
        start_city = random.randint(0, self.num_cities - 1)  # Random starting city
        ant_tour.append(start_city)
        visited = set(ant_tour)

        # Build the tour by selecting the next city probabilistically
        for _ in range(self.num_cities - 1):
            current_city = ant_tour[-1]
            probabilities = []

            for next_city in range(self.num_cities):
                if next_city not in visited:
                    pheromone = self.pheromones[current_city][next_city] ** self.alpha
                    heuristic_value = self.heuristic(current_city, next_city) ** self.beta
                    probabilities.append(pheromone * heuristic_value)
                else:
                    probabilities.append(0)  # If the city is already visited, set its probability to zero

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()  # Normalize probabilities
            next_city = np.random.choice(range(self.num_cities), p=probabilities)
            ant_tour.append(next_city)
            visited.add(next_city)

        return ant_tour

    # Update the pheromones based on the solutions of the ants
    def update_pheromones(self, ant_solutions):
        # Evaporate some pheromones
        self.pheromones *= (1 - self.evaporation_rate)

        # Deposit pheromones based on the solutions of the ants
        for ant_tour, tour_cost in ant_solutions:
            pheromone_to_add = self.pheromone_deposit / tour_cost
            for i in range(len(ant_tour) - 1):
                self.pheromones[ant_tour[i]][ant_tour[i + 1]] += pheromone_to_add
                self.pheromones[ant_tour[i + 1]][ant_tour[i]] += pheromone_to_add  # Ensure symmetry

            # Complete the cycle by returning to the start city
            self.pheromones[ant_tour[-1]][ant_tour[0]] += pheromone_to_add
            self.pheromones[ant_tour[0]][ant_tour[-1]] += pheromone_to_add

    # Run the Ant Colony Optimization algorithm
    def run(self):
        best_tour = None
        best_cost = float('inf')

        for iteration in range(self.num_iterations):
            ant_solutions = []

            # Each ant constructs a solution
            for _ in range(self.num_ants):
                ant_tour = self.construct_solution()
                tour_cost = calculate_tour_cost(ant_tour, self.distance_matrix)
                ant_solutions.append((ant_tour, tour_cost))

                # Update best solution found so far
                if tour_cost < best_cost:
                    best_tour = ant_tour
                    best_cost = tour_cost

            # Update pheromones after all ants have constructed their solutions
            self.update_pheromones(ant_solutions)

            if iteration % 10 == 0:  # Print progress every 10 iterations
                print(best_tour)
                # print(f"Iteration {iteration + 1}/{self.num_iterations}: Best Cost = {best_cost}")
            if iteration == self.num_iterations - 1:
                print("best_tour")

        return best_tour, best_cost

# Main function to read input and solve TSP using Ant Colony Optimization
def main(file_path):
    # Use the read_input function to get data from the file
    distance_type, N, coordinates, distance_matrix = read_input(file_path)

    # Parameters for Ant Colony Optimization
    num_ants = 50
    num_iterations = 200
    alpha = 1  # Pheromone importance
    beta = 5  # Heuristic importance
    evaporation_rate = 0.5
    pheromone_deposit = 100

    # Solve TSP using Ant Colony Optimization
    aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
    best_tour, best_cost = aco.run()

    # Output the best tour in path representation and its cost
    print("Best Tour (0-indexed):", best_tour)
    print("Minimum Cost of Tour:", best_cost)

# Example usage:
file_path = '.\data_set_50_NON-EUCLIDEAN'  # Replace with the path to your input file
main(file_path)

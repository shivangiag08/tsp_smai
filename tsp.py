import time
import sys
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

# Example usage:
import random
import numpy as np

# Function to calculate the total cost of a given tour using the distance matrix
def calculate_tour_cost(tour, distance_matrix):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += distance_matrix[tour[i]][tour[i + 1]]
    total_cost += distance_matrix[tour[-1]][tour[0]]  # Return to the starting city
    return total_cost

# Initialize population with random tours
def initialize_population(pop_size, N):
    population = []
    for _ in range(pop_size):
        tour = list(range(N))
        random.shuffle(tour)
        population.append(tour)
    return population

# Select parents based on tournament selection
def selection(population, fitness, k=3):
    selected = random.sample(range(len(population)), k)
    best = min(selected, key=lambda idx: fitness[idx])
    return population[best]

# Perform ordered crossover (OX) between two parents
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted([random.randint(0, size - 1) for _ in range(2)])
    child = [None] * size
    child[start:end+1] = parent1[start:end+1]
    
    pointer = 0
    for i in range(size):
        if parent2[i] not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = parent2[i]
    return child

# Perform mutation by swapping two cities in the tour
def mutate(tour, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# Evolve population using selection, crossover, and mutation
def evolve_population(population, fitness, elite_size, mutation_rate):
    new_population = []
    # Keep the elite (best solutions)
    elite = [population[i] for i in np.argsort(fitness)[:elite_size]]
    new_population.extend(elite)
    
    # Create rest of the population via crossover and mutation
    while len(new_population) < len(population):
        parent1 = selection(population, fitness)
        parent2 = selection(population, fitness)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)
    
    return new_population

# Main Genetic Algorithm for TSP
def genetic_algorithm_tsp(distance_matrix, N, pop_size=600, generations=2500, elite_size=30, mutation_rate=0.25):
    # Initialize population
    population = initialize_population(pop_size, N)
    
    # Iterate over generations
    for generation in range(generations):
        # Calculate fitness for each individual
        fitness = [calculate_tour_cost(tour, distance_matrix) for tour in population]
        
        # Evolve population
        population = evolve_population(population, fitness, elite_size, mutation_rate)
        
        # Print best solution every 50 generations
        if generation % 50 == 0:
            best_cost = min(fitness)
            print(f"Generation {generation}: Best Cost = {best_cost}")
    
    # Get the best solution from the final population
    final_fitness = [calculate_tour_cost(tour, distance_matrix) for tour in population]
    best_tour_idx = np.argmin(final_fitness)
    best_tour = population[best_tour_idx]
    best_cost = final_fitness[best_tour_idx]
    
    return best_tour, best_cost

# Main function to read input and solve TSP using Genetic Algorithm
def main(file_path):
    # Use the read_input function to get data from the file
    distance_type, N, coordinates, distance_matrix = read_input(file_path)
    
    # Solve TSP using Genetic Algorithm
    best_tour, best_cost = genetic_algorithm_tsp(distance_matrix, N)
    
    # Output the best tour in path representation and its cost
    print("Best Tour (0-indexed):", best_tour)
    print("Minimum Cost of Tour:", best_cost)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]  #get the file path from command line arguments
    main(file_path)
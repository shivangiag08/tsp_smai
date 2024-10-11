import time
import sys
start_time = time.time()

def read_input(file_path):
    with open(file_path, 'r') as file:
        distance_type = file.readline().strip()
        N = int(file.readline().strip())
        coordinates = []
        for _ in range(N):
            line = file.readline().strip()
            coordinates.append(tuple(map(float, line.split())))

        distance_matrix = []
        for _ in range(N):
            line = file.readline().strip()
            distance_matrix.append(list(map(float, line.split())))

    return distance_type, N, coordinates, distance_matrix

import numpy as np
import random

def calculate_tour_cost(tour, distance_matrix):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += distance_matrix[tour[i]][tour[i + 1]]
    total_cost += distance_matrix[tour[-1]][tour[0]] 
    return total_cost

class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, num_iterations, alpha=1, beta=5, evaporation_rate=0.5, pheromone_deposit=100):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  
        self.beta = beta  
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.num_cities = len(distance_matrix)
        self.pheromones = np.ones((self.num_cities, self.num_cities)) 

    def heuristic(self, i, j):
        return 1 / (self.distance_matrix[i][j] + 1e-10)  
    def construct_solution(self):
        ant_tour = []
        start_city = random.randint(0, self.num_cities - 1)  
        ant_tour.append(start_city)
        visited = set(ant_tour)

        for _ in range(self.num_cities - 1):
            current_city = ant_tour[-1]
            probabilities = []

            for next_city in range(self.num_cities):
                if next_city not in visited:
                    pheromone = self.pheromones[current_city][next_city] ** self.alpha
                    heuristic_value = self.heuristic(current_city, next_city) ** self.beta
                    probabilities.append(pheromone * heuristic_value)
                else:
                    probabilities.append(0)  

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()  
            next_city = np.random.choice(range(self.num_cities), p=probabilities)
            ant_tour.append(next_city)
            visited.add(next_city)

        return ant_tour

    def update_pheromones(self, ant_solutions):
        self.pheromones *= (1 - self.evaporation_rate)

        for ant_tour, tour_cost in ant_solutions:
            pheromone_to_add = self.pheromone_deposit / tour_cost
            for i in range(len(ant_tour) - 1):
                self.pheromones[ant_tour[i]][ant_tour[i + 1]] += pheromone_to_add
                self.pheromones[ant_tour[i + 1]][ant_tour[i]] += pheromone_to_add  

            self.pheromones[ant_tour[-1]][ant_tour[0]] += pheromone_to_add
            self.pheromones[ant_tour[0]][ant_tour[-1]] += pheromone_to_add

    def run(self):
        best_tour = None
        best_cost = float('inf')

        start_time = time.time()
        timeout = 280

        for iteration in range(self.num_iterations):

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Timeout!")
                break

            ant_solutions = []

            for _ in range(self.num_ants):
                ant_tour = self.construct_solution()
                tour_cost = calculate_tour_cost(ant_tour, self.distance_matrix)
                ant_solutions.append((ant_tour, tour_cost))

                if tour_cost < best_cost:
                    best_tour = ant_tour
                    best_cost = tour_cost

            self.update_pheromones(ant_solutions)

            if iteration % 10 == 0: 
                print(best_tour)
            if iteration == self.num_iterations - 1:
                print(best_tour)

        return best_tour, best_cost

def main(file_path):
    distance_type, N, coordinates, distance_matrix = read_input(file_path)

    num_ants = 50
    num_iterations = 200
    alpha = 1  
    beta = 5 
    evaporation_rate = 0.5
    pheromone_deposit = 100

    aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_deposit)
    best_tour, best_cost = aco.run()

    print("Best Tour (0-indexed):", best_tour)
    print("Minimum Cost of Tour:", best_cost)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]  # Get the file path from command line arguments
    main(file_path)

import time
import sys
import numpy as np
import random
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def read_input():
    distance_type = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    coordinates = []
    for _ in range(N):
        line = sys.stdin.readline().strip()
        coordinates.append(tuple(map(float, line.split())))

    distance_matrix = []
    for _ in range(N):
        line = sys.stdin.readline().strip()
        distance_matrix.append(list(map(float, line.split())))

    return distance_type, N, coordinates, distance_matrix

def calculate_tour_cost(tour, distance_matrix):
    return np.sum([distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)]) + distance_matrix[tour[-1]][tour[0]]

class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, alpha=1, beta=5, evaporation_rate=0.5, pheromone_deposit=100):
        self.distance_matrix = np.array(distance_matrix)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.num_cities = len(distance_matrix)
        self.pheromones = np.ones((self.num_cities, self.num_cities)) 
        self.heuristics = np.divide(1, self.distance_matrix + 1e-10)  
        
    def construct_solution(self):
        start_city = random.randint(0, self.num_cities - 1)
        ant_tour = [start_city]
        visited = {start_city}

        for _ in range(self.num_cities - 1):
            current_city = ant_tour[-1]

            pheromone = self.pheromones[current_city] ** self.alpha
            heuristic = self.heuristics[current_city] ** self.beta
            attractiveness = pheromone * heuristic

            for city in visited:
                attractiveness[city] = 0  

            probabilities = attractiveness / np.sum(attractiveness)
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
            self.pheromones[ant_tour[-1]][ant_tour[0]] += pheromone_to_add  

    def run(self, global_best, lock):
        best_tour = None
        best_cost = float('inf')
        start_time = time.time()
        timeout = 300  

        iteration = 0
        while True:  
            if time.time() - start_time > timeout:
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
                with lock:
                    if best_cost < global_best['cost']:
                        global_best['tour'] = best_tour
                        global_best['cost'] = best_cost
                        print(f"\n\nGlobal best tour so far: {best_tour} \ncost = {best_cost}")

            iteration += 1

        return best_tour, best_cost

def run_aco_simulation(distance_matrix, num_ants, alpha, beta, evaporation_rate, pheromone_deposit, global_best, lock):
    aco = AntColonyOptimization(distance_matrix, num_ants, alpha, beta, evaporation_rate, pheromone_deposit)
    return aco.run(global_best, lock)

def main(file_path):
    distance_type, N, coordinates, distance_matrix = read_input(file_path)

    num_ants = 50
    alpha = 1
    beta = 5
    evaporation_rate = 0.5
    pheromone_deposit = 100
    num_threads = max(4, os.cpu_count() - 3) 

    global_best = {'tour': None, 'cost': float('inf')}
    lock = Lock()  
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_aco_simulation, distance_matrix, num_ants, alpha, beta, evaporation_rate, pheromone_deposit, global_best, lock) for _ in range(num_threads)]

        for future in futures:
            future.result() 

    print("Best Tour (0-indexed):", global_best['tour'])
    print("Minimum Cost of Tour:", global_best['cost'])

if __name__ == "__main__":
    main()

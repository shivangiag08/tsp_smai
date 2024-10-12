import random
import time
import sys

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

import random
import time
import sys

def calculate_tour_cost(tour, distance_matrix):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += distance_matrix[tour[i]][tour[i + 1]]
    total_cost += distance_matrix[tour[-1]][tour[0]]
    return total_cost

class TabuSearch:
    def __init__(self, distance_matrix, num_iterations, tabu_tenure, neighborhood_size):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_iterations = num_iterations
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size

    def generate_initial_solution(self):
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

    def generate_neighborhood(self, current_tour):
        neighborhood = []
        moves = []
        for _ in range(self.neighborhood_size):
            new_tour = current_tour[:]
            i, j = random.sample(range(self.num_cities), 2)  #selecting two random cities to swap
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]  #swapping the two cities
            neighborhood.append(new_tour)
            moves.append((i, j))  #store the swapped cities (move)
        return neighborhood, moves

    def run(self):
        current_tour = self.generate_initial_solution()
        current_cost = calculate_tour_cost(current_tour, self.distance_matrix)
        best_tour = current_tour
        best_cost = current_cost

        tabu_list = []
        tabu_dict = {}

        for iteration in range(self.num_iterations):
            neighborhood, moves = self.generate_neighborhood(current_tour)

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            #evaluate all neighbors and choose the best non-tabu move
            for neighbor, move in zip(neighborhood, moves):
                neighbor_cost = calculate_tour_cost(neighbor, self.distance_matrix)

                move_sorted = tuple(sorted(move))  #sort the move to avoid order issues in tabu list
                if neighbor_cost < best_neighbor_cost and (move_sorted not in tabu_dict or tabu_dict[move_sorted] < iteration):
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move_sorted

            if best_neighbor:
                current_tour = best_neighbor
                current_cost = best_neighbor_cost

                if current_cost < best_cost:
                    best_tour = current_tour
                    best_cost = current_cost

                #update tabu list and dictionary with the best move
                tabu_list.append(best_move)
                tabu_dict[best_move] = iteration + self.tabu_tenure

            if iteration % 10 == 0:  #print progress every 10 iterations
                print(f"Iteration {iteration}, Best cost: {best_cost}")

        return best_tour, best_cost

def main(file_path):
    distance_type, N, coordinates, distance_matrix = read_input(file_path)

    num_iterations = 1000
    tabu_tenure = 50
    neighborhood_size = 100

    ts = TabuSearch(distance_matrix, num_iterations, tabu_tenure, neighborhood_size)
    best_tour, best_cost = ts.run()

    print("Best Tour (0-indexed):", best_tour)
    print("Minimum Cost of Tour:", best_cost)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]  #get the file path from command line arguments
    main(file_path)

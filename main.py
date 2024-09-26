import random
import copy
import matplotlib.pyplot as plt

MAX_X_DIM = 12
MAX_Y_DIM = 10

POPULATION_SIZE = 100
NUM_GENERATIONS = 1000

GENE_MUTATION_RATE = 0.01
ROTATION_MUTATION = 0.01

FRESH_BLOOD = 0.1

TOURNAMENT_SIZE = 6
USE_ROULETTE_SELECTION = True

ONE_CROSSOVER = False
TWO_CROSSOVER = True

ELITE = 0.05


class GardenPuzzleSolver:
    def __init__(self, custom_map=None):
        if custom_map:
            self.x_dim = len(custom_map)
            self.y_dim = len(custom_map[0])
            self.map = copy.deepcopy(custom_map)
        else:
            # Generate a random garden map with specified dimensions
            self.x_dim = random.randrange(2, MAX_X_DIM + 1)
            self.y_dim = random.randrange(2, MAX_Y_DIM + 1)
            self.map = []
            for _ in range(self.x_dim):
                row = []
                for _ in range(self.y_dim):
                    # Initialize cells with -1 (rock) or 0 (empty) based on mutation rate
                    row.append(-1 if random.random() < GENE_MUTATION_RATE else 0)
                self.map.append(row)
        # Calculate the count of rocks and empty cells in the garden
        self.rock_count = sum(row.count(-1) for row in self.map)
        self.max_empty_cells = sum(row.count(0) for row in self.map)

    def __str__(self):
        # Convert the garden map to a string for printing
        s = ""
        for row in self.map:
            for col in row:
                s += ' K ' if col == -1 else '%2d ' % col
            s += '\n'
        return s

class GeneticPathFinder:

    def __init__(self, garden, initialize=True):
        self.garden = garden
        self.final_garden = GardenPuzzleSolver(garden.map)
        self.genes = []
        self.fitness = 0
        if initialize:
            # Initialize the genes for the pathfinding
            for _ in range(garden.x_dim + garden.y_dim + garden.rock_count):
                self.genes.append(self.createGene())
            self.solution()

    def __str__(self):
        # Convert the genes to a string for printing
        return '\n'.join(str(g) for g in self.genes)

    def tournament_selection(self, population):
        # Perform tournament selection on a population
        tournament = random.sample(population, TOURNAMENT_SIZE)
        return max(tournament, key=lambda x: x.fitness)

    def select_parent(self, population, selection_method):
        # Select a parent based on the specified selection method (roulette or tournament)
        if selection_method == "roulette":
            total_fitness = sum(individual.fitness for individual in population)
            threshold = random.uniform(0, total_fitness)
            current_fitness = 0
            for individual in population:
                current_fitness += individual.fitness
                if current_fitness >= threshold:
                    return individual
        elif selection_method == "tournament":
            return self.tournament_selection(population)

    def solution(self):
        g = self.final_garden  # Get the final garden state
        i = 0 # Initialize the step count

        for gene in self.genes:  # Initialize the step count
            pos = list(gene.start)  # Initialize the current position
            direction = gene.direction
            ri = 0

            if g.map[pos[0]][pos[1]] != 0:
                continue  # Skip if the starting position is not empty
            i += 1

            while True:
                g.map[pos[0]][pos[1]] = i
                # Update the position based on the current direction
                if direction == 'up':
                    pos[0] -= 1
                elif direction == 'down':
                    pos[0] += 1
                elif direction == 'left':
                    pos[1] -= 1
                else:
                    pos[1] += 1

                if pos[0] not in range(g.x_dim) or pos[1] not in range(g.y_dim):
                    break

                if g.map[pos[0]][pos[1]] == 0:
                    continue
                # Handle movement when the position is not empty
                if direction == 'up':
                    pos[0] += 1
                elif direction == 'down':
                    pos[0] -= 1
                elif direction == 'left':
                    pos[1] += 1
                else:
                    pos[1] -= 1
                # Determine the neighboring cells based on the direction
                if direction == 'up' or direction == 'down':
                    n = (
                        [pos[0], pos[1] - 1],
                        [pos[0], pos[1] + 1],
                    )
                else:
                    n = (
                        [pos[0] - 1, pos[1]],
                        [pos[0] + 1, pos[1]],
                    )
                nv = []
                for p in n:
                    try:
                        nv.append(g.map[p[0]][p[1]])
                    except IndexError:
                        nv.append('e')

                if nv.count(0) == 1:
                    pos = n[nv.index(0)]

                elif nv.count(0) == 2:
                    pos = n[gene.rotation[ri]]
                    ri += 1
                    if ri == len(gene.rotation):
                        ri = 0

                else:
                    if 'e' not in nv:
                        self.set_fitness()  # Calculate and set the fitness value
                        return
                    break
                # Update the direction based on the neighbors
                if direction in ('up', 'down'):
                    direction = 'left' if n.index(pos) == 0 else 'right'
                else:
                    direction = 'up' if n.index(pos) == 0 else 'down'

            self.set_fitness()

    def set_fitness(self):
        self.fitness = sum(1 for x in sum(self.final_garden.map, []) if x > 0)

    def crossover(self, other):
        new_individual = GeneticPathFinder(self.garden, initialize=False)
        global ONE_CROSSOVER
        global TWO_CROSSOVER

        p = random.random()
        # Determine which crossover type to use
        if (ONE_CROSSOVER and TWO_CROSSOVER) or (ONE_CROSSOVER == False and TWO_CROSSOVER == False):
            if random.choice([True, False]):
                ONE_CROSSOVER = True
                TWO_CROSSOVER = False
            else:
                ONE_CROSSOVER = False
                TWO_CROSSOVER = True

        if ONE_CROSSOVER:
            # One point crossover
            if p < 0.40:
                point = random.randrange(len(self.genes))
                new_individual.genes = self.genes[:point] + other.genes[point:]
            elif p < 0.80:
                new_individual.genes = []
                for i in range(len(self.genes)):
                    new_individual.genes.append(random.choice((self.genes[i], other.genes[i])))
            else:
                new_individual.genes = random.choice((self.genes, other.genes))

        if TWO_CROSSOVER:
            # Two point crossover
            gene_len = min(len(self.genes), len(other.genes))
            if p < 0.40 and gene_len >= 2:  # Ensure gene_len is at least 2 for two-point crossover
                point1, point2 = sorted(random.sample(range(gene_len), 2))
                new_individual.genes = (
                        self.genes[:point1] + other.genes[point1:point2] + self.genes[point2:]
                )
            # Mutations
            for i in range(len(new_individual.genes)):
                p = random.random()
                if p < GENE_MUTATION_RATE:
                    new_individual.genes[i] = self.createGene()
                elif p < ROTATION_MUTATION:
                    new_individual.genes[i].generate_rotation()

        new_individual.solution()   # Perform pathfinding to update the garden

        return new_individual

    def createGene(self):
        m = self.garden.x_dim
        n = self.garden.y_dim

        # Generate a random number to determine gene type
        x = random.randrange(2 * (m + n))
        if x < n:
            start = (0, x)
            direction = 'down'
        elif n <= x < m + n:
            start = (x - n, n - 1)
            direction = 'left'
        elif m + n <= x < 2 * n + m:
            start = (m - 1, 2 * n + m - x - 1)
            direction = 'up'
        else:
            start = (2 * (m + n) - x - 1, 0)
            direction = 'right'
        gene = Gene(start, direction)
        gene.generate_rotation()
        return gene


class Gene:
    def __init__(self, start, direction):
        self.start = start
        self.direction = direction

    def generate_rotation(self):
        # Generate a random binary rotation pattern for the gene
        self.rotation = [
            int(a) for a in bin(random.randrange(1024))[2:].zfill(10)
        ]

    def __str__(self):
        return '%s, %s, %s' % (str(self.start), self.rotation, self.direction)


def generate_fresh_blood(garden):
    fresh_blood = GeneticPathFinder(garden)
    return fresh_blood


best_fitness_values = []  # Initialize an empty list to store best fitness values


def solve(custom_map=None, selection_method=None):
    if custom_map:
        garden = GardenPuzzleSolver(custom_map)
    else:
        garden = GardenPuzzleSolver()
    population = []

    for _ in range(POPULATION_SIZE):
        population.append(GeneticPathFinder(garden))

    for generation_count in range(NUM_GENERATIONS):
        best_individual = max(population, key=lambda x: x.fitness)
        best_fitness_values.append(best_individual.fitness)  # Record the best fitness in this generation

        next_generation = [best_individual]

        # Calculate the number of elite individuals to preserve
        num_elite = int(POPULATION_SIZE * ELITE)

        # Sort the population by fitness in descending order
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Preserve the elite individuals in the next generation
        next_generation.extend(population[:num_elite])

        if FRESH_BLOOD > 0:
            num_fresh_blood = int(POPULATION_SIZE * FRESH_BLOOD)
            for _ in range(num_fresh_blood):
                # Generate new individuals (fresh blood)
                fresh_blood = generate_fresh_blood(garden)
                next_generation.append(fresh_blood)

        for _ in range(POPULATION_SIZE - num_elite - num_fresh_blood):
            parent1 = best_individual.select_parent(population, selection_method)
            parent2 = best_individual.select_parent(population, selection_method)
            next_generation.append(parent1.crossover(parent2))
        population = next_generation
        print('Generation: %4d, Best: %4d'
              % (generation_count + 1, best_individual.fitness))
        if best_individual.fitness == garden.max_empty_cells:
            break
    else:
        print()
        print('Uncovered Empty Cells: %d' % (garden.max_empty_cells - best_individual.fitness))

    print()
    print(garden)
    print(best_individual.final_garden)

    return best_fitness_values


if __name__ == '__main__':
    if USE_ROULETTE_SELECTION == True:
        selection_method = "roulette"
    else:
        selection_method = "tournament"

    solve([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], selection_method)

    mean_fitness_values = []
    for i in range(1, len(best_fitness_values) + 1):
        mean_fitness = sum(best_fitness_values[:i]) / i
        mean_fitness_values.append(mean_fitness)

    # Plot the fitness progression graphs
    plt.figure(figsize=(10, 6))

    # Maximum fitness plot
    plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values, label='Maximum Fitness', color='b')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()

    # Average fitness plot
    plt.plot(range(1, len(mean_fitness_values) + 1), mean_fitness_values, label='Average Fitness', color='g')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()

    plt.title('Genetic Algorithm Fitness Progression')
    plt.tight_layout()

    plt.show()

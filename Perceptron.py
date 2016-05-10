import numpy as np
import random

class Perceptron:
    
    def __init__(self, 
                 population_size = 100,
                 number_of_generations = 100,
                 mutation_chance = 0.1,
                 crossover_chance = 0.8,
                 number_of_features = 3
                 ):
        self.w = None
        self.w0 = None
        self.population_size =population_size
        self.number_of_generations = number_of_generations
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        
        self.number_of_features = number_of_features

    def sort_by_best(self, population):
        pass

    def fit(self, x, y):
        population = self.create_initial_population()
        
        for generation in range(self.number_of_generations):
            self.select(population)
            self.crossover(population)
            self.mutate(population)
            
            best_individual = self.sort_by_best(population)[0]
            self.w0 = best_individual[0]
            self.w = best_individual[1:]

    def select(self, population):
        pop = list(population)
        pop.sort(key=lambda ind: self.fitness_function(ind))
        fits = [self.fitness_function(i) for i in pop]

        total = 0
        for i in range(len(fits)):
            total += fits[i]
            fits[i] = total

        selected = []
        for i in range(len(population)-1):
            n = random.randint(1, total)
            for j in range(len(fits)):
                if n <= fits[j]:
                    selected.append(pop[j])
                    break

        return selected
        
    def crossover(self, population, chance = 0.8):
        for i in range(0, len(population), 2):
            if i+1 == len(population): #população impar
                # o ultimo continua
                continue
            random.seed()
            if random.random() > chance: # 80% de chance de crossover
                continue

            ind1 = population[i]
            ind2 = population[i+1]

            point = random.choice(range(1, len(ind1)))

            population[i] = ind1[0:point] + ind2[point:]
            population[i+1] = ind2[0:point] + ind1[point:]
        
    def mutate(self, population):
        for i in range(len(population)):
            individual = population[i]
            if random.random() > self.mutation_chance: # mutation chance
                return individual
            mutation_position = random.choice(range(len(individual)))
            auxiliar_list = list(range(len(individual)))
            auxiliar_list.reverse()
            individual[mutation_position] = random.choice(range(auxiliar_list[mutation_position]+1))
    
            population[i] = individual
    
    def fitness_function(self, indv):
        erro_total = 0
        for i, example in enumerate(self.x):
            net = self.calcula_net(example)
            y_estimado = self.aplica_funcao_ativacao(net)
            erro = self.calcula_erro(y_estimado, self.y[i])
            erro_total += erro
            
        return 10000-erro_total #gambiarra

    def create_initial_population(self):
        population = []
        for i in range(self.population_size):
            random_individual = np.random.rand(1, self.number_of_features+1)[0]
            population.append(random_individual)
            
        return population

    def calcula_erro(self, y_estimado, y):
        return float(y) - y_estimado

    def calcula_net(self, xi):
        return np.dot(self.w, xi) + self.w0

    def aplica_funcao_ativacao(self, net):
        return 1/(1 + np.e ** -net)
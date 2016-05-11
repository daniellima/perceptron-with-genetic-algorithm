import numpy as np
import random

class Perceptron:
    
    def __init__(self, 
                 population_size = 5,
                 number_of_generations = 20,
                 mutation_chance = 0.5,
                 crossover_chance = 0.8,
                 number_of_features = 3
                 ):
        self.w = None
        self.w0 = None
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        
        self.number_of_features = number_of_features

    def sort_by_best(self, population, x, y):
        pop_with_fit = [(ind, self.fitness_function(ind, x, y)) for ind in population]
        sorted_population = sorted(pop_with_fit, key=lambda ind_fit: ind_fit[1])
        
        z = zip(*sorted_population)
        return list(next(z)), list(next(z)) # list de individuos e lista de suas fits

    def fit(self, x, y):
        population = self.create_initial_population()
        
        best = 0
        
        for generation in range(self.number_of_generations):
            self.maxError = 0
            population = self.select(population, x, y)
            population = self.crossover(population)
            population = self.mutate(population)
            
            pop, fits = self.sort_by_best(population, x, y)
            best = max(best, fits[-1])
            #print(fits[-1], "(best", best, ")", "Erro Maximo:", self.maxError)
            self.w0 = pop[-1][0]
            self.w = pop[-1][1:]

    def select(self, population, x, y):
        # Utiliza o metodo da roleta para selecionar.
        # E metodo escolhe aleatoriamente quem vai ser selecionado, mas dá mais chance para
        # quem tem o maior valor (fit)
        
        sorted_population, fits = self.sort_by_best(population, x, y)

        total = 0
        for i in range(len(sorted_population)):
            total += fits[i]
            fits[i] = total

        selected = []
        for i in range(len(sorted_population)):
            n = random.uniform(1, total)
            for j in range(len(fits)):
                if n <= fits[j]:
                    selected.append(sorted_population[j])
                    break

        return selected
        
    def crossover(self, population):
        # anda de dois em dois na população e faz um crossover entre os pares
        
        new_population = []
        for i in range(0, len(population), 2):
            if i+1 == len(population): #população impar
                new_population.append(population[i])
                continue
            random.seed()
            if random.random() > self.crossover_chance: # 80% de chance de crossover
                new_population.append(population[i])
                new_population.append(population[i+1])
                continue

            ind1 = population[i]
            ind2 = population[i+1]

            point = random.choice(range(1, len(ind1)))
            
            new_population.append(ind1[:point] + ind2[point:])
            new_population.append(ind2[:point] + ind1[point:])
            
        return new_population
        
    def mutate(self, population):
        # Escolhe uma posição e muda o valor dela aleatoriamente entre -5 e 5
        
        new_population = []
        
        for i in range(len(population)):
            individual = population[i]
            if random.random() > self.mutation_chance: # mutation chance
                new_population.append(individual)
                continue
            
            mutation_position = random.choice(range(len(individual)))
            individual[mutation_position] = random.random()* 10 - 5 # de -5 a 5
    
            new_population.append(individual)
            
        return new_population
    
    def fitness_function(self, individual, x, y):
        erro_total = 0
        for i, example in enumerate(x):
            y_estimado = self.evaluate(individual, example)
            erro = self.calcula_erro(y_estimado, y[i])
            erro_total += erro
        
        max_error = len(x) # o erro maximo para cada exemplo é 1
        return max_error-erro_total

    def create_initial_population(self):
        population = []
        for i in range(self.population_size):
            random_individual = np.random.rand(1, self.number_of_features+1)[0]
            population.append(list(random_individual))
            
        return population

    def evaluate(self, individual, example):
        return self.classify(self.aplica_funcao_ativacao(self.calcula_net(individual, example)))

    def calcula_erro(self, y_estimado, y):
        #erro = abs(float(y) - y_estimado)
        erro = abs(int(y) - y_estimado)
        self.maxError = max(self.maxError, erro)
        # if erro is self.maxError:
        #     print('Y estimado:', y_estimado, 'Classe', float(y))
        return erro

    def calcula_net(self, individual, example):
        self.net = (np.dot(individual[1:], example) + individual[0])
        self.individual = individual
        return self.net

    def aplica_funcao_ativacao(self, net):
        return 1/(1 + np.float64(np.e) ** -np.float64(net))
        
    def classify(self, x):
        if x > 0.5:
            return 1
        else:
            return 0
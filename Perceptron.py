import numpy as np
import random
import csv


class Perceptron:
    """
    Implementa o Perceptron utilizando o Delta como forma de calcular o erro
    """
    w = None
    w0 = None
    csv_train_path = None
    x = []
    y = []
    population_size = 100
    dim = 3
    generations = 100

    def __init__(self, csv_train_path):
        """
        csv_train_path indica o caminho onde o arquivo CSV está
        """
        self.csv_train_path = csv_train_path
        self.read_csv()
        self.fit()

    def fit(self):
        # quantidade de exemplos
        tamanho_x = len(self.x)

        self.create_random_w()

        population = self.create_initial_population()
        
        for generation in range(self.generations):
            population = self.select(population)
            self.crossover(population)
            
            for i in range(len(population)):
                population[i] = self.mutation(population[i])
            

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
        
    def mutation(self, individual, chance = 0.1):
        if random.random() > chance: # mutation chance
            return individual
        mutation_position = random.choice(range(len(individual)))
        auxiliar_list = list(range(len(individual)))
        auxiliar_list.reverse()
        individual[mutation_position] = random.choice(range(auxiliar_list[mutation_position]+1))

        #print("houve mutacao")

        return individual
    
    def fitness_function(self, indv):
        erro_total = 0
        for i, example in enumerate(self.x):
            net = self.calcula_net(example)
            y_estimado = self.aplica_funcao_ativacao(net)
            erro = self.calcula_erro(y_estimado, self.y[i])
            erro_total += erro
            
        return 10000-erro_total #gambiarra

    def create_random_w(self):
        # contruindo o w aleatoriamente
        w = np.random.rand(1, self.dim)
        w = w[0]
        w0 = np.random.rand(1, 1)
        w0 = w0[0]
        
        return [w0] + w

    def create_initial_population(self):
        return [self.create_random_w() for i in range(self.population_size)]

    def calcula_erro(self, y_estimado, y):
        return float(y) - y_estimado

    def calcula_net(self, xi):
        return np.dot(self.w, xi) + self.w0

    def aplica_funcao_ativacao(self, net):
        return 1/(1 + np.e ** -net)

    def read_csv(self):
        """
        Transforma os dados de um arquivo CSV na lista de exemplos x (com 3 features cada exemplo)
        e na lista y que diz qual a classe o exemplo pertence
        So funciona com exemplos que tem 3 features
        """
        with open(self.csv_train_path, 'r') as csv_file:
            exemples = csv.reader(csv_file, delimiter=';')

            for row in exemples:
                xi = []
                xi.append(float(row[0]))
                xi.append(float(row[1]))
                xi.append(float(row[2]))
                self.y.append(row[3])
                self.x.append(xi)

p = Perceptron('dataset.csv')
print(p.w)
print(p.w0)
from Perceptron import Perceptron
from DatasetLoader import DatasetLoader
from pprint import pprint

dataset = DatasetLoader('resources/dataset.csv')

p = Perceptron()

# pop = p.create_initial_population();
# t, f = p.sort_by_best(pop, dataset.x, dataset.y)
# print(t)
# print(f)

# pop = p.select(pop, dataset.x, dataset.y)
# [print(i) for i in pop]

# pop = p.crossover(pop)
# [print(i) for i in pop]

# t, f = p.sort_by_best(pop, dataset.x, dataset.y)
# print(t)
# print(f)


# pop = p.mutate(pop)
# [print(i) for i in pop]

# t, f = p.sort_by_best(pop, dataset.x, dataset.y)
# print(t)
# print(f)
# p.maxError = 0
# print(p.fitness_function([0.46510608158931277, 3.80882174, -1.61348909, -3.81282902], dataset.x, dataset.y))
# exit()
p.fit(dataset.x, dataset.y)
print("W:", p.w)
print("W0:", p.w0)


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

p.fit(dataset.x, dataset.y)


#print(p.w)
#print(p.w0)
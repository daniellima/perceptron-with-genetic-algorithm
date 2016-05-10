from Perceptron import Perceptron
from DatasetLoader import DatasetLoader

dataset = DatasetLoader('resources/dataset.csv')

p = Perceptron()
pop = p.create_initial_population();
print(p.sort_by_best(pop, dataset.x, dataset.y))

#p.fit(dataset.x, dataset.y)


print(p.w)
print(p.w0)
from Perceptron import Perceptron
from DatasetLoader import DatasetLoader

dataset = DatasetLoader('resources/dataset.csv')

p = Perceptron()
print(p.create_initial_population())

#p.fit(dataset.x, dataset.y)


print(p.w)
print(p.w0)
import csv

class DatasetLoader:
    
    def __init__(self, pathToDataset):
        self.x = []
        self.y = []
        
        with open(pathToDataset, 'r') as csv_file:
            examples = csv.reader(csv_file, delimiter=';')

            for row in examples:
                example = []
                example.append(float(row[0]))
                example.append(float(row[1]))
                example.append(float(row[2]))
                self.x.append(example)
                
                self.y.append(row[3])
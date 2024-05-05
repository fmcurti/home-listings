import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
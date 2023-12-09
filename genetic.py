from lib.field import *
from threading import Thread

class TruncationSelection:
    def __init__(self,k):
        self.k = k

    def select(self, y):
        p = np.argsort(y)
        return [p[np.random.randint(self.k, size=(2,))] for i in y]

class TournamentSelection:
    def __init__(self,k):
        self.k = k

    def select(self, y):
        return [[self.getparent(y), self.getparent(y)] for i in y]

    def getparent(self, y):
        p = np.random.permutation(len(y))
        y_np = np.array(y)
        return np.argmin(y_np[p[:self.k]])

class RouletteWheelSelection:
    def select(self):
        y = np.max(y) - np.array(y)
        normalize = y / np.linalg.norm(y, 1)
        return [np.random.choice(a=np.arange(0,len(y),1), p=normalize, size=2 ) for i in y]

class UniformCrossover:
    def crossover(self, a, b):
        child = a.copy()
        for i in range(len(a)):
            if np.random.random() < 0.5:
                child[i] = b[i]
        return child

class GaussianMutation:
    def __init__(self, sigma):
        self.sigma= sigma

    def mutate(self, child):
        return child + np.random.normal(size=len(child))*self.sigma

def genetic_algorithm(population, k_max, S, C, M):
    # def set_score(field, results, index):
    #     results[index] = field.generate_training_score()

    for k in range(k_max):
        print(f'Starting generation {k+1}')
        print('Getting training scores...')
        y = [HeuristicSearchField(weights=pop_val).generate_training_score() for pop_val in population]
        # y = [None for _ in range(len(population))]
        # threads = [None for _ in range(len(population))]

        # for i, pop_val in enumerate(population):
        #     threads[i] = Thread(target=set_score, args=(HeuristicSearchField(weights=pop_val), y, i))
        #     threads[i].start()

        # for i in range(len(population)):
        #     threads[i].join()
        
        print('Selecting parents...')
        parents = S.select(y)
        print('Breeding children...')
        children = [C.crossover(population[p[0]], population[p[1]] ) for  p in parents]
        print('Mutating children...')
        population = [M.mutate(child) for child in children]
        print(f'{k+1} generations finished\n')
    return population[np.argmin(HeuristicSearchField(weights=pop_val).generate_training_score() for pop_val in population)]

def main():
    m = 50
    k_max = 5
    population = [
        [random.uniform(-1, 1) for _ in range(7)]
        for _ in range(m)
    ]
    S = TruncationSelection(5)
    #S = RouletteWheelSelection()
    C = UniformCrossover()
    M = GaussianMutation(0.025)

    x = genetic_algorithm(population, k_max, S, C, M)
    print(x)

if __name__ == '__main__':
    main()
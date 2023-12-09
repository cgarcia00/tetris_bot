#!/usr/bin/python
from lib.field import HeuristicSearchField
from lib.tetromino import Tetromino
from multiprocessing import Pool
import random
from lib.field import HeuristicSearchField

"""
np.array([
    self.count_gaps(),             # Gap count
    np.mean(heights),              # Average height
    np.std(heights),               # Standard deviation of heights
    heights.max() - heights.min(), # Max height diff
    abs(ediff1d).max(),            # Max consecutive height diff
])
"""

NUM_TRIALS = 10

def get_score(field_weights, depth):
    # Create a new Field object in each process
    field = HeuristicSearchField(weights=field_weights)
    return field.generate_training_score(depth=depth, num_trials=1)

if __name__ == '__main__':
    field_weights1 = [ 7.60018953, 8.00419482,  1.88494599, -0.26924129,  6.31572023,  2.23826017, 4.64843085]
    # 100 10 weights
    field_weights2 = [24.71042388, 5.72456393, 15.55560536, 0.45457095, 10.81151936, 1.20316539, 6.26064719]
    # 25 3 weights
    field_weights3 = [26.15243376, 17.20669049, 17.70941622,  6.28345525, 13.27890416,  6.94699331, 4.25341613]
    # 100 10 weights round 2
    field_weights4=[18.33617477, 8.7924548, 10.0326507, -0.5597502, 10.35395414,  3.30750448, 10.00204665]

    field_weights5=[0.90897055,0.73683765,0.80142913,0.10518109,0.63010291,0.38493738,0.24763511]

    for field_weights in (field_weights5,):
        print(field_weights)            
        with Pool(8) as pool:
            results1 = pool.starmap(get_score, [(field_weights, 1)] * NUM_TRIALS)
            results2 = pool.starmap(get_score, [(field_weights, 2)] * NUM_TRIALS)
            results3 = pool.starmap(get_score, [(field_weights, 3)] * NUM_TRIALS)

        print("Depth 1 Results: " + f"{-sum(results1) / NUM_TRIALS}")
        print("Depth 2 Results: " + f"{-sum(results2) / NUM_TRIALS}")
        print("Depth 3 Results: " + f"{-sum(results3) / NUM_TRIALS}")

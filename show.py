#!/usr/bin/python

from collections import deque
from lib.field import Field, ForwardSearchField, HeuristicSearchField
from lib.tetromino import Tetromino
import random
import time

"""
return np.array([
    self.count_gaps(),             # Gap count
    np.mean(heights),              # Average height
    np.std(heights),               # Standard deviation of heights
    heights.max() - heights.min(), # Max height diff
    abs(ediff1d).max(),            # Max consecutive height diff
    heights.max(),                 # Max height
    -self.filled_lines(),          # Number of lines filled and cleared
])
"""

NUM_TETROMINO_TO_EXPOSE = 1

def get_random_tetromino():
    return Tetromino.create(random.choice(['I', 'O', 'T', 'S', 'Z', 'J', 'L']))

if __name__ == '__main__':
    # field = HeuristicSearchField(weights=[ 7.60018953, 8.00419482,  1.88494599, -0.26924129,  6.31572023,  2.23826017, 4.64843085])
    # 100 10 weights
    # field = HeuristicSearchField(weights=[24.71042388, 5.72456393, 15.55560536, 0.45457095, 10.81151936, 1.20316539, 6.26064719])
    # 25 3 weights
    # field = HeuristicSearchField(weights=[26.15243376, 17.20669049, 17.70941622,  6.28345525, 13.27890416,  6.94699331, 4.25341613])
    # 100 10 weights round 2
    #field = HeuristicSearchField(weights=[18.33617477, 8.7924548, 10.0326507, -0.5597502, 10.35395414,  3.30750448, 10.00204665])
    
    field = HeuristicSearchField(weights=[0.90897055,0.73683765,0.80142913,0.10518109,0.63010291,0.38493738,0.24763511])
    t = deque(get_random_tetromino() for _ in range(NUM_TETROMINO_TO_EXPOSE))
    count = 0
    while True:
        row, column, field, score = field.get_optimal_drop(t)
        if field == None:
            break
        print(field)
        print(field.get_scoring_vector())
        print(count)
        count += 1
        t.popleft()
        t.append(get_random_tetromino())

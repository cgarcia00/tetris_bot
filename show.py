#!/usr/bin/python

from collections import deque
from lib.field import Field, ForwardSearchField, HeuristicSearchField
from lib.tetromino import Tetromino
import random
import time

"""
np.array([
    self.count_gaps(),             # Gap count
    np.mean(heights),              # Average height
    np.std(heights),               # Standard deviation of heights
    heights.max() - heights.min(), # Max height diff
    abs(ediff1d).max(),            # Max consecutive height diff
])
"""

NUM_TETROMINO_TO_EXPOSE = 1

def get_random_tetromino():
    return Tetromino.create(random.choice(['I', 'O', 'T', 'S', 'Z', 'J', 'L']))

if __name__ == '__main__':
    # field = HeuristicSearchField(weights=[ 7.60018953, 8.00419482,  1.88494599, -0.26924129,  6.31572023,  2.23826017, 4.64843085])
    field = HeuristicSearchField(weights=[24.71042388, 5.72456393, 15.55560536, 0.45457095, 10.81151936, 1.20316539, 6.26064719])
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

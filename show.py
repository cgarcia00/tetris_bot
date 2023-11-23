#!/usr/bin/python

from lib.field import Field
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

if __name__ == '__main__':
    field = Field()
    current_tetromino = Tetromino.create(random.choice(['I', 'O', 'T', 'S', 'Z', 'J', 'L']))
    time.sleep(2)
    count = 0
    while True:
        next_tetromino = Tetromino.create(random.choice(['I', 'O', 'T', 'S', 'Z', 'J', 'L'])) 
        row, column, field, score = field.get_optimal_drop(current_tetromino, [1,1,1,1,1])
        if field == None:
            break
        print(count)
        print(field)
        count += 1
        current_tetromino = next_tetromino
        time.sleep(0.2)
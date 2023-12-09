#!/usr/bin/env python3

from lib.tetromino import Tetromino

import numpy as np
import math
import heapq
import random
from threading import Thread
from collections import deque

class Field():

    WIDTH = 10
    HEIGHT = 22
    SCORING_ELEMENTS = 6

    def __init__(self, state=None, weights=[1,1,1,1,1,1,1]):
        """
        Initializes a Tetris Field.
        Rows increase downward and columns increase to the right.
        """
        self.weights = weights
        if state is not None:
            self.state = np.array(state, dtype=np.uint8, copy=True)
        else:
            self.state = np.full((Field.HEIGHT, Field.WIDTH), 0, dtype=np.uint8)
        self._lines_cleared_ = 0

    def __lt__(self, other):
        return self.get_scoring_vector().dot(self.weights) > other.get_scoring_vector().dot(other.weights)

    def __str__(self):
        """
        Returns a string representation of the field.
        """
        bar = '   |' + ' '.join(map(str, range(Field.WIDTH))) + '|\n'
        mapped_field = np.vectorize(Tetromino.TYPES.__getitem__)(self.state)
        field = '\n'.join(['{:2d} |'.format(i) +
            ' '.join(row) + '|' for i, row in enumerate(mapped_field)])
        return bar + field + '\n' + bar

    def _test_tetromino_(self, tetromino, r_start, c_start):
        """
        Tests to see if a tetromino can be placed at the specified row and
        column. It performs the test with the top left corner of the
        tetromino at the specified row and column.
        """
        r_end, c_end = r_start + tetromino.height(), c_start + tetromino.width()
        if c_start < 0 or c_end > Field.WIDTH:
            return False
        if r_start < 0 or r_end > Field.HEIGHT:
            return False
        test_area = self.state[r_start:r_end, c_start:c_end]
        for s, t in zip(test_area.flat, tetromino.flat()):
            if s != 0 and t != 0:
                return False
        return True

    def _place_tetromino_(self, tetromino, r_start, c_start):
        """
        Place a tetromino at the specified row and column.
        The bottom left corner of the tetromino will be placed at the specified
        row and column. This function does not perform checks and will overwrite
        filled spaces in the field.
        """
        r_end, c_end = r_start + tetromino.height(), c_start + tetromino.width()
        if c_start < 0 or c_end > Field.WIDTH:
            return False
        if r_start < 0 or r_end > Field.HEIGHT:
            return False
        for tr, sr in enumerate(range(r_start, r_end)):
            for tc, sc, in enumerate(range(c_start, c_end)):
                if tetromino[tr][tc] != 0:
                    self.state[sr][sc] = tetromino[tr][tc]

    def _get_tetromino_drop_row_(self, tetromino, column):
        """
        Given a tetromino and a column, return the row that the tetromino
        would end up in if it were dropped in that column.
        Assumes the leftmost column of the tetromino will be aligned with the
        specified column.
        """
        if column < 0 or column + tetromino.width() > Field.WIDTH:
            return -1
        last_fit = -1
        for row in range(tetromino.height(), Field.HEIGHT):
            if self._test_tetromino_(tetromino, row, column):
                last_fit = row
            else:
                return last_fit
        return last_fit

    def _line_clear_(self):
        """
        Checks and removes all filled lines.
        """
        non_filled = np.array(
            [not row.all() and row.any() for row in self.state])
        if non_filled.any():
            tmp = self.state[non_filled]
            self.state.fill(0)
            self.state[Field.HEIGHT - tmp.shape[0]:] = tmp

    def generate_training_score(self, depth=1, num_trials=3):
        def get_random_tetromino():
            return Tetromino.create(random.choice(['I', 'O', 'T', 'S', 'Z', 'J', 'L']))
        
        result = 0
        for _ in range(num_trials):
            field = self.copy()
            t = deque(get_random_tetromino() for _ in range(depth))
            count = 0
            while True:
                row, column, field, score = field.get_optimal_drop(t)
                if field == None:
                    break
                count += 1
                t.popleft()
                t.append(get_random_tetromino())

            result += -count
        return result / num_trials
        # results = [None for _ in range(num_trials)]
        # threads = [None for _ in range(num_trials)]
        # def run_trial(results, index):
        #     field = self.copy()
        #     t = deque(get_random_tetromino() for _ in range(depth))
        #     count = 0
        #     while True:
        #         row, column, field, score = field.get_optimal_drop(t)
        #         if field == None:
        #             break
        #         count += 1
        #         t.popleft()
        #         t.append(get_random_tetromino())

        #     results[index] = -count

        # for i in range(num_trials):
        #     threads[i] = Thread(target=run_trial, args=(results, i))
        #     threads[i].start()

        # for i in range(num_trials):
        #     threads[i].join()

        # return sum(results) / num_trials

    def copy(self):
        """
        Returns a shallow copy of the field.
        """
        return Field(self.state, self.weights)

    def drop(self, tetromino, column):
        """
        Drops a tetromino in the specified column.
        The leftmost column of the tetromino will be aligned with the specified
        column.
        Returns the row it was dropped in for computations or -1 if a drop was
        unable to be computed.
        """
        assert isinstance(tetromino, Tetromino)
        row = self._get_tetromino_drop_row_(tetromino, column)
        if row == -1:
            return row
        self._place_tetromino_(tetromino, row, column)
        self._lines_cleared_ = sum(row.all() for row in self.state)
        self._line_clear_()
        return row

    def count_gaps(self):
        """
        Check each column one by one to make sure there are no gaps in the
        column.
        """
        # Cut off all the empty space above all the placed tetrominos
        top_indices = np.argmax(self.state.T != 0, axis = 1)
        # Count the number of gaps past the first filled space per column
        gaps = [np.count_nonzero(col[top:] == 0) for col, top in zip(
            self.state.T, top_indices)]
        return sum(gaps)

    def heights(self):
        """
        Return an array containing the heights of each column.
        """
        return Field.HEIGHT - np.argmax(self.state.T != 0, axis=1)

    def filled_lines(self):
        """
        Return the number of lines that were filled and cleared.
        """
        return self._lines_cleared_

    def get_scoring_vector(self):
        """
        Get a vector of values derived from the field used to score a tetromino
        placement.
        """
        heights = self.heights()
        ediff1d = np.ediff1d(heights)
        return np.array([
            self.count_gaps(),             # Gap count
            np.mean(heights),              # Average height
            np.std(heights),               # Standard deviation of heights
            heights.max() - heights.min(), # Max height diff
            abs(ediff1d).max(),            # Max consecutive height diff
            heights.max(),                 # Max height
            -self.filled_lines(),          # Number of lines filled and cleared
        ])
    
    def get_optimal_drop(self, t):
        """
        Given a slice of the upcoming tetrominos and a vector of scoring weights, this method
        calculates the best placement of the tetromino, scoring each placement
        with the weight vector.
        """
        tetromino = t[0]
        rotations = [
            tetromino,
            tetromino.copy().rotate_right(),
            tetromino.copy().flip(),
            tetromino.copy().rotate_left()
        ]
        best_row, best_column = None, None
        best_field = None
        best_drop_score = math.inf
        for rotation, tetromino_ in enumerate(rotations):
            for column in range(Field.WIDTH):
                f = self.copy()
                row = f.drop(tetromino_, column)
                if row == -1:
                    continue
                scoring_vector = f.get_scoring_vector()
                if self.weights is not None:
                    score = scoring_vector.dot(self.weights)
                else:
                    score = scoring_vector.sum()
                if score < best_drop_score:
                    best_drop_score = score
                    best_row, best_column = (row, column)
                    best_field = f
        return best_row, best_column, best_field, best_drop_score


class ForwardSearchField(Field):
    def __init__(self, state=None, weights=[1,1,1,1,1,1,1]):
        super().__init__(state, weights)

    def copy(self):
        """
        Returns a shallow copy of the field.
        """
        return ForwardSearchField(self.state, self.weights)
    
    # Returns a list of Fields after a valid move is played in t[0]
    def _get_drops_(self, tetromino):
        rotations = [
            tetromino,
            tetromino.copy().rotate_right(),
            tetromino.copy().flip(),
            tetromino.copy().rotate_left(),
        ]
        return [
            f
            for tetromino_ in rotations
            for column in range(Field.WIDTH)
            if (f := self.copy()).drop(tetromino_, column) != -1
        ]
    
    
    def get_optimal_drop(self, t):
        """
        Given a slice of the upcoming tetrominos and a vector of scoring weights, this method
        calculates the best placement of the tetromino, scoring each placement
        with the weight vector.
        """
        t = list(t)
        possible_fields = [self.copy()]
        if len(t) != 1:
            for tetromino in t[:-1]:
                possible_fields = [drop for field in possible_fields for drop in field._get_drops_(tetromino)]
        tetromino = t[-1]
        rotations = [
            tetromino,
            tetromino.copy().rotate_right(),
            tetromino.copy().flip(),
            tetromino.copy().rotate_left(),
        ]
        best_row, best_column = None, None
        best_field = None
        best_drop_score = math.inf

        for field in possible_fields:
            for rotation, tetromino_ in enumerate(rotations):
                for column in range(Field.WIDTH):
                    f = field.copy()
                    row = f.drop(tetromino_, column)
                    if row == -1:
                        continue
                    scoring_vector = f.get_scoring_vector()
                    if self.weights is not None:
                        score = scoring_vector.dot(self.weights)
                    else:
                        score = scoring_vector.sum()
                    if score < best_drop_score:
                        best_drop_score = score
                        best_row, best_column = (row, column)
                        best_field = f

        return best_row, best_column, best_field, best_drop_score


class HeuristicSearchField(Field):
    def __init__(self, state=None, weights=[1,1,1,1,1,1,1]):
        super().__init__(state, weights)

    def copy(self):
        """
        Returns a shallow copy of the field.
        """
        return HeuristicSearchField(self.state, self.weights)
    
    # Returns a list of Fields after a valid move is played in t[0]
    def _get_drops_(self, tetromino, k=5):
        rotations = [
            tetromino,
            tetromino.copy().rotate_right(),
            tetromino.copy().flip(),
            tetromino.copy().rotate_left(),
        ]

        k_best_fields = []
        for tetromino_ in rotations:
            for column in range(Field.WIDTH):
                f = self.copy()
                row = f.drop(tetromino_, column)
                if row == -1:
                    continue
                if len(k_best_fields) <= k:
                    heapq.heappush(k_best_fields, f)
                else:
                    heapq.heappushpop(k_best_fields, f)
        return k_best_fields
    
    def get_optimal_drop(self, t):
        """
        Given a slice of the upcoming tetrominos and a vector of scoring weights, this method
        calculates the best placement of the tetromino, scoring each placement
        with the weight vector.
        """
        t = list(t)
        possible_fields = [self.copy()]
        if len(t) != 1:
            for tetromino in t[:-1]:
                possible_fields = [drop for field in possible_fields for drop in field._get_drops_(tetromino)]
        tetromino = t[-1]
        rotations = [
            tetromino,
            tetromino.copy().rotate_right(),
            tetromino.copy().flip(),
            tetromino.copy().rotate_left(),
        ]
        best_row, best_column = None, None
        best_field = None
        best_drop_score = math.inf

        for field in possible_fields:
            for rotation, tetromino_ in enumerate(rotations):
                for column in range(Field.WIDTH):
                    f = field.copy()
                    row = f.drop(tetromino_, column)
                    if row == -1:
                        continue
                    scoring_vector = f.get_scoring_vector()
                    if self.weights is not None:
                        score = scoring_vector.dot(self.weights)
                    else:
                        score = scoring_vector.sum()
                    if score < best_drop_score:
                        best_drop_score = score
                        best_row, best_column = (row, column)
                        best_field = f

        return best_row, best_column, best_field, best_drop_score

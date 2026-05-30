import math

class MatchObject:
    def __init__(self):
        self.x = 0
        self.y = 0

    def get_dist(self, x, y):
        return math.sqrt(x**2 + y**2)
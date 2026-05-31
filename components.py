import math

class MatchObject:
    def __init__(self):
        self.x = 0
        self.y = 0

    def get_best_dist(self, options:list):
        min_ = min(*options, key=lambda p: math.hypot(p[0]-self.x, p[1]-self.y))

        return options.index(min_)
    
    def get_dist(self, position):
        return math.hypot(position[0]-self.x, position[1]-self.y)
    
    def set_position(self, position):
        self.x = position[0]
        self.y = position[1]

    def get_position(self):
        return [self.x, self.y]
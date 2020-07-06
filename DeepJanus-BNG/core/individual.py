from typing import Tuple

from numpy import mean

from core.member import Member


class Individual:
    def __init__(self, m1: Member, m2: Member):
        self.m1: Member = m1
        self.m2: Member = m2
        self.members_distance: float = None
        self.oob_ff: float = None
        self.seed: Member = None

    def clone(self) -> 'creator.base':
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()

    def mutate(self):
        raise NotImplemented()

    def distance(self, i2: 'Individual'):
        i1 = self
        a = i1.m1.distance(i2.m1)
        b = i1.m1.distance(i2.m2)
        c = i1.m2.distance(i2.m1)
        d = i1.m2.distance(i2.m2)

        dist = mean([min(a, b), min(c, d), min(a, c), min(b, d)])
        return dist

    def semantic_distance(self, i2: 'Individual'):
        raise NotImplemented()

    def members_by_sign(self) -> Tuple[Member, Member]:
        msg = 'in order to use this distance metrics you need to evaluate the member'
        assert self.m1.distance_to_boundary, msg
        assert self.m2.distance_to_boundary, msg

        result = self.members_by_distance_to_boundary()

        assert result[0].distance_to_boundary < 0, str(result[0].distance_to_boundary) + ' ' + str(self)
        assert result[1].distance_to_boundary >= 0, str(result[1].distance_to_boundary) + ' ' + str(self)
        return result

    def members_by_distance_to_boundary(self):
        def dist(m: Member):
            return m.distance_to_boundary

        result = sorted([self.m1, self.m2], key=dist)
        return tuple(result)
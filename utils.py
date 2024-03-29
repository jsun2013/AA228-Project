import random


def weighted_random_choice(choices):
    total = sum(w for c, w in choices.iteritems())
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices.iteritems():
        if upto + w >= r:
            return c
        upto += w
    raise Exception("Error in Weighted Random Choice")
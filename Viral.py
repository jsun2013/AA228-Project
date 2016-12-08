from __future__ import print_function
import snap
import numpy as np
import random
from Queue import PriorityQueue, Empty
import cvxpy as cvx
from collections import defaultdict
from copy import deepcopy

SUSCEPTIBLE, INFECTED, RECOVERED = range(3)

class StateException(Exception):
    pass


class State(object):
    def __init__(self, n):
        self.n = n
        self.num_susceptible = n
        self.num_infected = 0
        self.num_recovered = 0
        self.states = np.zeros(n, dtype=int)

    def infect(self, i):
        if self.states[i] == SUSCEPTIBLE:
            self.states[i] = INFECTED
            self.num_infected += 1
            self.num_susceptible -= 1
            return True
        return False

    def recover(self, i):
        if self.states[i] == INFECTED:
            self.states[i] = RECOVERED
            self.num_infected -= 1
            self.num_recovered += 1
            return True
        return False

    def copy(self, state2):
        self.num_susceptible = state2.num_susceptible
        self.num_infected = state2.num_infected
        self.num_recovered = state2.num_recovered
        self.states = deepcopy(state2.states)

    def __getitem__(self, i):
        return self.states[i]

    def __setitem__(self, key, value):
        if value == INFECTED:
            return self.infect(key)
        if value == RECOVERED:
            return self.recover(key)
        raise StateException("Unrecognized State: {}".format(value))

    def __hash__(self):
        return hash(tuple(self.states))

class ViralMarketing(object):

    def __init__(self, G, beta, delta=1.0, budget=10, horizon=50):
        self.G = G
        self.beta = beta
        self.delta = delta
        self.budget = budget
        self.parents = {}
        self.children = {}
        self.R = 1000
        self.horizon = horizon
        self.n = G.GetNodes()
        self.out_degs = np.zeros(self.n)
        self.max_deg = 0
        for node in G.Nodes():
            node_parents = set()
            self.parents[node.GetId()] = set(node.GetInEdges())
            self.children[node.GetId()] = set(node.GetOutEdges())
            self.out_degs[node.GetId()] = node.GetOutDeg()
            self.max_deg = max(self.max_deg, node.GetOutDeg())

    def MCTreeSearch(self, num_trajectories=200):
        T = set()
        Ns = defaultdict(int)
        Nsa = defaultdict(lambda: defaultdict(int))
        Q = defaultdict(lambda: defaultdict(float))

        def default_policy(s):
            a = set()
            if random.random() >= 0.25 or s.num_infected == self.n: # 25% of time, do nothing
                while True:
                    node_id = random.randint(0, self.n - 1)
                    if random.random() <= 0.5 or \
                            (s[node_id] == INFECTED and random.randint(0, self.max_deg) < self.out_degs[node_id]):
                        a.add(node_id)
                    if random.random() <= 0.5 or len(a) == 5:
                        break
            return tuple(sorted(list(a)))

        def reward(s, a):
            return s.num_infected - 2*len(a)

        def transition(s, a):
            sp = State(self.n)
            for target in a:
                sp.infect(target)
            for i, state in enumerate(s.states):
                if sp[i] == 0:
                    if state == 0:
                        for parent in self.parents[i]:
                            if (s[parent] == INFECTED) and random.random() <= self.beta:
                                sp.infect(i)
                                break
                    elif state == INFECTED and random.random() <= self.delta:
                        sp.infect(i)
                        sp.recover(i)
                    else:
                        sp.infect(i)
                        sp.recover(i)
            return sp

        def get_best_action(s):
            best_a = None
            max_val = -np.inf
            for a in Q[s]:
                temp = Q[s][a] + np.sqrt(np.log(Ns[s]) / Nsa[s][a])
                if temp > max_val:
                    max_val = temp
                    best_a = a
            if best_a is None:
                return default_policy(s)
            return best_a

        def Rollout(s, d, pi_0):
            if d == 0:
                return 0
            a = pi_0(s)
            r = reward(s, a)
            sp = transition(s, a)
            return r + Rollout(sp, d - 1, pi_0)

        def Simulate(s, d=self.horizon, pi_0=default_policy):
            if d == 0:
                return 0
            if s not in T:
                T.add(s)
                return Rollout(s, d, pi_0)
            a = get_best_action(s)
            r = reward(s, a)
            sp = transition(s, a)
            q = r + Simulate(sp, d - 1, pi_0)
            Nsa[s][a] += 1
            Ns[s] += 1
            Q[s][a] += (q - Q[s][a]) / Nsa[s][a]
            return q

        def SelectAction(s, d=self.horizon):
            num_iterations = 100
            for _ in xrange(num_iterations):
                Simulate(s, d)
            return get_best_action(s)

        actions = []
        rewards = []
        s = State(self.n)
        for t in xrange(self.horizon):
            a = SelectAction(s)
            rewards.append(reward(s, a))
            s = transition(s, a)
            actions.append(a)
            if s.num_susceptible == 0:
                break
        return rewards, actions, s

    def single_horizon(self, seed, T=100):
        episode_rewards = np.zeros(self.R)
        for sim in xrange(self.R):
            states = {}
            reward = 0.0
            for i in xrange(self.n):
                states[i] = 0
            for i in seed:
                states[i] = 1
                reward -= 1


            num_infected = 0
            for _ in xrange(T):
                next_states = {}
                for nid in xrange(self.n):
                    next_states[nid] = states[nid]
                    if states[nid] == 0:
                        for parent in self.parents[nid]:
                            if states[parent] == 1 and random.random() <= self.beta:
                                next_states[nid] = 1
                                reward += 0.5
                                break
                    elif states[nid] == 1:
                        if random.random() <= self.delta:
                            next_states[nid] = 2
                        num_infected += 1
                if num_infected == 0:
                    break
                for nid in states:
                    states[nid] = next_states[nid]
            episode_rewards[sim] = reward
        return episode_rewards

n = 50
G_pa = snap.GenPrefAttach(n, 5)
viral_pa = ViralMarketing(G_pa, beta=0.1, delta=1)
test_r, test_a, test_s  = viral_pa.MCTreeSearch()


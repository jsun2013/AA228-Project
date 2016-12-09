from __future__ import print_function
import snap
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy
import utils
from time import time

SUSCEPTIBLE, INFECTED, RECOVERED = range(3)


class StateException(Exception):
    pass


class State(object):
    def __init__(self, n, t, budget=None):
        self.n = n
        self.num_susceptible = n
        self.num_infected = 0
        self.num_recovered = 0
        self.states = np.zeros(n, dtype=int)
        self.t = t
        self.budget = budget

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
        self.n = state2.n
        self.t = state2.t
        self.budget = state2.budget

    def __getitem__(self, i):
        return self.states[i]

    def __setitem__(self, key, value):
        if value == INFECTED:
            return self.infect(key)
        if value == RECOVERED:
            return self.recover(key)
        raise StateException("Unrecognized State: {}".format(value))

    def __hash__(self):
        test = list(self.states)
        test.append(self.t)
        test.append(self.budget)
        return hash(tuple(test))

    def __eq__(self, other):
        return self.t == other.t and np.all(self.states == other.states) and self.budget == other.budget


class ViralMarketing(object):

    def __init__(self, G, beta, delta=1.0, budget=10, horizon=20):
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
            self.parents[node.GetId()] = set(node.GetInEdges())
            self.children[node.GetId()] = set(node.GetOutEdges())
            self.out_degs[node.GetId()] = node.GetOutDeg()
            self.max_deg = max(self.max_deg, node.GetOutDeg())

    @staticmethod
    def get_reward(s, a):
        return s.num_infected - 2 * len(a)

    def transition(self, s, a):
        sp = State(self.n, s.t)
        for target in a:
            if s[target] == SUSCEPTIBLE:
                sp.infect(target)
        for i, state in enumerate(s.states):
            if sp[i] == 0:
                if state == SUSCEPTIBLE:
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
        sp.t -= 1
        return sp

    def MCTreeSearch(self, k=10, kp=10, alpha=0.5, alphap=0.5):
        T = set()
        actions_tried = defaultdict(set)
        Ns = defaultdict(int)
        Nsa = defaultdict(lambda: defaultdict(int))
        Nsas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        V = defaultdict(lambda: defaultdict(set))
        Q = defaultdict(lambda: defaultdict(float))

        def default_policy(s):
            a = set()
            if tuple() not in Nsa[s] and s.t < self.horizon:
                return tuple()
            if random.random() <= 0.2 or s.num_susceptible == self.n:  # 90% of time, do nothing
                for _ in xrange(5):
                    node_id = random.randint(0, self.n - 1)
                    if random.random() <= 0.5 or \
                            (s[node_id] == INFECTED and random.randint(0, self.max_deg) < self.out_degs[node_id]):
                        a.add(node_id)
            return tuple(sorted(list(a)))

        def explore_actions(s):
            best_a = None
            max_val = -np.inf
            for a in Q[s]:
                temp = Q[s][a] + np.sqrt(np.log(Ns[s]) / Nsa[s][a])
                if temp > max_val:
                    max_val = temp
                    best_a = a
            return best_a

        def get_best_action(s):
            best_a = None
            max_val = -np.inf
            for a, val in Q[s].iteritems():
                if val > max_val:
                    max_val = val
                    best_a = a
            return best_a

        def Rollout(s, d, pi_0):
            if d == 0:
                return 0
            a = pi_0(s)
            r = self.get_reward(s, a)
            sp = self.transition(s, a)
            return r + Rollout(sp, d - 1, pi_0)

        def Simulate(s, d=self.horizon, pi_0=default_policy):
            if d == 0:
                return 0
            if s not in T:
                T.add(s)
                Ns[s] = 0
                return Rollout(s, s.t, pi_0)
            Ns[s] += 1
            if len(actions_tried[s]) < k*Ns[s]**alpha:
                a = None
                while True:
                    a = pi_0(s)
                    if a not in Nsa[s]:
                        break
                Nsa[s][a] = 1
                Ns[s] += 1
                Q[s][a] = -2*len(a)
                V[s][a] = set()
                actions_tried[s].add(a)
            a = explore_actions(s)

            if len(V[s][a]) < kp*Nsa[s][a]**alphap:
                r = self.get_reward(s, a)
                sp = self.transition(s, a)
                V[s][a].add(sp)
                Nsas[s][a][sp] += 1
            else:
                sp = utils.weighted_random_choice(Nsas[s][a])
                r = self.get_reward(s, a)
                Nsas[s][a][sp] += 1
            q = r + Simulate(sp, d - 1, pi_0)
            Nsa[s][a] += 1
            Ns[s] += 1
            Q[s][a] += (q - Q[s][a]) / Nsa[s][a]
            return q

        def SelectAction(s):
            num_iterations = 100
            for _ in xrange(num_iterations):
                Simulate(s, s.t)
            return get_best_action(s)

        actions = []
        rewards = 0
        s = State(self.n, self.horizon)
        for t in xrange(self.horizon):
            a = SelectAction(s)
            rewards += self.get_reward(s, a)
            s = self.transition(s, a)
            actions.append(a)
            if s.num_susceptible == 0:
                break
        return rewards, actions, s

    def degree_discount(self, s, k=5):
        assert k < self.n
        S = np.zeros(self.n, dtype=bool)
        S_bar = np.logical_not(S)
        t = np.zeros(self.n)
        ddv = np.zeros(self.n)
        ddv_temp = np.zeros(self.n)

        for i in xrange(self.n):
            ddv[i] = self.out_degs[i]
            ddv_temp[i] = self.out_degs[i]
            if s[i] == INFECTED or s[i] == RECOVERED:
                S[i] = True
                S_bar[i] = False

        for i, state in enumerate(s.states):
            if state == INFECTED or state == RECOVERED:
                for v in self.children[i]:
                    if not S[v]:
                        t[v] += 1
                        ddv[v] = self.out_degs[v] - 2 * t[v] - (self.out_degs[v] - t[v]) * t[v] * self.beta

        for i in xrange(k):
            ddv_temp[S] = -np.inf
            u = np.argmax(ddv_temp)
            S[u] = True
            S_bar[u] = False
            for v in self.children[u]:
                if not S[v]:
                    t[v] += 1
                    ddv[v] = self.out_degs[v] - 2 * t[v] - (self.out_degs[v] - t[v]) * t[v] * self.beta
        out = set()
        for i, val in enumerate(S):
            if val and s[i] == SUSCEPTIBLE:
                out.add(i)

        return out

    def HybridSearch(self, k=10, kp=10, alpha=0.5, alphap=0.5):
        T = set()
        actions_tried = defaultdict(set)
        Ns = defaultdict(int)
        Nsa = defaultdict(lambda: defaultdict(int))
        Nsas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        V = defaultdict(lambda: defaultdict(set))
        Q = defaultdict(lambda: defaultdict(float))

        def get_possible_actions(s, k=5):
            if s.t == 0:
                return [tuple()]
            choices = sorted(self.degree_discount(s, k))
            actions = [tuple()]
            for i in xrange(k):
                for j in xrange(i+1, k+1):
                    actions.append(tuple(choices[i:j]))
            return actions

        def smart_policy(s):
            if random.random() <= 0.1:
                return tuple()
            choices = self.degree_discount(s, 10)
            num = random.randint(0,min(len(choices), 3))
            a = random.sample(choices, num)
            return tuple(sorted(a))

        def explore_actions(s):
            best_a = None
            max_val = -np.inf
            for a in Q[s]:
                temp = Q[s][a] + np.sqrt(np.log(Ns[s]) / Nsa[s][a])
                if temp > max_val:
                    max_val = temp
                    best_a = a
            return best_a

        def get_best_action(s):
            best_a = None
            max_val = -np.inf
            for a, val in Q[s].iteritems():
                if val > max_val:
                    max_val = val
                    best_a = a
            return best_a

        def Rollout(s, d, pi_0):
            if d == 0:
                return 0
            # a = pi_0(s)
            a = ()
            r = self.get_reward(s, a)
            sp = self.transition(s, a)
            return r + Rollout(sp, d - 1, pi_0)

        def Simulate(s, d=self.horizon, pi_0=smart_policy):
            if d == 0:
                return 0
            if s not in T:
                for a in get_possible_actions(s):
                    Nsa[s][a] = 1
                    Q[s][a] = -2*len(a)
                T.add(s)
                return Rollout(s, s.t, pi_0)
            Ns[s] += 1
            a = explore_actions(s)
            sp = self.transition(s, a)
            r = self.get_reward(s, a)
            q = r + Simulate(sp, d - 1, pi_0)
            Nsa[s][a] += 1
            Ns[s] += 1
            Q[s][a] += (q - Q[s][a]) / Nsa[s][a]
            return q

        def SelectAction(s):
            num_iterations = 100
            for _ in xrange(num_iterations):
                Simulate(s, s.t)
            return get_best_action(s)

        actions = []
        rewards = 0
        s = State(self.n, self.horizon)
        for t in xrange(self.horizon):
            a = SelectAction(s)
            rewards += self.get_reward(s, a)
            s = self.transition(s, a)
            actions.append(a)
            if s.num_susceptible == 0:
                break
        return rewards, actions, s

    def greedy(self, k = 5):
        s0 = State(self.n, self.horizon)
        seed_set = self.degree_discount(s0, k)
        return self.single_horizon(seed_set)

    def single_horizon(self, seed):
        s = State(self.n, t=self.horizon)
        reward = self.get_reward(s, seed)
        for i in seed:
            s.infect(i)
        s.t -= 1

        for _ in xrange(1, self.horizon):
            sp = State(self.n, s.t)
            sp.copy(s)
            for nid in xrange(self.n):
                if s[nid] == SUSCEPTIBLE:
                    for parent in self.parents[nid]:
                        if s[parent] == INFECTED and random.random() <= self.beta:
                            sp.infect(nid)
                            break
                elif s[nid] == INFECTED:
                    if random.random() <= self.delta:
                        sp.recover(nid)
            sp.t -= 1
            s.copy(sp)
            reward += self.get_reward(s, tuple())
            if s.num_infected == 0:
                break
        return reward, tuple(sorted(seed)), s

def main(G, name):
    viral = ViralMarketing(G, beta=0.1, delta=1)
    num_trials = 200

    rewards = []
    actions = []
    times = []
    for i in xrange(num_trials):
        start_time = time()
        test_r, test_a, _ = viral.greedy()
        times.append(time() - start_time)
        rewards.append(test_r)
        actions.append(test_a)
    print("Done")

    with open("{}-greedy.txt".format(name), "w+") as f:
        f.write("Reward,Time,Seed\n")
        for i in xrange(len(rewards)):
            f.write(str(rewards[i]))
            f.write(",")
            f.write(str(times[i]))
            f.write(",")
            f.write(str(actions[i]))
            f.write("\n")
    rewards = []
    actions = []
    times = []
    for i in xrange(num_trials):
        start_time = time()
        test_r, test_a, _ = viral.MCTreeSearch()
        times.append(time() - start_time)
        rewards.append(test_r)
        actions.append(test_a)
    print("Done")
    with open("{}-MCT.txt".format(name), "w+") as f:
        f.write("Reward,Time,Actions\n")
        for i in xrange(len(rewards)):
            f.write(str(rewards[i]))
            f.write(",")
            f.write(str(times[i]))
            f.write(",")
            f.write(str(actions[i]))
            f.write("\n")

    # rewards = []
    # actions = []
    # times = []
    # for i in xrange(num_trials):
    #     start_time = time()
    #     test_r, test_a, _ = viral.HybridSearch()
    #     times.append(time() - start_time)
    #     rewards.append(test_r)
    #     actions.append(test_a)
    # print("Done")
    #
    # with open("{}-hybrid.txt".format(name), "w+") as f:
    #     f.write("Reward,Time,Seed\n")
    #     for i in xrange(len(rewards)):
    #         f.write(str(rewards[i]))
    #         f.write(",")
    #         f.write(str(times[i]))
    #         f.write(",")
    #         f.write(str(actions[i]))
    #         f.write("\n")

    return viral


n = 100
# G_rmat = snap.GenRMat(n, 5*n, 0.55, 0.228, 0.212)
# G_er = snap.GenRndGnm(snap.PNGraph, n, 5*n)
# G_pa = snap.GenPrefAttach(n, 5)
# snap.SaveEdgeList(G_rmat, 'G_rmat.txt')
# snap.SaveEdgeList(G_er, 'G_er.txt')
# snap.SaveEdgeList(G_pa, 'G_pa.txt')

G_rmat = snap.LoadEdgeList(snap.PNGraph, "G_rmat.txt")
G_er = snap.LoadEdgeList(snap.PNGraph, "G_er.txt")
G_pa = snap.LoadEdgeList(snap.PNGraph, "G_pa.txt")

viral_pa = main(G_pa, "PA")
viral_rmat = main(G_rmat, "RMAT")
viral_er = main(G_er, "ER")

viral_pa = ViralMarketing(G_pa, beta=0.1, delta=1)
start_time = time()
test_r, test_a, test_s = viral_pa.HybridSearch()
print(time()-start_time)

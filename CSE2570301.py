import copy
import random

import numpy as np

initial_state = [[0 for i in range(3)] for j in range(3)]
check_seq = [  # row
    [[0, 0], [0, 1], [0, 2]],
    [[1, 0], [1, 1], [1, 2]],
    [[2, 0], [2, 1], [2, 2]],
    # column
    [[0, 0], [1, 0], [2, 0]],
    [[0, 1], [1, 1], [2, 1]],
    [[0, 2], [1, 2], [2, 2]],
    # diagonal
    [[0, 0], [1, 1], [2, 2]],
    [[0, 2], [1, 1], [2, 0]], ]

dis = 0.9
state_dict = {}
c = 1.


class Node:
    def __init__(self, state=initial_state, turn=-1):
        self.state = copy.deepcopy(state)
        self.turn = turn
        self.ind = self.index()

        self.son = []
        self.sonlist = []

        terminal = self.terminal()
        self.term = terminal
        if terminal == 2:
            self.R = c
        elif terminal == -1:
            self.R = -10.
        elif terminal == 1:
            #print("win")
            #print(self.turn)
            #print(np.array(self.state))
            self.R = 10.
        elif terminal == 0:
            self.R = 0.
        self.visit = False

    def terminal(self):  # 1--I win; -1--opponent win; 0--draw; 2--continue
        # win
        for i in range(len(check_seq)):
            consist = True
            seq = check_seq[i]
            type = self.state[seq[0][0]][seq[0][1]]
            if type == -1 or type == 1:
                for j in range(len(seq)):
                    if self.state[seq[j][0]][seq[j][1]] != type:
                        consist = False
                        break
                if consist:
                    return type

        # empty or continue?
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return 2

        # no more space---draw
        return 0

    def index(self):  # +turn
        sum = self.turn
        for i in range(3):
            for j in range(3):
                sum = sum * 3 + self.state[i][j]
        return sum


class AI:
    def __init__(self, state=initial_state, dis=0.9):
        self.root = Node(state)
        self.dis = dis

    def node_indict(self,state,turn):
        sum = turn
        for i in range(3):
            for j in range(3):
                sum = sum * 3 + state[i][j]
        if sum in state_dict.keys():
            return state_dict[sum]
        else:
            node=Node(state,turn)
            state_dict[sum] = node
            return node

    def buildtree(self, node=None):
        if node == None:
            node = self.root

        # already count
        if node.visit:
            return
        else:
            node.visit = True

        # opponent turn
        if node.turn == -1:
            son_turn = 1
            if node.term == 1:  # I already win, remain
                son_state = copy.deepcopy(node.state)
                son = self.node_indict(son_state, son_turn)
                node.son.append(son)
                #node.sonlist.append(None)
            elif node.term == 2:  # opponent take action
                for i in range(3):
                    for j in range(3):
                        if node.state[i][j] == 0:
                            son_state = copy.deepcopy(node.state)
                            son_state[i][j] = node.turn
                            node.sonlist.append([i, j])
                            son = self.node_indict(son_state, son_turn)
                            node.son.append(son)
                            self.buildtree(son)
                total = len(node.sonlist)
                node.prob = 1. / total
            else:
                print("error!")

        # my turn
        elif node.turn == 1:
            son_turn = -1
            if node.term != 2:  # cannot continue
                return
            else:
                for i in range(3):
                    for j in range(3):
                        if node.state[i][j] == 0:
                            son_state = copy.deepcopy(node.state)
                            son_state[i][j] = node.turn
                            node.sonlist.append([i, j])
                            son = self.node_indict(son_state, son_turn)
                            node.son.append(son)
                            self.buildtree(son)


    def random_trace(self, node=None):
        if node == None:
            node = self.root

        trace = [node]
        thnode = node
        while len(thnode.son) != 0: # sonlist---maybe the last one(R=10) did not show up
            index = random.randint(0, len(thnode.son) - 1)
            thnode = thnode.son[index]
            trace.append(thnode)

        # compute Reward to Go
        GoR = [None for t in range(len(trace))]
        nowG = 0  # the last one
        for t in reversed(range(len(trace))):
            if trace[t].turn == 1:
                nowG = self.dis * nowG + trace[t].R
                GoR[t] = nowG

        for t in range(1, len(trace)):
            if trace[t].turn == 1:
                print("--->")
                print("reward-to-go___", GoR[t])
                print("reward___",trace[t].R)
                print(np.array(trace[t].state))
        return trace

    def MDP(self,T=100):
        # initialize
        for key in state_dict.keys():
            node = state_dict[key]
            if node.turn == 1:  # my turn, optimize
                if len(node.son) == 0:  # terminal
                    node.val = node.R
                else:  # initialize
                    node.val = 0
            elif node.turn == -1 and len(node.son) == 0:  # terminal
                node.val = node.R

        # optimize
        for t in range(T):
            # Q-value
            for key in state_dict.keys():
                node = state_dict[key]
                if node.turn == -1:
                    if len(node.son) > 0:
                        sum = 0
                        for i in range(len(node.son)):
                            sum += node.son[i].val
                        node.val = sum / len(node.son)

            # val-value
            for key in state_dict.keys():
                node = state_dict[key]
                if node.turn == 1:
                    if len(node.son) > 0:
                        max = node.son[0].val
                        for i in range(len(node.son)):
                            if max < node.son[i].val:
                                max = node.son[i].val
                        node.val = node.R + self.dis * max


"""
#node= Node([[1,-1,1],[-1,-1,1],[1,1,-1]])
node=Node()
print(node.state)
print(node.terminal())
"""

ai = AI()
print("build tree")
ai.buildtree()
print("print random trace")
trace_list = []
for t in range(5):
    print("==================", t + 1, "======================")
    trace_list.append(ai.random_trace())
    print()

print()
print()
T=100
print("MDP__",T,"_times")
ai.MDP()
"""
for key in state_dict.keys():
    print(key)
    print(np.array(state_dict[key].state))
    print(state_dict[key].val)
"""
print("==================state value======================")
for t in range(5):
    print("==================", t + 1, "======================")
    trace = trace_list[t]
    for t in range(len(trace)):
        if trace[t].turn == 1:
            print("--->")
            print("state value___", trace[t].val)
            print(np.array(trace[t].state))
    print()


print()
print()
T=1000
print("MDP__",T,"_times")
ai.MDP()
"""
for key in state_dict.keys():
    print(key)
    print(np.array(state_dict[key].state))
    print(state_dict[key].val)
"""
print("==================state value======================")
for t in range(5):
    print("==================", t + 1, "======================")
    trace = trace_list[t]
    for t in range(len(trace)):
        if trace[t].turn == 1:
            print("--->")
            print("state value___", trace[t].val)
            print(np.array(trace[t].state))
    print()


print("==================Q-value======================")
initialchoice = [
    [[0, -1, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, -1, 0], [0, 0, 0]]
]
for t in range(len(initialchoice)):
    print("==================", t + 1, "======================")
    state = initialchoice[t]
    node = state_dict[Node(state,1).index()]
    print("initial state____")
    print(np.array(node.state))
    for i in range(len(node.son)):
        print("potential action____", node.sonlist[i])
        print("Q-value___", (node.R + dis * node.son[i].val))
        print("result state___")
        print("turn___", node.son[i].turn)
        print(np.array(node.son[i].state))
    print()
    print()

print()
print()
print("==================Q-value comparison======================")
gamestate = [[-1, 0, -1], [1, 1, 0], [-1, 0, 0]]
gameturn = 1
reward_list = [5, -3, 0]
for t in range(len(reward_list)):
    print("=============  c=", reward_list[t], "  ===============")
    if reward_list[t] != c:  # need to rebuild
        print("rebuild tree...")
        c = reward_list[t]
        state_dict.clear()
        ai = AI()
        ai.buildtree()
        ai.MDP()
    node = state_dict[Node(gamestate,1).index()]
    print("initial state____")
    print(np.array(node.state))
    max = 0
    index = 0
    for i in range(len(node.son)):
        print("potential action____", node.sonlist[i])
        Qvalue = node.R + dis * node.son[i].val
        print("Q-value___", Qvalue)
        print("result state___")
        print("turn___", node.son[i].turn)
        print(np.array(node.son[i].state))
        if max < Qvalue:
            max = Qvalue
            index = i
    print("optimal choice______", node.sonlist[index])
    print()
    print()

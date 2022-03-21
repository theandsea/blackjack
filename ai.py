import copy
import random

import numpy as np

from game import Game, states, WIN_STATE, LOSE_STATE

HIT = 0
STAND = 1
DISCOUNT = 0.95  # This is the gamma value for all value calculations


class Agent:
    def __init__(self):

        # For MC values
        self.MC_values = {}  # Dictionary: Store the MC value of each state
        self.S_MC = {}  # Dictionary: Store the sum of returns in each state
        self.N_MC = {}  # Dictionary: Store the number of samples of each state
        # MC_values should be equal to S_MC divided by N_MC on each state (important for passing tests)

        # For TD values
        self.TD_values = {}  # Dictionary storing the TD value of each state
        self.N_TD = {}  # Dictionary: Store the number of samples of each state

        # For Q-learning values
        self.Q_values = {}  # Dictionary storing the Q-Learning value of each state and action
        self.N_Q = {}  # Dictionary: Store the number of samples of each state

        # Initialization of the values
        for s in states:
            self.MC_values[s] = 0
            self.S_MC[s] = 0
            self.N_MC[s] = 0
            self.TD_values[s] = 0
            self.N_TD[s] = 0
            self.Q_values[s] = [0, 0]  # First element is the Q value of "Hit", second element is the Q value of "Stand"
            self.N_Q[s] = 0
        # NOTE: see the comment of `init_cards()` method in `game.py` for description of game state       
        self.simulator = Game()

    # NOTE: do not modify
    # This is the policy for MC and TD learning. 
    @staticmethod
    def default_policy(state):
        user_sum = state[0]
        user_A_active = state[1]
        actual_user_sum = user_sum + user_A_active * 10
        if actual_user_sum < 14:
            return 0
        else:
            return 1

    # NOTE: do not modify
    # This is the fixed learning rate for TD and Q learning. 
    @staticmethod
    def alpha(n):
        return 10.0 / (9 + n)

    def MC_run(self, num_simulation, tester=False):
        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):
            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "MC")
            self.simulator.reset()  # Restart the simulator

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: Go through game.py file and figure out which functions will be useful
            # Useful variables:
            #     - DISCOUNT
            #     - self.MC_values     (read comments in self.__init__)
            # remember to update self.MC_values, self.S_MC, self.N_MC for the autograder!
            # print(simulation)
            game = self.simulator
            initial = game.state
            factor = DISCOUNT  # stand is also a step
            # print(game.state)
            while (not self.default_policy(game.state)) and (not game.game_over()):
                game.act_hit()
                # print("hit--->",game.state)
                factor *= DISCOUNT
            # print("stand")
            game.act_stand()
            # print("---->",game.state)

            # reward
            if game.state == WIN_STATE:
                reward = 1 * factor
            elif game.state == LOSE_STATE:
                reward = (-1) * factor

            # update...actually no need to decise whether in
            if initial in self.S_MC:
                self.S_MC[initial] += reward
                self.N_MC[initial] += 1
            else:
                self.S_MC[initial] = reward
                self.N_MC[initial] = 1
            # update
            self.MC_values[initial] = self.S_MC[initial] / self.N_MC[initial]
            # print(simulation)
            # print(initial, self.MC_values[initial])

    def TD_run(self, num_simulation, tester=False):
        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):
            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "TD")
            self.simulator.reset()

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: Go through game.py file and figure out which functions will be useful
            # Hint: The learning rate alpha is given by "self.alpha(...)"
            # Useful variables/funcs:
            #     - DISCOUNT
            #     - self.TD_values  (read comments in self.__init__)
            # remember to update self.TD_values and self.N_TD for the autograder!
            game = self.simulator
            initial = game.state
            trace = [initial]
            while (not self.default_policy(game.state)) and (not game.game_over()):
                game.act_hit()
                trace.append(game.state)
            game.act_stand()

            trace.append(game.state)  # last one
            # reward--->last one, deterministic
            if game.state == WIN_STATE:
                reward = 1
            elif game.state == LOSE_STATE:
                reward = (-1)

            # update
            for t in range(len(trace) - 1):
                the_state = trace[t]
                self.N_TD[the_state] += 1  # R=0 for other state
                self.TD_values[the_state] += self.alpha(self.N_TD[the_state]) * (
                        DISCOUNT * self.TD_values[trace[t + 1]] - self.TD_values[the_state])
            # last one , deterministic

            self.N_TD[trace[len(trace) - 1]] += 1
            self.TD_values[trace[len(trace) - 1]] = reward

    def Q_run(self, num_simulation, tester=False):
        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):
            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "Q")
            self.simulator.reset()

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: Go through game.py file and figure out which functions will be useful
            # Hint: The learning rate alpha is given by "self.alpha(...)"
            # Hint: Implement epsilon-greedy method in "self.pick_action(...)"
            # Useful variables:
            #     - DISCOUNT
            #     - self.Q_values  (read comments in self.__init__)
            # remember to update self.Q_values, self.N_Q for the autograder!
            game = self.simulator
            initial = game.state
            trace = [initial]
            while not game.game_over():  # no policy
                father = copy.deepcopy(game.state)
                action = self.pick_action(game.state, 0.4)
                if action == 0:  # hit
                    game.act_hit()
                else:  # stand
                    game.act_stand()

                son = game.state
                # update after the son generated... must not be terminal---R=0
                self.N_Q[father] += 1
                if self.Q_values[son][0] > self.Q_values[son][1]:
                    maxQ = self.Q_values[son][0]
                else:
                    maxQ = self.Q_values[son][1]
                self.Q_values[father][action] += self.alpha(self.N_Q[father]) * (
                        DISCOUNT * maxQ - self.Q_values[father][action])

            son = game.state
            # reward--->last one, deterministic
            if game.state == WIN_STATE:
                self.N_Q[son] += 1
                self.Q_values[son] = [1, 1]
            elif game.state == LOSE_STATE:
                self.N_Q[son] += 1
                self.Q_values[son] = [-1, -1]

    def pick_action(self, s, epsilon):
        # TODO: Replace the following random return value with the epsilon-greedy strategy
        # return random.randint(0, 1)
        if random.random() < epsilon:  # explore random
            return random.randint(0, 1)
        else:  # exploit the best
            if self.Q_values[s][0] > self.Q_values[s][1]:
                return 0  # hit
            else:
                return 1  # stand

    # Note: do not modify
    def autoplay_decision(self, state):
        hitQ, standQ = self.Q_values[state][HIT], self.Q_values[state][STAND]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT  # Before Q-learning takes effect, just always HIT

    # NOTE: do not modify
    def save(self, filename):
        with open(filename, "w") as file:
            for table in [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q]:
                for key in table:
                    key_str = str(key).replace(" ", "")
                    entry_str = str(table[key]).replace(" ", "")
                    file.write(f"{key_str} {entry_str}\n")
                file.write("\n")

    # NOTE: do not modify
    def load(self, filename):
        with open(filename) as file:
            text = file.read()
            MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text, _ = text.split(
                "\n\n")

            def extract_key(key_str):
                return tuple([int(x) for x in key_str[1:-1].split(",")])

            for table, text in zip(
                    [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q],
                    [MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text]
            ):
                for line in text.split("\n"):
                    key_str, entry_str = line.split(" ")
                    key = extract_key(key_str)
                    table[key] = eval(entry_str)

    # NOTE: do not modify
    @staticmethod
    def tester_print(i, n, name):
        print(f"\r  {name} {i + 1}/{n}", end="")
        if i == n - 1:
            print()


import matplotlib.pyplot as plt

def multiplt(y_s, row, col, row_name=["f1", "f2", "f3"], col_name=["", "",""], ylabel="function value",
             plotname=""):
    plt.figure(figsize=(12, 12), dpi=80)
    plt.figure(1)
    pre = 100 * row + 10 * col
    for i in range(row):
        for j in range(col):
            ax = plt.subplot(pre + i * col + j + 1)
            titlename=row_name[i]+" for "+col_name[j]
            plt.title(titlename)
            if i == row - 1:
                plt.xlabel("x / times")
            plt.ylabel(ylabel)

            # plot for each of them
            y = y_s[i * col + j]
            x = np.arange(len(y))
            # print(y)
            #colval = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            plt.plot(x, y) #, color=colval[t]
    plt.show()


if __name__ == '__main__':
    state_A = (10, 0, 9)
    state_B = (20, 0, 9)
    agent = Agent()

    # plot for A,B state under Monte Carlo policy and Temporal-Difference
    y_Mon_A = []
    y_Mon_B = []
    y_TD_A = []
    y_TD_B = []
    y_sum = [y_Mon_A, y_Mon_B, y_TD_A, y_TD_B]
    T = 50000
    print("Monte Carlo policy evaluation...")
    now_A = 0
    now_B = 0
    while now_A <= T or now_B <= T:
        agent.MC_run(1)
        if agent.N_MC[state_A] == now_A and now_A <= T:
            y_Mon_A.append(agent.MC_values[state_A])
            now_A += 1
        if agent.N_MC[state_B] == now_B and now_B <= T:
            y_Mon_B.append(agent.MC_values[state_B])
            now_B += 1
    print("Temporal-Difference policy evaluation...")
    now_A = 0
    now_B = 0
    while now_A <= T or now_B <= T:
        agent.TD_run(1)
        if agent.N_TD[state_A] == now_A and now_A <= T:
            y_TD_A.append(agent.TD_values[state_A])
            now_A += 1
        if agent.N_TD[state_B] == now_B and now_B <= T:
            y_TD_B.append(agent.TD_values[state_B])
            now_B += 1
    multiplt(y_sum, 2, 2, ["Monte Carlo policy evaluation", "Temporal-Difference policy evaluation"],
             ["state_A=(10,0,9)", "state_B=(20,0,9)"], ylabel="value estimate")


    # plot how the Q-value changes over the number of visits to each action for the same two game states
    y_hit_A = []
    y_hit_B = []
    y_stand_A = []
    y_stand_B = []
    y_sum = [y_hit_A, y_hit_B, y_stand_A, y_stand_B]
    T = 50000
    print("Q-learning...")
    now_A = 1
    now_B = 1
    while now_A <= T or now_B <= T:
        agent.Q_run(1)
        if agent.N_Q[state_A] == now_A and now_A <= T:
            y_hit_A.append(agent.Q_values[state_A][0])
            y_stand_A.append(agent.Q_values[state_A][1])
            now_A += 1
        if agent.N_Q[state_B] == now_B and now_B <= T:
            y_hit_B.append(agent.Q_values[state_B][0])
            y_stand_B.append(agent.Q_values[state_B][1])
            now_B += 1
    multiplt(y_sum, 2, 2, ["Q-learning (Hit)", "Q-learning (stand)"],
             ["state_A=(10,0,9)", "state_B=(20,0,9)"], ylabel="Q-value")

    print("winning rate of auto-play by Q-learning...")
    winrate = []
    y_sum = [winrate]
    T = 50000
    now = 1
    agent = Agent()
    game = Game()
    while now <= T:
        while not (game.game_over() or game.stand):
            decision = agent.autoplay_decision(copy.deepcopy(game.state))
            if decision == 0:
                game.act_hit()
            else:
                game.act_stand()
        game.update_stats()
        game.reset()

        agent.Q_run(1)
        winrate.append(game.winNum / (game.winNum + game.loseNum))
        now += 1
    multiplt(y_sum, 1, 1, ["Q-learning"],
             ["Blackjack ", ""], ylabel="winning rate")




import numpy as np
import matplotlib.pyplot as plt
import random


class WindyGridworld:
    def __init__(self, kings_moves):
        self.width = 10
        self.height = 7
        self.state = (3, 0)
        self.winds = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
        self.goal = (3, 7)
        self.kings_moves = kings_moves

    def at_goal(self):
        if (self.state == self.goal):
            return True

    def move(self, move):
        if not self.kings_moves:
            new_position = list(self.state)
            if move == "N":
                new_position[0] = self.state[0] - 1
            elif move == "S":
                new_position[0] = self.state[0] + 1
            elif move == "W":
                new_position[1] = self.state[1] - 1
            else:
                new_position[1] = self.state[1] + 1
            # apply winds to the current position
            new_position[0] -= self.winds[self.state[1]]
        else:
            new_position = list(self.state)
            if move == "N":
                new_position[0] = self.state[0] - 1
            elif move == "S":
                new_position[0] = self.state[0] + 1
            elif move == "W":
                new_position[1] = self.state[1] - 1
            elif move == "E":
                new_position[1] = self.state[1] + 1
            elif move == "NE":
                new_position[0] = self.state[0] - 1
                new_position[1] = self.state[1] + 1
            elif move == "NW":
                new_position[0] = self.state[0] - 1
                new_position[1] = self.state[1] - 1
            elif move == "SE":
                new_position[0] = self.state[0] + 1
                new_position[1] = self.state[1] + 1
            else:
                new_position[0] = self.state[0] + 1
                new_position[1] = self.state[1] - 1
            # apply stochastic winds to the current position
            if self.winds[self.state[1]] > 0:
                r = random.randint(1, 3)
                if r == 1:
                    # 1/3 of the time, apply normal wind
                    new_position[0] -= self.winds[self.state[1]]
                elif r == 2:
                    # 1/3 of the time, apply normal wind +1
                    new_position[0] -= (self.winds[self.state[1]] + 1)
                else:
                    # 1/3 of the time, apply normal wind -1
                    new_position[0] -= (self.winds[self.state[1]] - 1)

        # ensure that the map cannot be left
        if new_position[0] < 0 or new_position[0] > (self.height-1):
            new_position[0] = 0
        self.state = tuple(new_position)
        # determine reward contribution
        if (self.at_goal()):
            reward = 1
        else:
            reward = -1
        return (self.state, reward)

    def get_possible_moves(self, state):
        possible_moves = []
        if state[0] != 0:
            possible_moves.append("N")
        if state[0] != (self.height - 1):
            possible_moves.append("S")
        if state[1] != 0:
            possible_moves.append("W")
        if state[1] != (self.width - 1):
            possible_moves.append("E")
        if self.kings_moves:
            if state[0] != 0 and state[1] != (self.width - 1):
                possible_moves.append("NE")
            if state[0] != 0 and state[1] != 0:
                possible_moves.append("NW")
            if state[0] != (self.height - 1) and state[1] != (self.width - 1):
                possible_moves.append("SE")
            if state[0] != (self.height - 1) and state[1] != 0:
                possible_moves.append("NW")
        return possible_moves

    def get_world_dimensions(self):
        return self.height, self.width

    def reset(self):
        self.state = (3, 0)

    def get_possible_states(self):
        states = []
        for height in range(self.height):
            for width in range(self.width):
                states.append((height, width))
        return states


class Agent:
    def __init__(self, epsilon, alpha, world: WindyGridworld):
        self.epsilon = epsilon
        self.alpha = alpha
        self.world = world
        self.states = np.full(world.get_world_dimensions(), -1, dtype=float)
        self.states[3, 7] = 1
        self.Q = self.initialize_Q(self.world.get_possible_states())

    def initialize_Q(self, states):
        Q = {}
        for s in states:
            Q[s] = {}
            possible_moves = self.world.get_possible_moves(s)
            for a in possible_moves:
                Q[s][a] = 0
        return Q

    def max_Q(self, Q, state):
        max_value = -99999
        max_action = "N"
        for k, v in Q[state].items():
            if v > max_value:
                max_value = v
                max_action = k
        return max_action

    def max_Q_value(self, Q, state):
        max_value = -99999
        max_action = "N"
        for k, v in Q[state].items():
            if v > max_value:
                max_value = v
                max_action = k
        return max_value

    def get_move_epsilon_greedy(self, epsilon, position):
        possible_moves = self.world.get_possible_moves(world.state)
        rand = np.random.rand()
        if (rand < epsilon):
            move = possible_moves[np.random.choice(len(possible_moves))]
        else:
            move = self.max_Q(self.Q, position)
        return move

    def sarsa(self, iteration, gamma=0.5, alpha=0.5, epsilon=1):
        visiting = np.zeros(self.world.get_world_dimensions())
        self.world.reset()
        rewards = []
        state_history = []
        visiting = np.zeros(self.world.get_world_dimensions())
        nSteps = 0
        while True:
            nSteps += 1
            state = world.state[:]
            # move the player
            move = self.get_move_epsilon_greedy(
                epsilon/iteration, state)
            tmp = self.world.move(move)
            visiting[tmp[0][0], tmp[0][1]] += 1
            reward = tmp[1]
            # get the action for the next state
            next_position = world.state[:]
            next_move = self.get_move_epsilon_greedy(
                epsilon/iteration, next_position)

            # update SARSA rule
            self.Q[state][move] = self.Q[state][move] + alpha * \
                (reward + gamma * self.Q[next_position][next_move] -
                 self.Q[state][move])

            rewards.append(reward)
            if self.world.at_goal():
                break
        return np.sum(rewards)

    def Q_learning(self, iteration, gamma=0.5, alpha=0.5, epsilon=1):
        visiting = np.zeros(self.world.get_world_dimensions())
        self.world.reset()
        rewards = []
        state_history = []
        visiting = np.zeros(self.world.get_world_dimensions())
        nSteps = 0
        while True:
            nSteps += 1
            state = world.state[:]
            # move the player
            move = self.get_move_epsilon_greedy(
                epsilon/iteration, state)
            tmp = self.world.move(move)
            visiting[tmp[0][0], tmp[0][1]] += 1
            reward = tmp[1]
            # get the action for the next state
            next_position = world.state[:]
            next_move = self.get_move_epsilon_greedy(
                epsilon/iteration, next_position)

            # TD Update
            td_target = reward + gamma * \
                self.max_Q_value(self.Q, next_position)
            td_error = td_target - self.Q[state][move]
            self.Q[state][move] += alpha * td_error

            rewards.append(reward)
            if self.world.at_goal():
                break
        return np.sum(rewards)

    def follow_optimal_policy(self):
        self.world.reset()
        rewards = []
        visiting = np.zeros(self.world.get_world_dimensions())
        t = 0
        while True:
            t += 1
            move = self.get_move_epsilon_greedy(
                epsilon=0, position=self.world.state[:])
            tmp = self.world.move(move)
            visiting[tmp[0][0], tmp[0][1]] = t
            rewards.append(tmp[1])
            if self.world.at_goal():
                break

        reward = np.sum(rewards)
        return reward, visiting

# Logging functionality


def print_world(world: WindyGridworld):
    dim = world.get_world_dimensions()
    print("----------------------------------------")
    for i in range(dim[0]):
        for j in range(dim[1]):
            if (world.state == (i, j)):
                print("X", end="\t")
            else:
                print("0", end="\t")
        print()
    print("----------------------------------------")
    print("----------------------------------------")
    print()


# Logging functionality
def print_path(world: WindyGridworld, visiting):
    dim = world.get_world_dimensions()
    print("Path taken:")
    print("----------------------------------------")
    for i in range(dim[0]):
        for j in range(dim[1]):
            print('{:1.0f}'.format(visiting[(i, j)]), end="\t")
        print()
    print("----------------------------------------")
    print()


if __name__ == '__main__':
    world = WindyGridworld(kings_moves=False)
    player = Agent(epsilon=.1, alpha=.2, world=world)
    training_rounds = np.arange(1, 1001, 1)

    rewards = []
    for round in training_rounds:
        rewards.append(player.sarsa(iteration=round,
                                    epsilon=1, gamma=0.5, alpha=0.5))

    print("Training done")
    ret = player.follow_optimal_policy()
    print("Reward is", ret[0])

    # print out optimal path chosen
    print_path(player.world, ret[1])

    plt.plot(rewards)
    plt.show()

    world = WindyGridworld(kings_moves=True)
    player = Agent(epsilon=0.05, alpha=0.5, world=world)
    training_rounds = np.arange(1, 1001, 1)

    rewards = []
    for round in training_rounds:
        rewards.append(player.Q_learning(iteration=round,
                                         epsilon=0.05, gamma=0.9, alpha=0.5))

    print("Training done")
    ret = player.follow_optimal_policy()
    print("Reward is", ret[0])

    # print out optimal path chosen
    print_path(player.world, ret[1])

    plt.plot(rewards)
    plt.show()

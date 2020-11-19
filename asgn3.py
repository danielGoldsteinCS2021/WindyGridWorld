import numpy as np
import matplotlib.pyplot as plt
import random


class WindyGridworld:
    def __init__(self, kings_moves):
        self.kings_moves = kings_moves
        self.state = (3, 0)
        self.winds = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
        self.goal = (3, 7)
        self.width = 10
        self.height = 7
        self.possible_states = [(h, w) for h in range(self.height) for w in range(self.width)]

    def action(self, action):
        if not self.kings_moves:
            new_state = list(self.state)
            if action == "N":
                new_state[0] = self.state[0] - 1
            elif action == "S":
                new_state[0] = self.state[0] + 1
            elif action == "W":
                new_state[1] = self.state[1] - 1
            else:
                new_state[1] = self.state[1] + 1
            # apply winds to the current position
            new_state[0] -= self.winds[self.state[1]]
        else:
            new_state = list(self.state)
            if action == "N":
                new_state[0] = self.state[0] - 1
            elif action == "S":
                new_state[0] = self.state[0] + 1
            elif action == "W":
                new_state[1] = self.state[1] - 1
            elif action == "E":
                new_state[1] = self.state[1] + 1
            elif action == "NE":
                new_state[0] = self.state[0] - 1
                new_state[1] = self.state[1] + 1
            elif action == "NW":
                new_state[0] = self.state[0] - 1
                new_state[1] = self.state[1] - 1
            elif action == "SE":
                new_state[0] = self.state[0] + 1
                new_state[1] = self.state[1] + 1
            else:
                new_state[0] = self.state[0] + 1
                new_state[1] = self.state[1] - 1
            # apply stochastic winds to the current position
            if self.winds[self.state[1]] > 0:
                r = random.randint(1, 3)
                if r == 1:
                    # 1/3 of the time, apply normal wind
                    new_state[0] -= self.winds[self.state[1]]
                elif r == 2:
                    # 1/3 of the time, apply normal wind +1
                    new_state[0] -= self.winds[self.state[1]] + 1
                else:
                    # 1/3 of the time, apply normal wind -1
                    new_state[0] -= self.winds[self.state[1]] - 1

        # Stop at walls
        if new_state[0] < 0 or new_state[0] > self.height-1:
            new_state[0] = 0
        self.state = tuple(new_state)  # Ensures state is still a hashable object

        if self.state == self.goal:
            reward = 1
        else:
            reward = -1
        return self.state, reward

    def possible_actions(self, state):
        possible_actions = []
        if state[0] != 0:
            # Not at ceiling - can move north
            possible_actions.append("N")
        if state[0] != self.height - 1:
            # Not at floor - can move south
            possible_actions.append("S")
        if state[1] != 0:
            # Not at leftmost column - can move west
            possible_actions.append("W")
        if state[1] != self.width - 1:
            # Not at rightmost column - can move east
            possible_actions.append("E")
        if self.kings_moves:
            # Add diagonal King's moves if applicable
            if state[0] != 0 and state[1] != self.width - 1:
                # Not at ceiling or rightmost column - can move north-east
                possible_actions.append("NE")
            if state[0] != 0 and state[1] != 0:
                # Not at ceiling or leftmost column - can move north-west
                possible_actions.append("NW")
            if state[0] != self.height - 1 and state[1] != self.width - 1:
                # Not at floor or rightmost column - can move south-east
                possible_actions.append("SE")
            if state[0] != self.height - 1 and state[1] != 0:
                # Not at floor or leftmost column - can move south-west
                possible_actions.append("SW")
        return possible_actions


class Agent:
    def __init__(self, epsilon, alpha, environment: WindyGridworld):
        self.epsilon = epsilon
        self.alpha = alpha
        self.environment = environment
        self.states = np.full(
            (self.environment.height, self.environment.width), -1, dtype=float)
        self.states[3, 7] = 1
        self.Q = self.initialize_Q(self.environment.possible_states)

    def initialize_Q(self, states):
        Q = {}
        for s in states:
            Q[s] = {}
            possible_actions = self.environment.possible_actions(s)
            for a in possible_actions:
                Q[s][a] = 0
        return Q

    def max_Q(self, Q, state):
        max_value = -6969
        max_action = "N"
        for k, v in Q[state].items():
            if v > max_value:
                max_value = v
                max_action = k
        return max_action, max_value

    def epsilon_greedy(self, epsilon, position):
        possible_actions = self.environment.possible_actions(world.state)
        r = np.random.rand()
        if r < epsilon:
            action = possible_actions[np.random.choice(len(possible_actions))]
        else:
            action = self.max_Q(self.Q, position)[0]
        return action

    def sarsa(self, iteration, gamma=0.5, alpha=0.5, epsilon=1):
        visits = np.zeros((self.environment.height, self.environment.width))
        self.environment.state = (3, 0)  # Reset agent's position
        rewards = []
        state_history = []
        nSteps = 0
        while True:
            nSteps += 1
            state = world.state[:]
            # move the agent
            action = self.epsilon_greedy(
                epsilon/iteration, state)
            tmp = self.environment.action(action)
            visits[tmp[0][0], tmp[0][1]] += 1
            reward = tmp[1]
            # get the action for the next state
            next_position = world.state[:]
            next_action = self.epsilon_greedy(
                epsilon/iteration, next_position)

            # update SARSA rule
            self.Q[state][action] = self.Q[state][action] + alpha * \
                (reward + gamma * self.Q[next_position][next_action] -
                 self.Q[state][action])

            rewards.append(reward)
            if self.environment.state == self.environment.goal:
                break
        return np.sum(rewards), nSteps

    def Q_learning(self, iteration, gamma=0.5, alpha=0.5, epsilon=1):
        visits = np.zeros((self.environment.height, self.environment.width))
        self.environment.state = (3, 0)  # Reset agent's position
        rewards = []
        state_history = []
        nSteps = 0
        while True:
            nSteps += 1
            state = world.state[:]
            # move the player
            action = self.epsilon_greedy(
                epsilon/iteration, state)
            tmp = self.environment.action(action)
            visits[tmp[0][0], tmp[0][1]] += 1
            reward = tmp[1]
            # get the action for the next state
            next_position = world.state[:]
            next_action = self.epsilon_greedy(
                epsilon/iteration, next_position)

            # TD Update
            td_target = reward + gamma * \
                self.max_Q(self.Q, next_position)[1]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += alpha * td_error

            rewards.append(reward)
            if self.environment.state == self.environment.goal:
                break
        return np.sum(rewards), nSteps

    def optimal_policy(self):
        self.environment.state = (3, 0)  # Reset agent's position
        rewards = []
        visits = np.zeros((self.environment.height, self.environment.width))
        t = 0
        while True:
            t += 1
            action = self.epsilon_greedy(
                epsilon=0, position=self.environment.state[:])
            tmp = self.environment.action(action)
            visits[tmp[0][0], tmp[0][1]] = t
            rewards.append(tmp[1])
            if self.environment.state == self.environment.goal:
                break

        reward = np.sum(rewards)
        return reward, visits


def pathPrinter(world: WindyGridworld, visiting):
    # Print the path the agent takes through the windy gridworld
    dimensions = (world.height, world.width)
    print("\n")
    print("*************************************************************************")
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            print(int(visiting[(i, j)]), end="\t")
        print()
    print("*************************************************************************")
    print("\n")


if __name__ == '__main__':
    print("4 moves - SARSA")
    world = WindyGridworld(kings_moves=False)
    player = Agent(epsilon=.1, alpha=.2, environment=world)
    rounds = np.arange(1, 1001, 1)

    rewards = []

    steps = []
    for r in rounds:
        _return = player.sarsa(iteration=r,
                               epsilon=1, gamma=0.5, alpha=0.5)
        rewards.append(_return[0])
        steps.append(_return[1])

    print("Training complete")
    policy = player.optimal_policy()
    print("Reward:", policy[0])

    # print out optimal path chosen
    pathPrinter(player.environment, policy[1])
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.plot(steps)
    plt.show()

    print("4 moves - Q-Learning")
    world = WindyGridworld(kings_moves=False)
    player = Agent(epsilon=.1, alpha=.2, world=world)
    rounds = np.arange(1, 1001, 1)
    rewards = []
    steps = []
    for round in rounds:
        _return = player.Q_learning(iteration=round,
                                    epsilon=1, gamma=0.5, alpha=0.5)
        rewards.append(_return[0])
        steps.append(_return[1])

    print("Training complete")
    policy = player.optimal_policy()
    print("Reward:", policy[0])
    # print out optimal path chosen
    vizualize_path(player.world, policy[1])
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.plot(steps)
    plt.show()

    print("King's Moves, Stochastic Wind - SARSA")
    world = WindyGridworld(kings_moves=True)
    player = Agent(epsilon=0.05, alpha=0.5, environment=world)
    rounds = np.arange(1, 1001, 1)
    rewards = []

    steps = []
    for r in rounds:
        _return = player.sarsa(iteration=r,
                               epsilon=0.05, gamma=0.9, alpha=0.5)
        rewards.append(_return[0])
        steps.append(_return[1])
        
    print("Training complete")
    policy = player.optimal_policy()
    print("Reward:", policy[0])
    # print out optimal path chosen
    pathPrinter(player.environment, policy[1])
    vizualize_path(player.world, policy[1])
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.plot(steps)
    plt.show()

    print("King's Moves, Stochastic Wind - Q-Learning")
    world = WindyGridworld(kings_moves=True)
    player = Agent(epsilon=0.05, alpha=0.5, world=world)
    rounds = np.arange(1, 1001, 1)
    rewards = []
    steps = []
    for round in rounds:
        _return = player.Q_learning(iteration=round,
                                    epsilon=0.05, gamma=0.9, alpha=0.5)
        rewards.append(_return[0])
        steps.append(_return[1])
    print("Training complete")
    policy = player.optimal_policy()
    print("Reward:", policy[0])
    # print out optimal path chosen
    vizualize_path(player.world, policy[1])
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.plot(steps)
    plt.show()

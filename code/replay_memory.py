import random


class ReplayMemory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.finals = []

    def store_transition(self, state, action, reward, next_state, final):
        index = self.memory_counter % self.memory_size
        if len(self.states) < self.memory_size:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_states.append(None)
            self.finals.append(None)

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.finals[index] = final
        self.memory_counter += 1

    def r(self, element_list, num):
        return element_list[num]

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.memory_size)
        batch = random.sample(range(max_mem), k=batch_size)
        states = [self.r(self.states, n) for n in batch]
        actions = [self.r(self.actions, n) for n in batch]
        rewards = [self.r(self.rewards, n) for n in batch]
        next_states = [self.r(self.next_states, n) for n in batch]
        finals = [self.r(self.finals, n) for n in batch]

        return states, actions, rewards, next_states, finals

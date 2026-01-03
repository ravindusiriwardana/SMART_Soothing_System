import random
import pickle
import os
from collections import defaultdict

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.6, gamma=0.9, epsilon=0.2):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_vals = self.Q[state]
        max_val = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
        
        # Q-Learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.Q), f)
        print("💾 Q-table saved.")

    def load(self, filepath):
        if not os.path.exists(filepath):
            print("⚠️ No previous Q-table found. Starting fresh.")
            return
        try:
            with open(filepath, "rb") as f:
                q_dict = pickle.load(f)
            self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions}, q_dict)
            print("✅ Q-table loaded.")
        except Exception as e:
            print(f"❌ Error loading Q-table: {e}")
import numpy as np

class QTable():
    def __init__(self, mapsize: int = 10, num_actions: int = 0, num_agents: int = 0) -> None:
        self.map_size = mapsize
        state_num = mapsize ** (num_agents*2)
        action_num = num_actions ** (num_agents)
        self.table = np.zeros((state_num, action_num), dtype=np.float32)
        
    def update(self, current_state_x, current_action_x, reward, next_state_x, next_action_x, α, γ):
        current_state_10 = int(current_state_x, self.map_size)
        current_action_10 = int(current_action_x, 5)
        next_state_10 = int(next_state_x, self.map_size)
        next_action_10 = int(next_action_x, 5)
        # print(current_state_10, current_action_10)
        # print(next_state_10, next_action_10)
        predict = self.table[current_state_10][current_action_10]
        target = reward + γ * self.table[next_state_10][next_action_10]
        self.table[current_state_10][current_action_10] += α * (target - predict)
    
if __name__ == "__main__":
    qtable = QTable(gridsize=8, num_actions=5, num_agents=4)
    print(qtable.table.shape)
    print(np.argmax(qtable.table[11455732]))
    max_val_indices = np.where(qtable.table[11455732] == 0)[0][0]
    print(max_val_indices)
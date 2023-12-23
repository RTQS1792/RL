from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from maps import *
import copy


class GridEnvironment:
    def __init__(self, map_name: str = None) -> None:
        if map_name and map_name in globals():
            self.map = np.array(globals()[map_name])  # Define the map
            # Find the start and goal indices
            start_indices = np.where(self.map == 2)
            goal_indices = np.where(self.map == 3)

            # Convert start tuples to concatenated string representation
            self.starts_x = "".join(f"{r}{c}" for r, c in zip(start_indices[0], start_indices[1]))
            self.goals_x = "".join(f"{r}{c}" for r, c in zip(goal_indices[0], goal_indices[1]))

            self.map_size = self.map.shape[0]
            self.num_agents = len(start_indices[0])
            self.agents_x = self.starts_x

    def reset(self):
        self.agents_x = self.starts_x
        return self.agents_x

    def choose_action(self, current_state_x, qtable, ε, ACTIONS):
        action = ""
        while True:
            if np.random.uniform(0, 1) < ε:
                original_action = ""
                for i in range(self.num_agents):
                    original_action += str(np.random.randint(0, 4))
                action = original_action.zfill(4)  # Add leading zeros to make it 4 digits
            else:
                current_state_10 = int(current_state_x, self.map_size)
                max_action_10 = qtable[current_state_10].argmax()
                max_action_x = np.base_repr(max_action_10, base=5)
                max_action_x = max_action_x.zfill(4)  # Add leading zeros to make it 4 digits
                action = max_action_x
            if self.valid_actions(action, ACTIONS):
                break
        return action
    
    def valid_actions(self, max_action_x, ACTIONS):
        # Check if a action is valid
        next_state_x = ""
        for i in range(self.num_agents):
            next_state_x += str(int(str(self.agents_x)[2 * i]) + ACTIONS[max_action_x[i]][0])
            next_state_x += str(int(str(self.agents_x)[2 * i + 1]) + ACTIONS[max_action_x[i]][1])
        # Check if the agents are in the wall
        for i in range(self.num_agents):
            if self.map[int(next_state_x[2 * i]), int(next_state_x[2 * i + 1])] == 1:
                return False

        # Check if the agents have collided
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if next_state_x[2 * i:2 * i + 2] == next_state_x[2 * j:2 * j + 2]:
                    return False
        return True

    def step(self, current_action_x, ACTIONS):
        reward = -0.1
        next_state_x = ""
        done = False
        for i in range(self.num_agents):
            next_state_x += str(int(str(self.agents_x)[2 * i]) + ACTIONS[current_action_x[i]][0])
            next_state_x += str(int(str(self.agents_x)[2 * i + 1]) + ACTIONS[current_action_x[i]][1])
        
        # Check if the agents are in the wall
        for i in range(self.num_agents):
            if self.map[int(next_state_x[2 * i]), int(next_state_x[2 * i + 1])] == 1:
                reward += -1
                done = True
        
        # Check if the agents have collided
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if next_state_x[2 * i:2 * i + 2] == next_state_x[2 * j:2 * j + 2]:
                    reward += -1
        # if reward != -1:
        #     return self.agents_x, reward, done 
                    done = True
        if done: return next_state_x, reward, done
        
        # Check if the agents have reached the goal
        for i in range(self.num_agents):
            if self.map[int(next_state_x[2 * i]), int(next_state_x[2 * i + 1])] == 3:
                reward += 10
                
        # Check if all agents have reached the goal
        if all(self.map[int(next_state_x[2 * i]), int(next_state_x[2 * i + 1])] == 3 for i in range(self.num_agents)):
            done = True
        
        self.agents_x = next_state_x        
        return next_state_x, reward, done
                

    def render(self) -> None:
        custom_cmap = ListedColormap(["white", "black", "green", "red"])  # Define custom color map
        plt.figure(figsize=(self.map.shape[0] / 2, self.map.shape[1] / 2))  # Set figure size\
        temp_map = copy.deepcopy(self.map)
        plt.imshow(temp_map, cmap=custom_cmap)  # Plot the map
        plt.xticks([])  # Remove xticks
        plt.yticks([])  # Remove yticks
        plt.show()
        
    def show_result(self, qtable, ACTIONS):
        temp = self.starts_x
        while True:
            print(temp)
            current_state_10 = int(temp, self.map_size)
            max_action_10 = np.argmax(qtable[current_state_10])
            max_action_x = np.base_repr(max_action_10, base=5)
            max_action_x = max_action_x.zfill(4)  # Add leading zeros to make it 4 digits
            next_state_x = ""
            for i in range(self.num_agents):
                next_state_x += str(int(str(temp)[2 * i]) + ACTIONS[max_action_x[i]][0])
                next_state_x += str(int(str(temp)[2 * i + 1]) + ACTIONS[max_action_x[i]][1])
            temp = next_state_x
            
            # custom_cmap = ListedColormap(["white", "black", "green", "red"])  # Define custom color map
            # plt.figure(figsize=(self.map.shape[0] / 2, self.map.shape[1] / 2))  # Set figure size\
            # temp_map = copy.deepcopy(self.map)
            # plt.imshow(temp_map, cmap=custom_cmap)  # Plot the map
            # plt.xticks([])  # Remove xticks
            # plt.yticks([])  # Remove yticks
            # plt.show()
        


if __name__ == "__main__":
    env = GridEnvironment("map2")
    print(env.map)
    print(env.starts_x)
    print((env.goals_x))

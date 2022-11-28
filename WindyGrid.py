import numpy as np

class WindyGrid:

    def __init__(self, epsilon, alpha, episodes, kings_moves, stochastic_wind):
        self.epsilon = epsilon
        self.alpha = alpha
        self.dimension = (7,10) #self.location < self.dimensions
        self.start_location = (3,0) #starting location
        self.absorbing_loc = (3,7) #absorbing location
        if kings_moves:
            self.actions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)] #n,ne,e,se,s,sw,w,nw
        else:
            self.actions = [(-1,0),(0,1),(1,0),(0,-1)] #n,e,s,w
        self.qtable = np.zeros((self.dimension[0]*self.dimension[1], len(self.actions))) #totState x length(actions)
        #self.qtable = np.random.random((self.dimension[0]*self.dimension[1], len(self.actions))) #totState x length(actions)
        self.stochastic_wind = stochastic_wind
        self.wind_strength = [0,0,0,-1,-1,-1,-2,-2,-1,0]
        self.default_reward = -1
        self.absorbing_reward = 0
        self.episodes = episodes #how many episodes to run
        self.episodes_steps = np.zeros((episodes,1)) #keeping track of the total steps for each episode (steps * -1)

    #retrieving the state from location in the grid
    def get_state_from_location(self, location):
        return location[0]*10+location[1]

    #retrieving Q_value(state,action)
    def get_qvalue(self, state, action):
        return self.qtable[state, action]
    
    #set Q(state, action) = value
    def set_qvalue(self, state, action, value):
        self.qtable[state, action] = value

    #retrieving the effect of wind given current location
    def get_wind(self, location):
        cur_wind = self.wind_strength[location[1]]
        if self.stochastic_wind:
            return (np.random.randint(low=cur_wind-1, high=cur_wind+2),0)
        return (cur_wind, 0)
    
    #checks if location is inside the grid
    def location_inside_grid(self, location):
        return location[0]>=0 and location[0]<self.dimension[0] and location[1]>=0 and location[1]<self.dimension[1]
    
    #eps greedy action
    def eps_greedy_action(self, location):
        state = self.get_state_from_location(location)
        if self.epsilon > np.random.uniform() or (self.qtable[state]==np.zeros((1,len(self.actions)))).all():
            return self.actions[np.random.randint(len(self.actions))]
        return self.actions[np.argmax(self.qtable[state])]

    #returns next location given action and wind (stays if move outside)
    def get_next_location(self, location, action):
        if not self.location_inside_grid((location[0]+action[0], location[1]+action[1])):
            return location
        wind = self.get_wind(location)
        next_loc = [location[0]+action[0]+wind[0], location[1]+action[1]+wind[1]]
        if next_loc[0]>=self.dimension[0]:
            next_loc[0]=self.dimension[0]-1
        elif next_loc[0]<0:
            next_loc[0]=0
        if next_loc[1]>=self.dimension[1]:
            next_loc[1]=self.dimension[1]-1
        elif next_loc[1]<0:
            next_loc[1]=0
        return (next_loc[0],next_loc[1])

    #return reward in given location
    def get_reward(self, location):
        if location==self.absorbing_loc:
            return self.absorbing_reward
        return self.default_reward
    
    def get_reward_and_location(self, location, action):
        reward = self.get_reward(location)
        state = self.get_next_location(location, action)
        return reward, state

    def sample_episode_sarsa(self, episode_idx):
        location = list(self.start_location)
        action = self.eps_greedy_action(location)
        while location != self.absorbing_loc:
            reward, next_location = self.get_reward_and_location(location, action)
            next_action = self.eps_greedy_action(next_location)
            action_idx = self.actions.index(action)
            next_action_idx = self.actions.index(next_action)
            state_idx = self.get_state_from_location(location)
            next_state_idx = self.get_state_from_location(next_location)
            qvalue = self.get_qvalue(state_idx, action_idx) + self.alpha*(reward + self.get_qvalue(next_state_idx, next_action_idx) - self.get_qvalue(state_idx, action_idx))
            self.set_qvalue(state_idx, action_idx, qvalue)
            location = next_location
            action = next_action
            self.episodes_steps[episode_idx] += 1
        return self.episodes_steps[episode_idx]
    
    def sarsa_learning(self):
        for eps_idx in range(self.episodes):
            self.sample_episode_sarsa(eps_idx)
            if eps_idx/self.episodes*100 % 20 == 0:
                print(str(int(eps_idx/self.episodes*100))+"%")
        total_steps = int(sum(self.episodes_steps))
        average_last_100 = np.average(self.episodes_steps[self.episodes-1000:])
        min_steps = np.min(self.episodes_steps)
        print("Total number of steps: " + str(total_steps) + ". Average of last 1000 episodes: " + str(average_last_100) + ". Min: " + str(min_steps))
    
    #given current loc returns A_t, S_t+1, R_t+1
    def q_step(self, location):
        action = self.eps_greedy_action(location)
        action_idx = self.actions.index(action)
        next_loc = self.get_next_location(location, action)
        reward = self.get_reward(next_loc)
        return (action_idx, next_loc, reward)
    
    #sample an episode using q-learning and returns the total nbr of steps in episode
    def sample_episode_q(self, episode_idx):
        location = list(self.start_location)
        while location != self.absorbing_loc:
            action_idx, next_loc, reward = self.q_step(location)
            state_idx = self.get_state_from_location(location)
            next_state_idx = self.get_state_from_location(next_loc)
            next_max_q = np.max(self.qtable[next_state_idx])
            qvalue = self.get_qvalue(state_idx, action_idx) + self.alpha*(reward + next_max_q - self.get_qvalue(state_idx, action_idx))
            self.set_qvalue(state_idx, action_idx, qvalue)
            location = next_loc
            self.episodes_steps[episode_idx] += 1
        return self.episodes_steps[episode_idx]
    
    def q_learning(self):
        for eps_idx in range(self.episodes):
            self.sample_episode_q(eps_idx)
            if eps_idx/self.episodes*100 % 20 == 0:
                print(str(int(eps_idx/self.episodes*100))+"%")
        total_steps = int(sum(self.episodes_steps))
        average_last_100 = np.average(self.episodes_steps[self.episodes-1000:])
        min_steps = np.min(self.episodes_steps)
        print("Total number of steps: " + str(total_steps) + ". Average of last 1000 episodes: " + str(average_last_100) + ". Min: " + str(min_steps))
    
    #prints out optimal policy
    def optimal_policy(self):
        opt_value=np.zeros(self.dimension)
        dir = ("N","NE","E","SE","S","SW","W","NW")
        str = ""
        for row in range(self.dimension[0]):
            for col in range(self.dimension[1]):
                if (row, col) == self.absorbing_loc:                       
                    str += "{:>6}".format("Goal")
                    opt_value[row, col] = -np.inf
                else:
                    state = self.get_state_from_location((row,col))
                    opt_value[row, col] = np.max(self.qtable[state])
                    best_action = dir[np.argmax(self.qtable[state])]
                    str += "{:>6}".format(best_action)
            str += "\n"
        return str, opt_value

    def __str__(self):
        state = 0
        s = ""
        for row in range(self.dimension[0]*self.dimension[1]):
            s += str(state) + "  "
            for col in range(len(self.actions)):
                s += "{:>6.1f}".format(int(self.get_qvalue(row, col))) + "  "
            s += "\n"
            state += 1
        return s
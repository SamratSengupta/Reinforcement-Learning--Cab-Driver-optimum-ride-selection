
Cab drivers, like most people, are incentivised by a healthy growth in income. The goal of your project is to build an RL-based algorithm which can help cab drivers maximise their profits by improving their decision-making process on the field.

The Need for Choosing the 'Right' Requests
Most drivers get a healthy number of ride requests from customers throughout the day. But with the recent hikes in electricity prices (all cabs are electric), many drivers complain that although their revenues are gradually increasing, their profits are almost flat. Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to maximise the total profit earned by the driver that day. 

For example, say a driver gets three ride requests at 5 PM. The first one is a long-distance ride guaranteeing high fare, but it will take him to a location which is unlikely to get him another ride for the next few hours. The second one ends in a better location, but it requires him to take a slight detour to pick the customer up, adding to fuel costs. Perhaps the best choice is to choose the third one, which although is medium-distance, it will likely get him another ride subsequently and avoid most of the traffic. 

There are some basic rules governing the ride-allocation system. If the cab is already in use, then the driver won’t get any requests. Otherwise, he may get multiple request(s). He can either decide to take any one of these requests or can go ‘offline’, i.e., not accept any request at all. 

Markov Decision Process
Taking long-term profit as the goal, we proposed a method based on reinforcement learning to optimize taxi driving strategies for profit maximization. This optimization problem is formulated as a Markov Decision Process.

Deep RL for List-wise Recommendationsfile_download	Download
In this project, we need to create the environment and an RL agent that learns to choose the best request. we trained our agent using vanilla Deep Q-learning (DQN) only and NOT a double DQN. Y

There’s a renowned paper called ‘Deep Reinforcement Learning for List-wise Recommendations’ by Xiangyu Zhao, Liang Zhang, Zhuoye Ding. They have mentioned a few recommendations on how to select the Q-network architecture.  
 

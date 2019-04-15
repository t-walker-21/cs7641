I have not implemented any code for this assignment. All software was completed stolen. So, I will make a folder in my CS7641 repo called RL_SIM. This what you will need to recreate any experiments of mine. 

Clone the repo here:

'git clone git@github.com:t-walker-21/cs7641.git'

cd into RL_sim:

'cs7641/RL_sim'


For experiment recreation:

run the java app:

'java -jar rl_sim.jar'

a gui will appear showing options for different algorithms, select whichever one you'd like

the two MDPs I used were:

1_smallest2.maze

and 

7_medium_complex_multigoal.maze

you'll need to select 'Load Maze' to load the maze file

once loaded, there will be a panel on the left-hand side of the screen. For the smaller MDP, I put PJOG (transition prob) as 0 to force determinism. I choose PJOG to be 0.3 with the other MDP.

To set up the Q-learning agent, I experimented with 0 and 0.5 for epsilon. I used 500K training iterations (denoted as cycles in the panel). I left learning rate to the default.



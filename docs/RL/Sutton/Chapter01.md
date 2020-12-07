#Chapter01
## 1.1 Self-Play
### 1.1.1 Players use and update the same model(e.g. NN)
When the player and the opponent player play with the same intelligence (NN),
the model would reach some kinds of equilibrium and stop to be optimized:
The same state, the winner treats it as good state, loser treats as bad state.
The winner would increase possibility of the state while loser would decrease it, value/possibility of the state would hard to be updated.
Or, the winner would manage to reach the certain state while the loser would manage to prevent the opponent to reach the state,
the state would be hard to be reached and updated. Then agents would turn to secondly optimized alternatives, 
making the model vibrating between these relatively good states.
### 1.1.1 Different NN
Self-play is a hot topic currently. 
In general, if one agent(A) controlled by a model play with another agent(B) that is 
controlled by a better model, the agent A would try to  improve itself to beat agent B.
Then the models would be improved alternatively.

[Competitive self play](https://openai.com/blog/competitive-self-play/)

[Self play](https://towardsdatascience.com/what-can-agents-learn-through-self-play-37adb3f3581b)
## 1.2 Symmetries
The number of (state) and (state, action) are decreased --> 
dimension decreased --> memory and computation decreased.

If the opponent did not take advantage of it, 
the player would get a tree with less state roll out.
That is, the player loss some state exploration as opponent may take different actions
to symmetric state. While the model may converge to a distribution that symmetric states will have the similar values (or other properties)
## 1.3 Greedy Play
It may converge to a local optimization. 
It may play better as it converges quicker.
It also may be trapped in a sub-optimized state that never wins
## 1.4 Learning from Exploration
Learning without exploration would converge to a certain policy.
The policy depends heavily on initialization state and random opponents.

Learning with exploration intend to get expectation of state values,
while it violated policy learning to some degree. 

Seemed the exploration is better as it knows the game better.
## 1.5 Other Improvements
The tic-tac-toe game is a simple game and the states and actions are numbered.
It could be learned from dynamic programming to traverse all possible states and actions.

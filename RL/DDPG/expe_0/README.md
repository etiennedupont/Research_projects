# RL project : Off-policy DDPG

In this first experiment, I implement a DDPG algorithm on the basis of D. Silver's paper but modified so that the critic that is built approximates the optimal Q-value $Q^*$ rather than the Q-value corresponding to the policy, $Q^\pi$.

I tried two different methods to optimize the Q-value, that differ on the way to find an approximation of $\max_a Q(s,a)$ - over a continuum of actions. The first method gives a really poor approximation, which looks more like a random sampling over the action domain. The second one is much more accurate.

Tests were done on the pendulum game from gym library, comparing the modified DDPG algorithm to the original one. Interestingly, the first method yields the best results, with performances that are comparable to the original method. The second method seems not to lead to any convergence of the algorithm.

In the two next experiments, we will try to go further those first conclusions by trying to substitute the critic by a totally random Q-value (expe_1) or an optimal (already trained) Q-value (expe_2).

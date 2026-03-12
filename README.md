# Optimising Simulated Robot Soccer Through Multi-Agent Co-Design

This repository contains the code of "Optimising Simulated Robot Soccer Through Multi-Agent Co-Design", an MSc thesis in Computational Science &amp; Engineering at the Technical University of Munich 2026. 

Author: Tiia Tikkala

## Abstract

The goal of robot co-design is to find the optimal morphology (physical shape) and controller of a robot in a given environment simultaneously. This thesis develops a co-design framework for multi-agent co-design in the context of robot soccer, which could enable faster and more robust design of teams of robots for other complex tasks like search and rescue. A co-design framework includes three main components: a controller, an optimisation method, and a surrogate model used to estimate the performance of candidate designs. The controllers were trained using deep multi-agent reinforcement learning, specifically the MATD3 algorithm. Particle Swarm Optimisation was the optimiser used to find the design with the highest estimated performance. Two methods were explored for estimating the fitness of a candidate design: one based on an action-value-function (also called a Q-function) trained on diverse designs, and one based on a Gaussian process (GP) regression model. MATD3 was found to learn good controllers for 2v2 robot soccer, where the agents clearly learn to approach the ball and manipulate it to move it towards the opponent's goal. The Q-function-based co-design method was found to converge immediately to the design with the longest legs possible. While this did not perform worse than a randomly sampled baseline, it implies that there are issues with the algorithm. The GP-based co-design method was found to explore the design space diversely and to perform at a similar level as the random baseline. The GP-based method gained a final average return that is 17\% higher than the baseline when looking at the best-so-far metric. The results show that multi-agent reinforcement learning can be applied to robot co-design, and establish a baseline for future work.

## Supplementary videos

The following video shows 33 seconds of game play at the end of training:

https://github.com/user-attachments/assets/e631c700-00e8-49d1-988b-2e4778155bd0

The away team (red) are playing with a pre-trained, fixed policy (trained with MATD3), while the home team (blue) have started from a pre-trained, fixed policy but have been trained for 500 episodes to play against the away team using MATD3. All players approach the ball, the home team scores in the away team's goal.

The following video shows dribbling behaviour demonstrated by a single agent trained in a 1v1 game against a random-action opponent:

https://github.com/user-attachments/assets/6f0af96a-9108-4106-bedf-25d4463fdd7c

The agent has clearly learned to approach and manipulate the ball.


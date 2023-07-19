# Lunar Lander from scratch implementation

Details:
- this is a from scratch implementation of the lunar lander
- the Reinforcement Learning environment is implemented in Python with the Gym Library
- in the background there is a nummerical simulation of the physical system 
- the system is modeled as a 2D motion of rigid body considering the earth gravitational field and turbulent drag (proportional to the squared velocity)
- the simulation is done with ODEINT from the SciPy library
- this package uses the isoda solver that automatically based on the stiffness of the problem switches bewteen explicit and implicit Runge-Kutta
- For the Reinforcement Learning part the DDPG algortihm and implemenation of the stable_baseline3 package that is based on PyTorch is used

For practical use: 
- the environment and simulation is located in the "utils" Folder
-  The "Lunar_Lander_train.py" file contains the code to train the agent on the definded enironement with the DDPG algorithm
-  The "Lunar_Lander_test.py" contains the code to test a random agent or an agent that receive action manully and you are able to load a trained model and test the trained agent 

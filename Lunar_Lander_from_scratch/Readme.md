#Lunar Lander

Details:
- this is an from scratch implementation of the lunar lander
- the Reinforcement Learning environment is implemented in Python with the Gym Library
- in background there is a nummerical simulation of the physical system in the background
- the system is model as a 2D motion of rigid body considering the earth gravitational field and turbulent drag (proportional to the squared velocity)
- the simulation is done with ODEINT from the SciPy library
- this package uses the isoda solver that automatically based on the stiffness of the problem switches bewteen explicit and implicit Runge-Kutta
- For the Reinforcement Learning part the DDPG algortihm is used and an implemenation of the stable_baseline3 package that is based on PyTorch is used

For practical use: 
- the environment and simulation is located in the "utils" Folder
-  The "

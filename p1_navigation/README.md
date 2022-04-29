[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  
En este proyecto se entrena un agente para que se desplace por un espacio acotado de forma cuadrada en el que se encuentran de forma aleatoria banananas de color amarillo y color azul.

The agent's goal will be to collect yellow bananas (passing over them) and avoiding the blue bananas.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The environment is implemented in Unity and collects the agent's actions while providing the agent's information in a state space with 37 dimensions that contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Setting up the environment

The version of pytorch included in the requirements file of the practice causes that the configuration of the environment can not be done directly (at least in my case), so I will indicate below the exact steps that I have followed to configure the environment for the project.

It is assumed in the following steps that Anaconda is installed on a Windows based system.

1. Enable Conda-forge Channel For Conda Package Manager:

    ```console
    conda config --append channels conda-forge
    ```

2. Create a new Anaconda environment with python 3.6

    ```console
    conda create -n unity python=3.6
    ```

3. Active in the newly created environment

    ```console
    activate unity
    ```

4. Install pytorch in its version 0.4.0 with the command conda

    ```console
    conda install pytorch=0.4.0 -c pytorch
    ```

5. Install the dependencies specified in the project requirements (requirements.txt).

    ```console
    pip install .\python\
    ```

6. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

7. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

### Report of the implemented solution

The Jupyter notebook `Navigation.ipynb` contains the details of the implemented solution, as well as the results obtained.

---
**Note:** This markdown document and others in this repository partially include content from the repository provided for the Deep Reinforcement Learning Ucacity Nanodegree projects. The code of this solution is based on the exercises performed in the nanodegree and follows a similar structure to facilitate its understanding and review.
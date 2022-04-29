import numpy as np

# Moving averages are used on the plot to reduce noise
def moving_average(values, window) :
    """Return an array with the moving averages of the given array.
    
    Params
    ======
        values (array_like): current state
        window (int): window size for the moving average
    """         

    ret = np.cumsum(values, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def write_scores(i_episode, scores_window, eps, goal):
    """Write the scores while the agent runs the training episodes.
    
    Params
    ======
        i_episode (int): current episode
        scores_window (array_like): array with the scores for the episodes
        eps (float): epsilon
        goal (float): the score to be achieved
    """             
    
    done = False
    print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps}", end="")
    if i_episode % 100 == 0:
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
    if np.mean(scores_window)>=goal:
        print(f"\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}")        
        done = True
    return done
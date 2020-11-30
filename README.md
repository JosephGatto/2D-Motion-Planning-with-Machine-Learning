# Machine Learning Motion Planner for Mobile Robots

This repository contains a machine learning-based motion planner for either an omnidirectional or differential-drive robot. 

## How It Works 

Simply using Random Forest, we can navigate new environments given training data generated using a set of randomly generated maps. Treating the robot as a point on a 2D grid, we first get a ground truth path determined by A*. Next, we find the 2D laser scan data at each point, as well as the robot's relative distance to the goal. We create training samples X = (laser_scan, relative goal position) Y = (direction). Using 4-point connectivity, the set of directions will be (up, down, left, right). 

Our implementation is entirely numpy/Shapely based and does not utilize ROS. 

## Getting Started 

After cloning the repository, first run 

```
pip install -r requirements.txt
```

Next, see main.ipynb for an example of how to 

1) Generate random maps
2) Generate training samples
3) Train a model
4) Test model on new maps 
5) Plot results


![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

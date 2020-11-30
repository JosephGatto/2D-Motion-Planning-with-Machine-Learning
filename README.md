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

## Example output

![Alt Text](https://github.com/JosephGatto/ML_Motion_Planner/blob/main/images/out.gif)

Here, we see the system output. The model was trained using data generated from 100 randomly generated paths across 10 randomly generated maps. This gif above shows the model navigating in an unseen environment using the trained Random Forest model. 

## TODO
- Only implemented for fixed map size of (50,50) 
- Need to implement Deep Learning model in-place of Random Forest. 
- Would like to allow for non-rectangular obstacles. 

### References / Aknowledgements 
- Project inspired by / related to https://arxiv.org/abs/1609.07910. 
- A* code is from https://github.com/richardos/occupancy-grid-a-star 
- Written by Joseph Gatto & Haowen Liu for Dartmouth COSC 281 final project. 

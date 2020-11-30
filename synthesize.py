import sys
import shapely
from shapely.geometry import box, MultiPolygon, LineString, Point, MultiLineString, Polygon
import math
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

from gridmap import OccupancyGridMap
import matplotlib.pyplot as plt
from a_star import a_star
from utils import plot_path
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
import random
from tqdm import tqdm
import imageio

from planning_utils import *

import warnings
warnings.filterwarnings('ignore')

# Mapping from label encoder (just hard coded it here)
dirs = {0:(1.0, 0.0), 1:(0.0, 1.0), 2:(-1.0, 0.0), 3:(0.0, -1.0)}
inv_dirs = {d: label for label, d in dirs.items()}

# num_points_on_circ = 4
pi = math.pi
def PointsInCircum(r,n=4, center = (0,0)):
    return np.array([(center[0] + math.cos(2*pi/n*x)*r, center[1] + math.sin(2*pi/n*x)*r) for x in range(0,n+1)])

# 4n = 4 point connectivity. 8N = 8 point connectivity
def synthetic_sensor(MAP, robot_location, direction, movement='4N', num_sensor_readings=4, robot_type = 'omni'):
    '''
    Given a shapely map and the robots current location, 
    return the 360 degree laser scanner. 

    robot_type-- String denoting robot type. use 'omni' for omnidirectional or 'ddr' for differential drive robot
    '''
    lines = []
    # 100 is arbitrary radius. Just needs to be big enough to draw line across map.
    # print(num_sensor_readings) 
    points = PointsInCircum(71, n=num_sensor_readings, center = robot_location)
    # Create line to all points on circle
    for point in points:
        A = Point(robot_location)
        B = Point(point)
        line = LineString([A,B])
        lines.append(line)
    # Get points of intersection. 
    for i, line in enumerate(lines):

        # These two types of objects signify multiple intersections. 
        if type(MAP.intersection(line)) == shapely.geometry.multilinestring.MultiLineString or \
               type(MAP.intersection(line)) == shapely.geometry.multipoint.MultiPoint:
            # Get the closest point
            temp_dist = []
            for point in MAP.intersection(line):
#                 try:
                temp_dist.append(LineString([robot_location, point]).length)
            inter = MAP.intersection(line)[np.argmin(temp_dist)]
        elif type(MAP.intersection(line)) == shapely.geometry.collection.GeometryCollection:
            cp = MAP.intersection(line)[0] 
            lines[i].coords = [robot_location, cp]
            continue
        # One intersection
        else:
            inter = MAP.intersection(line)
        # Create new point and update end point in list of lines. 
        new_point = (inter.xy[0][0], inter.xy[1][0])
        lines[i].coords = [robot_location, new_point]
    
    # Get lase scan data (distances)
    distances = [line.length for line in lines][:num_sensor_readings]
    # Account for robot orientation
    if robot_type == 'ddr':
        if direction != (0, 0): # Start node
            if movement == '4N':
                offset = int(len(distances) / 4.) * inv_dirs[direction]
            else: 
                offset = int(len(distances) / 8.) * inv_dirs[direction]
            distances = distances[offset:] + distances[:offset]  
    return distances, lines
    

def synthesize_data(start, goal, MAP, map_arr, polar=False, num_sensor_readings=4, robot_type = 'omni'):
    '''
    synthesize data for one step. 
    
    -Given start and goal, we first get the A* ground truth path. 
    -If get_path() returns an error, that means there was no path found. 
    -For each element in the path, use the synthetic sensor to get the readings 
        the relative goal in odom, and the target movement
    '''
    
    # Get path if one is available
    try:
        path = get_path(start, goal, map_arr, False)
    except:
        return
    
    sensor_readings = []
    goals = []
    relative_goals = []
    polar_goals = [] # Polar coordinates used by paper
    directions = []
    steering = []
    prev = start
    for i, loc in enumerate(path):
        # Get direction to next cell
        direction = (loc[0] - prev[0], loc[1] - prev[1])
        
        # Get rotation
        offset = 0
        if direction != (0, 0): # Start node
            offset = inv_dirs[direction]
        rot = np.pi/2 * offset
        # Get laser scan
        ls,_ = synthetic_sensor(MAP, (loc[0]+0.5, loc[1]+0.5), direction, num_sensor_readings=num_sensor_readings)
        sensor_readings.append(ls)
        # Get goal in odom
        goal_loc = (goal[0]-loc[0], goal[1]-loc[1])
        goals.append(goal_loc)
        if robot_type == 'ddr':  
            # Get goal in odom
            goal_orn = (goal_loc[0]*np.cos(rot) + goal_loc[1]*np.sin(rot), goal_loc[0]*-np.sin(rot) + goal_loc[1]*np.cos(rot))
            relative_goals.append(goal_orn)   
            
            # Get polar distance
            polar_distance = np.linalg.norm(np.array([goal_loc[0],goal_loc[1]]))
            # Get polar rotation
            polar_rotation = math.atan2(goal_orn[1], goal_orn[0])
            polar_goals.append((polar_distance, polar_rotation))
            
            # Get relative steering
            if len(directions) > 1:
                ds = inv_dirs[direction] - inv_dirs[directions[len(directions)-1]]
            elif len(directions) == 1: # First node after start
                ds = inv_dirs[direction]
            else: ds = 0 # Start node
            
            steering.append(ds)
            
            # Get movement to next cell
        directions.append(direction) 
        prev=loc
    if robot_type == 'ddr':    
        if polar:
            return np.array(sensor_readings), np.array(polar_goals), np.array(steering[1:] + steering[:1]), path # Rotated 0 to stop
        return np.array(sensor_readings), np.array(relative_goals), np.array(steering[1:] + steering[:1]), path
    else:
        
        return np.array(sensor_readings), np.array(goals), np.array(directions[1:] + directions[:1]), path

def synthesize_train_set(MAPs, num_runs = 5, polar=False, num_sensor_readings=4, robot_type = 'omni'):
    '''
    -Get 'num_runs' different start/goal locations inside the map
    -If path is available, get training data for a given path
    -Return pandas dataframe where first 360 columns are sensor data, cols
     361 and 362 are odom info and then last 2 columns are x,y movement directions. 
    '''
    
    df = []
    
    for i in tqdm(range(num_runs)):
        # TODO: Generalize to any map shape
        start = (random.randint(1,49), random.randint(1,49))
        goal = (random.randint(1,49), random.randint(1,49))
        MAP, map_arr = random.choice(MAPs)
        # If path is available, get training info
        try:
            
            (sensor_readings, relative_goals, steering, path) = synthesize_data(start, goal, MAP, map_arr, polar, num_sensor_readings= num_sensor_readings, robot_type= robot_type)
            if robot_type == 'omni':
                train = np.concatenate((sensor_readings, relative_goals, steering), axis=1)
            else:
                train = np.concatenate((sensor_readings, relative_goals, np.expand_dims(steering, axis=1)), axis=1)
            
            df.append(train)
            print(start, goal)
        
        except Exception as e:
            # No path found
            # print("err")
            continue
    return pd.DataFrame(np.vstack(df))


def quadFits(map_np, sx, sy, rx, ry, margin):
        """
        looks to see if a quad shape will fit in the grid without colliding with any other tiles
        used by placeRoom() and placeRandomRooms()
         
        Args:
            sx and sy: integer, the bottom left coords of the quad to check
            rx and ry: integer, the width and height of the quad, where rx > sx and ry > sy
            margin: integer, the space in grid cells (ie, 0 = no cells, 1 = 1 cell, 2 = 2 cells) to be away from other tiles on the grid
             
        returns:
            True if the quad fits
        """
         
        sx -= margin
        sy -= margin
        rx += margin*2
        ry += margin*2
        if sx + rx < np.size(map_np, axis=1) and sy + ry < np.size(map_np, axis=0) and sx >= 0 and sy >= 0:
            for x in range(rx):
                for y in range(ry):
                    if map_np[sy+y, sx+x]: 
                        return False
            return True
        return False
    
def placeRandomRooms(map_size, minRoomSize, maxRoomSize, roomStep = 1, margin = 1, attempts = 500):
    """ 
    randomly places quads in the grid
    takes a brute force approach: randomly a generate quad in a random place -> check if fits -> reject if not
    Populates self.rooms

    Args:
        minRoomSize: integer, smallest size of the quad
        maxRoomSize: integer, largest the quad can be
        roomStep: integer, the amount the room size can grow by, so to get rooms of odd or even numbered sizes set roomSize to 2 and the minSize to odd/even number accordingly
        margin: integer, space in grid cells the room needs to be away from other tiles
        attempts: the amount of tries to place rooms, larger values will give denser room placements, but slower generation times

    Returns:
        none
    """
    
    pols = [box(0,0,map_size[0], map_size[1])]
    map_np = np.zeros(map_size)
    
    for attempt in range(attempts):
        roomWidth = random.randrange(minRoomSize, maxRoomSize, roomStep)
        roomHeight = random.randrange(minRoomSize, maxRoomSize, roomStep)
        startX = random.randint(0, map_size[1])
        startY = random.randint(0, map_size[0])            
        if quadFits(map_np, startX, startY, roomWidth, roomHeight, margin):
            for x in range(roomWidth):
                for y in range(roomHeight):
                    map_np[startY+y, startX+x] = 1
            pols.append(box(startX, startY, startX+roomWidth, startY+roomHeight))
    
    lines = []
    for pol in pols:
        boundary = pol.boundary
        if boundary.type == 'MultiLineString':
            for line in boundary:
                lines.append(line)
        else:
            lines.append(boundary)

    MAP = MultiLineString(lines)
    return map_np, MAP
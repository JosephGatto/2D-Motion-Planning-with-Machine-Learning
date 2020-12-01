import sys
import shapely
from shapely.geometry import box, MultiPolygon, LineString, Point, MultiLineString, Polygon
import math
import matplotlib.pyplot as plt
import numpy as np

from astar.gridmap import OccupancyGridMap
import matplotlib.pyplot as plt
from astar.a_star import a_star
from astar.utils import plot_path
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

from utils.planning import *

import warnings
warnings.filterwarnings('ignore')


pi = math.pi
def PointsInCircum(r,n=4, center = (0,0)):
    return np.array([(center[0] + math.cos(2*pi/n*x)*r, center[1] + math.sin(2*pi/n*x)*r) for x in range(0,n+1)])

# 4n = 4 point connectivity. 8N = 8 point connectivity
def synthetic_sensor(MAP, robot_location, direction, args):
    '''
    Generate synthetic sensor data for robot at given location on given map. 

    @params
    MAP-- Shapely map 
    robot_location-- (x,y) location of robot on map
    direction-- Current robot orientation. Only used for args.robot_type=='ddr'
    '''
    lines = []
    # 100 is arbitrary radius. Just needs to be big enough to draw line across map.
    # print(num_sensor_readings) 
    points = PointsInCircum(71, n=args.num_sensor_readings, center = robot_location)
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
    
    # Get laser scan data (distances)
    distances = [line.length for line in lines][:args.num_sensor_readings]
    # Account for robot orientation
    if args.robot_type == 'ddr':
        if direction != (0, 0): # Start node
            if args.movement == '4N':
                offset = int(len(distances) / 4.) * args.inv_dirs[direction]
            else: 
                offset = int(len(distances) / 8.) * args.inv_dirs[direction]
            distances = distances[offset:] + distances[:offset]  
    return distances, lines
    

def synthesize_data(start, goal, MAP, map_arr, args):
    '''
    synthesize data for one step. 
    
    
    @params
    start-- (x,y) position of start node
    goal-- (x,y) position of goal node
    MAP-- Shapely map representation
    map_arr-- Numpy map representation. 
    args-- hyperparameters from Args class

    '''
    
    # Get path if one is available
    try:
        path = get_path(start, goal, map_arr, False)
    except:
        return
    
    sensor_readings = []
    goals, relative_goals, polar_goals = [], [], []
    directions = []
    steering = []
    
    prev = start
    for i, loc in enumerate(path):
        # Get direction to next cell
        direction = (loc[0] - prev[0], loc[1] - prev[1])
        
        # Get rotation
        offset = 0
        if direction != (0, 0): # Start node
            offset = args.inv_dirs[direction]
        rot = np.pi/2 * offset
        # Get laser scan
        ls,_ = synthetic_sensor(MAP, (loc[0]+0.5, loc[1]+0.5), direction, args)
        sensor_readings.append(ls)
        # Get goal in odom
        goal_loc = (goal[0]-loc[0], goal[1]-loc[1])
        goals.append(goal_loc)
        if args.robot_type == 'ddr':  
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
                ds = args.inv_dirs[direction] - args.inv_dirs[directions[len(directions)-1]]
            elif len(directions) == 1: # First node after start
                ds = args.inv_dirs[direction]
            else: ds = 0 # Start node
            
            steering.append(ds)
            
            # Get movement to next cell
        directions.append(direction)
        prev=loc
    if args.robot_type == 'ddr':    
        if args.use_polar:
            return np.array(sensor_readings), np.array(polar_goals), np.array(steering[1:] + steering[:1]), path # Rotated 0 to stop
        return np.array(sensor_readings), np.array(relative_goals), np.array(steering[1:] + steering[:1]), path
    else:
        return np.array(sensor_readings), np.array(goals), np.array(directions[1:] + directions[:1]), path

def synthesize_train_set(MAPs, args):
    '''
    Main generation loop for creating a full dataset. 

    @params
    MAPs-- List of training maps 
    args-- System hyperparameters from class Args
    '''
    
    df = []
    for i in tqdm(range(args.num_train_paths)):
        # TODO: Generalize to any map shape
        min_node, max_node = 1, args.map_size[1]-1
        start = (random.randint(min_node, max_node), random.randint(min_node, max_node))
        goal = (random.randint(min_node, max_node), random.randint(min_node, max_node))
        MAP, map_arr = random.choice(MAPs)
        # If path is available, get training info
        try:

            (sensor_readings, relative_goals, steering, path) = synthesize_data(start, goal, MAP, map_arr, args)
            if args.robot_type == 'omni':
                train = np.concatenate((sensor_readings, relative_goals, steering), axis=1)
            else:
                train = np.concatenate((sensor_readings, relative_goals, np.expand_dims(steering, axis=1)), axis=1)
            df.append(train)
            
        except Exception as e:
            # No path found
            continue
    return pd.DataFrame(np.vstack(df))


def quadFits(map_np, sx, sy, rx, ry, margin):
        """
        looks to see if a quad shape will fit in the grid without colliding with any other tiles
        used by placeRoom() and placeRandomRooms()
         
        @params
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

    @params
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
import matplotlib.pyplot as plt
from shapely.geometry import LineString 
import imageio, os
from synthesize.synthesize import synthetic_sensor
import numpy as np 
def plot_path_(MAP, robot_location, goal_location, path):
    '''
    Plot path on map using shapely. Not currently used. 
    '''
    fig = plt.figure(frameon=False)
    fig.set_size_inches(6,6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    
    for index, l in enumerate(MAP): 
        if index != 0:
            ax.fill(*l.xy, alpha=1)
        else:
            ax.plot(*l.xy, alpha=1)
    
    ax.plot(*LineString(path).xy)
    ax.scatter(robot_location[0], robot_location[1], s=30, alpha=1.0) 
    ax.scatter(goal_location[0], goal_location[1], s=30, alpha=1.0) 


marker_dict = {0:">", np.pi/2:"^", np.pi:"<", 3*np.pi/2:"v"}
def plot_path_with_lines(pred_path, MAP, directions, args, file_out = 'out.gif'):
    '''
    Given predicted path nodes and map, 
    plot the path and the sensor readings for each node
    '''
    # Dir for final GIF
    os.makedirs(args.img_root, exist_ok=True)
    # Dir for intermediate imgs
    os.makedirs(args.img_root +'/imgs', exist_ok=True)
    # Update node positions to be at center of shapely cell
    pred_path = [(p[0]+0.5, p[1]+0.5) for p in pred_path]
    # Save filenames for GIF creation
    filenames=[]
    offset=0
    for i, node in enumerate(pred_path):
        # Create fig
        fig = plt.figure(frameon=False)
        fig.set_size_inches(15,10)
        # Plot start and goal
        plt.scatter(*pred_path[0], s=100, alpha=1.0)
        plt.scatter(*pred_path[-1], s=100, alpha=1.0)
        # Plot path
        plt.plot(*LineString(pred_path).xy)
        
        # Plot properly oriented triangle
        if directions[i] != (0, 0):
            offset = args.inv_dirs[directions[i]]
        rot = np.pi/2 * offset 
        plt.scatter(*node, s=100, alpha=1.0, marker = marker_dict[rot])

        # Get sensor data and plot it
        distances, lines = synthetic_sensor(MAP, node, directions[i], args)
        for index, l in enumerate(MAP): 
            if index != 0:
                plt.fill(*l.xy, alpha=1)
            else:
                plt.plot(*l.xy, alpha=1)
        for k, line in enumerate(lines):
            plt.plot(*line.xy, alpha=0.25)

        filenames.append(args.img_root+'/imgs/img_{}.png'.format(i))
        plt.savefig(args.img_root+'/imgs/img_{}.png'.format(i))
        plt.close()
    
    # Make GIF
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(args.img_root + '/' + file_out, images)
    


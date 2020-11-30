
def make_tuple(x):
    '''
    Function used to turn 2 pandas columns into
    a tuple in one column. See create_classification_problem
    for use. 
    '''
    return (x.iloc[0], x.iloc[1])

def get_inv_dirs(x):
    '''
    Function to encode direction tuple according to hard-coded encoding
    '''
    return inv_dirs[x]

def get_steer(x):
    '''
    Get either ccw, cw or straight
    '''
    if x == 1 or x == -3: # ccw
        return 1
    elif x == -1 or x == 3: # cw
        return 2
    else: return 0 # straight

def create_classification_problem(df, args, one_hot=False):
    '''
    Now that training data has been synthesized, 
    prepare data for use with ML model. 
    '''

    if args.robot_type == 'omni':
        df['out'] = df[[args.num_sensor_readings+2, args.num_sensor_readings+3]].apply(make_tuple, axis=1)
        df['out'] = df['out'].astype(str)
        
        # Drop the sample where we are at the target location. 
        # We don't want to learn to stay still. 
        df = df[df['out']!='(0.0, 0.0)']
     
        # Label encode targets
        df['out'] = args.enc.fit_transform(df['out'])
        df.drop([args.num_sensor_readings+2, args.num_sensor_readings+3], axis=1, inplace=True)
    else:

        df['out'] = df[args.num_sensor_readings+2]
        df['out'] = df['out'].astype(str)
        df['out'] = args.enc.fit_transform(df['out'])
        
        df.drop([args.num_sensor_readings+2], axis=1, inplace=True)

    df = df.sample(frac=1)
    
    return df
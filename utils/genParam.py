import numpy as np

LOWER_BOUND = 0
UPPER_BOUND = 1

def getMean(lower_bound, upper_bound):
    return 0.5 * (lower_bound + upper_bound)

def getUniformRandomSamples(lower_bound, upper_bound, num_samples):
    return np.random.rand(num_samples) * (upper_bound - lower_bound) + lower_bound

# TODO: This function is kind of weird. It generates samples, but it also leaves certain parameters
# as 'members', which are handled later. Look at setSuspensionParam as an example.
def genParam(data, new_param=None, num_samples=1, gen_mean=False):
    """
    Recursively parses through a yaml

    Args:
        data (dict): generated from yaml file containing parameters
        new_param ()
        num_samples (int)
        gen_mean (bool): if true, 
    """

    # At first function call, new_param doesn't exist. Create a dictionary to pass through calls.
    if new_param is None:
        new_param = {}
    
    # Iterates through the yaml file, where key = main headings
    # ("drive, steer, suspension, tireContact, inertia")
    # and value = dictionaries
    for key, value in data.items():
        # i.e. members: ['frtire', 'fltire', 'brtire', 'bltire'] 
        # new_param['members'] = ['frtire', 'fltire', 'brtire', 'bltire']
        if key == 'members':
            new_param[key] = value.copy()
        
        # For yaml files with headings (i.e. robotRange.yaml), this populates new_param with the heading, 
        # then recursively calls genParam with data = all the members.
        # Generate samples for each member (assumes all members in yaml will use the same numbers)
        elif isinstance(value, dict):
            new_param[key] = {}
            n_members = len(value.get('members', [None]))            
            genParam(value, new_param[key], n_members, gen_mean=gen_mean)

        # Assumes that when a list is in a config file, it'll be in [min, max] format.
        elif isinstance(data[key], list):
            lower_bound = value[LOWER_BOUND]
            upper_bound = value[UPPER_BOUND]
            if gen_mean:
                new_param[key] = [getMean(lower_bound, upper_bound)] * num_samples
            else:
                new_param[key] = getUniformRandomSamples(lower_bound, upper_bound, num_samples).tolist()
        
        else:
            new_param[key] = value if num_samples == 1 else [value] * num_samples

    return new_param
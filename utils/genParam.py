import numpy as np

LOWER_BOUND = 0
UPPER_BOUND = 1

def getMean(lower_bound, upper_bound):
    return 0.5 * (lower_bound + upper_bound)

def getUniformRandomSamples(lower_bound, upper_bound, num_samples):
    return np.random.rand(num_samples) * (upper_bound - lower_bound) + lower_bound

def genParam(data, new_param=None, num_samples=1, gen_mean=False):
    """
    Iterates through a YAML file read into a dictionary and recursively generates a dictionary of parameters.
    If the YAML file does not contain lists [min_value, max_value] or multiple members, then the function essentially
    reconstructs the same dictionary file.

    This function was used to construct different sets of parameters for domain randomization.

    Params:
        data: sub-dictionary of YAML file we're parsing.
        new_param: sub-dictionary to populate.
        num_samples: for use when multiple members are specified.
        gen_mean: if True, uses the mean of the minimum and maximum values to generate parameters. If not, uniformly 
        samples between [min, max]
    """
    if new_param is None:
        new_param = {}
        
    for key, value in data.items():
        # 'members' indicate more than one member under a parameter. i.e. ['frtire', 'fltire', 'brtire', 'bltire'] 
        if key == 'members':
            new_param[key] = value.copy()
        
        # Generate samples for each member
        elif isinstance(value, dict):
            new_param[key] = {}
            n_members = len(value.get('members', [None]))            
            genParam(value, new_param[key], n_members, gen_mean=gen_mean)

        # Assumes that when a list is in a config file, it'll be in [min, max] format.
        elif isinstance(data[key],list):
            lower_bound = value[LOWER_BOUND]
            upper_bound = value[UPPER_BOUND]
            if gen_mean:
                new_param[key] = [getMean(lower_bound, upper_bound)] * num_samples
            else:
                new_param[key] = getUniformRandomSamples(lower_bound, upper_bound, num_samples).tolist()
        
        else:
            new_param[key] = value if num_samples == 1 else [value] * num_samples

    return new_param
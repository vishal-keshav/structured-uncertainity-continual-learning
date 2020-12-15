"""Generic utilities, used by run.py and other scripts
"""

def get_configurations(configs):
    """Builds a list of all possible configuration dictionary
    from one configuration dictionary that contains all values for a key

    Args:
        configs (dict[str: List[any]]): a dictionary of configurations
    
    Returns:
        List: A list of configuration
    """
    if type(configs) == list:
        return configs
    all_configs = []
    config_keys = list(configs.keys())

    def recursive_config_list_builder(param_type_index, current_param_dict,
                                      param_list):
        if param_type_index == len(param_list):
            all_configs.append(current_param_dict)
        else:
            if type(configs[config_keys[param_type_index]]) == list:
                for val in configs[param_list[param_type_index]]:
                    temp = current_param_dict.copy()
                    temp[param_list[param_type_index]] = val
                    recursive_config_list_builder(param_type_index+1, temp,
                                                  param_list)
            else:
                temp = current_param_dict.copy()
                temp[param_list[param_type_index]] = configs[
                                            config_keys[param_type_index]]
                recursive_config_list_builder(param_type_index+1, temp,
                                              param_list)

    recursive_config_list_builder(0, dict(), config_keys)
    return all_configs

def stringify(vals):
    """Return a string version of vals (a list of object implementing __str__)

    Args:
        vals (List[any]): List of object that implements __str__

    Returns:
        str: A string representation
    """
    if type(vals) == list:
        return '_'.join([str(e) for e in vals])
    else:
        return str(vals)

def valuefy(strings, type_cast=None):
    """Return a list of value, type casted by type_cast list
    
    Args:
        strings (str): A string with value seperated by '_'
        type_cast (List[types]): A list with type for each elements
    
    Returns:
        List[any]: A list of values typecasted by type_cast array elements.
                   If type_cast is None, all elements are type casted to int.
    """
    vals_string = strings.split('_')
    if type_cast is None:
        type_cast = [int]*len(vals_string)
    assert len(vals_string) == len(type_cast)
    return [t(e) for e, t in zip(vals_string, type_cast)]
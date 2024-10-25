import yaml
import copy
import os
import numpy as np
from pathlib import Path

class ObjDict(dict):
    def __init__(self, ins_dict=None):
        super().__init__()
        if ins_dict is not None:
            for n in ins_dict.keys(): 
                self[n] = ins_dict[n]

    def insert_dict(self, ins_dict):
        for n in ins_dict.keys(): 
            self[n] = ins_dict[n]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def create_directory(pth, drop_last = False):
    try:
        os.makedirs(pth)
    except OSError as e:
        pass
    return pth

def stringify(strval):
    print(type(strval), strval)
    if type(strval) == str:
        strval = strval.replace("/", "_")
        strval = strval.replace(".", "_")
    if type(strval) == list: return "_".join([str(sv) for sv in strval])
    return str(strval)


def merge_dict(d1, d2):
    new_dict = dict()
    for key in d1:
        if key in d2: new_dict[key] = d1[key] + d2[key]
        else: new_dict[key] = d1[key]
    for key in d2:
        if key not in d1: new_dict[key] = d2[key]
    return new_dict

REPO_PATH = Path(__file__).resolve().parents[1]

def write_multi_config(multi_pth):
    # read in base config

    # read in hyperparameter grid file
    with open(multi_pth, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print("error: ", exception)
    data = ObjDict(data)
    bash_filename = data["meta"]["bash_filename"] + ".sh"
    alt_path, alt_path_endpoint = "", ""
    if "path_endpoint" in data["meta"]:
        alt_path_endpoint = data["meta"]["path_endpoint"]
        alt_path = "alt_path=" + alt_path_endpoint
    gpus = [int(gpu) for gpu in (data["meta"]["gpu"].split(" ") if type(data["meta"]["gpu"]) == str else [data["meta"]["gpu"]])] # the gpu to use
    match = data["meta"]["match"] # if 0, will one hot the parameters, if 1 will match up parameters, if 2 will run grid search
    default_settings = data["meta"]["default_settings"] if "default_settings" in data["meta"] else ""
    num_trials = data["meta"]["num_trials"]
    simul_run = data["meta"]["simul_run"] if 'simul_run' in data['meta'] else -1  # runs simul_run operations simultaniously
    del data["meta"]

    def convert_key(key_string):
        print(key_string)
        try:
            keyvals = key_string.split(',')
        except AttributeError as e:
            return [key_string] # not actually a string, just return
        try: # convert keyvals to int
            vals = list()
            for k in keyvals:
                vals.append(int(k))
            return vals
        except ValueError as e:
            pass
        try: # convert keyvals to floats
            vals = list()
            for k in keyvals:
                vals.append(float(k))
            return vals
        except ValueError as e:
            pass
        try: # convert keyvals to lists of floats
            vals = list()
            for kl in keyvals:
                vals.append([float(k) for k in kl.split(" ")])
            return vals
        except ValueError as e:
            pass
        return keyvals # assumes that they are just strings

    # for every key, convert it to a path, then convert that path into a hydra file
    def get_all_keys(data_up_to, keys_up_to):
        all_keys = list()
        matchkey_setting = dict()
        for k in data_up_to.keys():
            if type(data_up_to[k]) != dict:
                print(keys_up_to, k)
                data_settings = convert_key(data_up_to[k])
                for kvalues in data_settings:
                    all_keys.append(keys_up_to + [k] + [kvalues])
                    if tuple(keys_up_to + [k]) in matchkey_setting: matchkey_setting[tuple(keys_up_to + [k])].append(kvalues)
                    else: matchkey_setting[tuple(keys_up_to + [k])] = [kvalues]
            else:
                new_keys, mks = get_all_keys(data_up_to[k], keys_up_to + [k])
                all_keys += new_keys
                matchkey_setting = merge_dict(mks, matchkey_setting)
        return all_keys, matchkey_setting
    all_keys, matchkey = get_all_keys(data, list())
    print(matchkey, all_keys)

    # write the yaml config files (if setting a parameter greater than depth 0)
    for keylist in all_keys:
        if len(keylist) <= 2:
            continue
        last_two = keylist[-2:] # takes off the last two to put in the file
        keylist = [str(REPO_PATH)] + ["configs"] + keylist[:-2]
        keypath = os.path.join(*keylist)
        try:
            os.makedirs(os.path.join(*keylist))
        except OSError as e:
            pass
        writeline = last_two[-2] + ": " + str(last_two[-1])
        print(last_two[-2], last_two[-1], stringify(last_two[-1]))
        with open(os.path.join(keypath, last_two[-2] + '_' + stringify(last_two[-1]) + ".yaml"), 'w') as f:
            f.write(writeline)
    # write the bash files
    bashlist = [str(REPO_PATH)] + ["bashes"]
    bashpath = os.path.join(*bashlist)
    try:
        os.makedirs(os.path.join(*bashlist))
    except OSError as e:
        pass
    if match == 0: # create a separate line for every trial, every individual hp setting
        base_string = "python train_HRL.py " + default_settings + " "
        with open(os.path.join(bashpath, bash_filename), 'w') as f:
            counter = 1
            for keylist in all_keys:
                for i in range(num_trials):
                    append_string = base_string
                    print_target = ""
                    if len(alt_path_endpoint) > 0: # put the parameter being affected in alt_path so it's visible to tensorboard
                        try:
                            os.makedirs(os.path.join(alt_path_endpoint, "logs"))
                        except OSError as e:
                            pass
                        print_target = " > " + os.path.join(alt_path_endpoint, "logs", "_".join(keylist[:-1] + [stringify(keylist[-1])]) + "_trial" + str(i) + ".txt")
                        append_string += os.path.join(alt_path, "_".join(keylist[:-1] + [stringify(keylist[-1])])) + "_trial" + str(i) + " "
                    if len(keylist) > 2: # use folder access terminology
                        # cycle through seeds and gpus
                        append_string += "seed=" + str(np.random.randint(100000)) + " " \
                            + "cuda_id=" + str(gpus[(counter-1) % len(gpus)]) + " " \
                            + "+" + "/".join(keylist[:-2]) + "=" + keylist[-2] + "_" + stringify(keylist[-1]) + print_target
                    else: # directly change parameter here
                        append_string += "seed=" + str(np.random.randint(100000)) + " " \
                            + "cuda_id=" + str(gpus[(counter-1) % len(gpus)]) + " " \
                            + keylist[-2] +"="+ stringify(keylist[-1]) + print_target
                    if counter % simul_run != 0: append_string += " &\nsleep 3\n"
                    else: append_string += "\n"
                    f.write(append_string)
                    counter += 1
    else: # use hydra's grid search settings # TODO: ignores simul_run
        append_string = "python train_HRL.py " + alt_path + default_settings + " "
        with open(os.path.join(bashpath, bash_filename), 'w') as f:
            for k in matchkey.keys():
                if len(k) > 1:
                    append_string += "/".join(k) + "=" + ",".join(map(str, matchkey[k])) + " "
                else:
                    append_string += k + "=" + ",".join(map(str, matchkey[k])) + " "
            for i in range(num_trials):
                write_string = append_string + "seed=" + str(np.random.randint(100000)) + " " \
                                + "cuda_id=" + str(gpus[(i) % len(gpus)]) + " -m\n"
                f.write(write_string)
from tianshou.data import Batch
from State.extractor import Extractor
from State.utils import ObjDict
import numpy as np
import sys, os

def get_factor_params(extractor):
    factor = ObjDict()
    factor.first_obj_dim, factor.single_obj_dim, factor.object_dim, factor.all_obj_dim, factor.named_first_obj_dims = extractor._get_dims()
    factor.name_idxes = extractor.name_idxes
    return factor

class ACExtractor(Extractor):
    '''
    extracts observations from factored states, or vise versa.
    Also can be used to convert flattened collections of states to pick out particular objects
    obs_selector is used to select from factored states with ID padding
    target_selector is used without (to limit the number of targets to predict, which artificially raises the log likelihood) 
    '''
    def __init__(self, args, environment, config):
        super().__init__(environment)
        # needed components in extractor: get_index, num_objects, convert_data
        # get the environment specific components
        # TODO: the names are just numbers right now, and each name is a different class, but that could be altered
        if os.path.join(sys.path[0],"Causal", "ac_infer") not in sys.path: sys.path.append(os.path.join(sys.path[0],"Causal", "ac_infer"))
        from Causal.ac_infer.State.pad_selector import PadSelector
        self.names = np.arange(config.num_objects)
        self.object_names = np.arange(config.num_objects)
        self.sizes = {n: self.longest for n in self.names}
        self.instanced = {n: 1 for n in self.names}
        self.num_objects = config.num_objects

        # store proximity values here
        self.pos_size = 2 # environment.pos_size # TODO: make sure this is actually present in environments, otherwise assume 2d
        self.sp = args.state

        # initialze the two main selectors, where most of the logic is
        self.obs_selector = PadSelector(self.sizes, self.instanced, self.names, True)
        self.target_selector = PadSelector(self.sizes, self.instanced, self.names, True)
        self.unappend_selector = PadSelector(self.sizes, self.instanced, self.names, False)

        # comput the important dimensions
        self.pad_dim = self.longest
        self.append_dim = self.num_objects # len(list(environment.object_sizes.keys())) # I'm not sure we can actually append IDs, if so, TODO making that work properly
        self.expand_dim = self.pad_dim
        self.key_expand_dim = self.pad_dim
        self.first_obj_dim, self.target_dim,self.object_dim,self.all_obj_dim,self.named_first_obj_dims = self._get_dims()
        self.name_idxes = {name: self.get_index_single(name) for name in self.object_names}
    
    def get_index_single(self, name):
        if name in self.instanced and self.instanced[name] > 1:
            return [self.names.index(name + str(i)) for i in range(self.instanced[name])]
        else: return [self.names.index(name)]
        
    def get_index(self, name):
        if type(name) == list: # MUST send multiinstanced as list, to ensure return type usage
            return sum([self.get_index_single(n) for n in name], start=list())
        return self.names.index(name)

    def get_name(self, idxes):
        if type(idxes) == list:
            return [self.names[idx] for idx in idxes]
        return self.names[idxes]

    def _get_dims(self):
        first_obj_dim = self.key_expand_dim * len(self.names)
        target_dim = self.key_expand_dim
        all_obj_dim = self.expand_dim * len(self.names)
        named_first_obj_dims = Batch()
        for k in self.sizes.keys():
            named_first_obj_dims[k] = int(target_dim * self.instanced[k])
        return int(first_obj_dim), int(target_dim), int(self.expand_dim), int(all_obj_dim), named_first_obj_dims

    def get_selectors(self):
        return self.obs_selector, self.target_selector

    def get_obs(self, factored_state, names=[]):
        return self.obs_selector(factored_state, names)

    def get_target(self, factored_state, names=[]):
        return self.target_selector(factored_state, names)

    def get_named_target(self, flat_state, names=[]):
        if len(names) == 0: return flat_state
        flat_state = flat_state.reshape(flat_state.shape[0], -1, self.target_dim)
        return flat_state[:,self.get_index(names)]

    def get_named_obs(self, flat_state, names=[]):
        if len(names) == 0: return flat_state
        flat_state = flat_state.reshape(flat_state.shape[0], -1, self.object_dim)
        return flat_state[:,self.get_index(names)]
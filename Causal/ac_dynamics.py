import copy, os, sys
from torch import nn
import numpy as np
from State.utils import ObjDict
from tianshou.data import Batch
from Causal.dynamics import Dynamics
from Causal.ac_extractor import ACExtractor

def regenerate(args, environment, config):
    # instead of calling the regenerate in ac_infer, call one that uses the environment we received in init
    # also provides just enough in extractor and normalization so that they can be used
    # TODO: actually implement
    # needed components in normalization: (I think this component is unused)
    normalization = None
    extractor = ACExtractor(args, environment, config)
    args.factor.first_obj_dim, args.factor.single_obj_dim, args.factor.object_dim, args.factor.all_obj_dim, args.factor.named_first_obj_dim = extractor._get_dims()
    args.factor.num_objects = len(environment.all_names)
    args.factor.name_idxes = extractor.name_idxes
    args.factor.name_idx = -1
    return extractor, normalization

class DynamicsAC(Dynamics):
    '''
    Returns the ground truth causal graph by accessing data.true_graph
    '''
    def __init__(self, env, extractor, config):
        super().__init__(env, extractor)
        # initialize necessary components
        self.extractor = extractor
        # just in case it produces import errors, import here
        # TODO: ac_infer should be in this folder
        if os.path.join(sys.path[0],"Causal", "ac_infer") not in sys.path: sys.path.append(os.path.join(sys.path[0],"Causal", "ac_infer"))
        from Causal.ac_infer.Model.base_model import InferenceModel
        from Causal.ac_infer.Hyperparam.read_config import read_config
        from Causal.ac_infer.ActualCausal.Updater.update_params import compute_params
        from Causal.ac_infer.ActualCausal.Train.train_model import train_model
        from Causal.ac_infer.ActualCausal.train_loop import pretrain
        self.args = read_config(config.dynamics.ac.dynamics_config_path)
        extractor, normalization = regenerate(self.args, env, config)
        self.ac_model = InferenceModel(self.args, extractor, normalization, env)
        self.compute_params = compute_params
        self.params = self.compute_params(0, self.args, None, result=None)
        self.log_batch = [] # trace, valid could be used here
        self.pretrain_fn = pretrain
        self.train_model_fn = train_model

    def wrap(self, data):
        # takes in data and assigns the necessary keys for ac_infer to run
        # todo: actually implement
        batch = Batch(obs=data.obs, 
                      target=data.obs, 
                      target_next =data.obs_next, 
                      target_diff = data.obs_next - data.obs, 
                      valid=np.ones(self.args.factor.num_objects), 
                      trace = data.true_graph)
        return batch

    def __call__(self, data):
        batch = self.wrap(data)
        result = self.ac_model.infer(data, None, "all_mask").all_mask
        return result.mask_logits
    
    def pretrain(self, batch_size, buffer):
        self.pretrain_fn(self.args, self.ac_model, buffer, wrap=self.wrap)

    def update(self, batch_size, buffer):
        # TODO: instead of wrapping the buffer, which is probably expensive
        # edit the train_model and compute_params code so that it can take in the wrap function, and use that instead
        params = self.compute_params(self.update_counter, self.args, buffer, pretrain=False, result=result, params=params)
        result = self.train_model_fn(self.args, params, self.ac_model, buffer, log_batch=self.log_batch, wrap_function=self.wrap)
        self.update_counter += 1 # since it's not passed in
        return ObjDict({"bin_error": result.bin_error}) # TODO: store the desired values for record keeping here

    def compute_weight(self, data, dynamics, graph, true_graph, proximity):
        return np.ones((len(data), ))

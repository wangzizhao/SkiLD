# SkiLD: Unsupervised Skill Discovery Guided by Factor Interactions (NeurIPS 2024)

[Website](https://wangzizhao.github.io/SkiLD/) | [Paper](https://arxiv.org/pdf/2410.18416)

### Installation
1. Install required packages:
    ```
    conda create -y -n skild python=3.8
    pip -r requirements.txt
    ```
2. Install mini-behavior following instructions in  https://github.com/JiahengHu/mini-behavior-ihrl.

### Training
```
python train_HRL.py
```

### Code Structure
train_HRL:
- Use a config.yaml file in configs to choose the desired hyperparameters for hydra
- environment initialization in Initializers.init_utils
    - Supported environments accessed through @fn get_single_env (MiniGrid)
    - Multithreaded train and test environments through @fn init_logistics (separate train/test)
- dynamics initialization in Initializers.model
    - dynamics base class in Causal.dynamics: @method forward returns a graph given a batch, and can be trained with @method update
- policy initialization in Initializers.model
    - Option.hierarchical_model @class HierarchicalModel used as wrapper for upper and lower trainers (both are ts trainers)
        - forward, random_sample and check_rew_term_trunc used by Collector
        - update called by trainer
        - rewtermdone contained in upper and lower are used to determine the reward, termination and done flags from a given batch
        - Option.Terminate.rewtermdone rtt_init can be used for hyperparameter initiazation of rtt types
    - upper base class in Option.Upper.upper_policy
        - wraps around a tianshou learning algorithm
        - @method process_fn and post_process_fn used to arrange data for learning
        - lower_check_termination is used to determine if the lower policy reached a goal
        - action_sampler is used to sample the appropriate action spaces from lower policy logits
        - currently, only historical method is implemented
    - lower base class in Option.Lower.lower_policy
        - wraps around policies number of tianshou learning algorithms (and rewtermdones)
        - uses extractor to get the relevant information from the observation in factorized forms
- collector modified to grab the additional information for training, upper_data aggregation and HRL actions
        - upper_data aggregation in utils.upper_buffer_ready. adds the data if resampled (upper called for new action), terminated or done
        - aggregation code is handled by keeping a list of num_environment lists, each getting appended when upper_buffer_ready is called
- buffer modified to handle prioritized, weighted and hindsight
        - extends tianshou prioritized replay handling
        - weighted sampling buffer.weights
        - hindisght code reimplemented in buffer implementation
- trains using Option.ihrl_trainer IHRLTrainer:
    - takes a train step:
        - collects n_steps or n_episodes with Collector.collect
        - performs HierarchicalModel.update
        - performs Dynamics.update
    - takes a test step:
        - collects n_episodes for testing

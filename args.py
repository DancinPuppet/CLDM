class Args(object):
    def __init__(self):
        self.gpu_id = 1

        self.graph_type = 'twitter25'
        # self.graph_type = 'twitter15'
        # self.graph_type = 'twitter16'
        # self.graph_type = 'weibo'

        self.max_num_node = None
        self.features_with_profiles = False
        self.model_name = "CLDM"

        self.root_dir = './data/'
        self.data_path = self.root_dir + self.graph_type + '/'
        self.graphs_saved_path = self.root_dir + 'saved_graphs/' + self.graph_type + '_graph.pkl'
        self.save_path = './model_saves/checkpoints'
        self.output_save_path = './output'


        self.vae_noise = False
        self.resume_train = False
        if self.resume_train:
            self.checkpoint_path = f'./model_saves/checkpoints/{self.model_name}_{self.graph_type}_checkpoint_epoch_120.pt'
        self.load_model =False
        if self.load_model:
            self.load_model_path = f'./model_saves/checkpoints/{self.model_name}_checkpoint_epoch_300.pt'
            self.best_model_path = f'./model_saves/checkpoints/{self.model_name}_best_model.pt'

        self.num_workers = 1
        self.batch_size = 16
        self.num_batches = 1000
        self.node_features_dim = 5
        self.lr = 0.0001
        self.lr_rate = 0.1
        self.epochs = 300
        self.milestones = [100, 200]


        self.in_channels = self.node_features_dim
        self.hidden_channels = 32
        self.out_channels = 16
        self.heads = 4
        self.dropout = 0.2

        self.latent_dim = 16
        self.hidden_dim = 32
        self.node_features_hidden_dim = 32
        self.kl_start_weight = 0.001
        self.kl_end_weight = 0.1

        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.max_time_steps = 1000
        self.pred_start_weight = 0.5
        self.pred_end_weight = 1.5
        self.prime_predictor_hidden_dim = 128
        self.time_embed_dim = 64
        self.temperature = 1.0

        self.epoch_test = 1
        self.save_interval = 1000

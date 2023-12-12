import ml_collections

config_0 = ml_collections.ConfigDict()
config_0.patches = ml_collections.ConfigDict({'size': (16, 16)})
config_0.fea_size =16
config_0.width = 4
config_0.in_channels = 64 
config_0.mlp_dim = 3072 #3072=768*4
config_0.num_heads = 12
config_0.hidden_size = int(768)
config_0.attention_dropout_rate = 0.0
config_0.dropout_rate = 0.1
config_0.num_layers_enc = 12
config_0.activation = 'softmax'
config_0.zero_head=False
config_0.vis=False

config_1 = ml_collections.ConfigDict()
config_1.patches = ml_collections.ConfigDict({'size': (16, 16)})
config_1.fea_size =16
config_1.width = 4
config_1.in_channels = 64  
config_1.mlp_dim = 3072 
config_1.num_heads = 12
config_1.hidden_size = int(768)
config_1.attention_dropout_rate = 0.0
config_1.dropout_rate = 0.1
config_1.num_layers_enc = 12
config_1.activation = 'softmax'
config_1.zero_head=False
config_1.vis=False

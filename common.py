
seed = 114514
np.random.seed(seed)
random.seed(seed)
BATCH_SIZE = 512

hidden_dim = 16
epochs = 1
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
import time
import torch
from torch import nn
from SwinLSTM_D import SwinLSTM
from configs import get_args
from functions import test_np
from utils import set_seed, make_dir, init_logger
import numpy as np

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                     in_chans=args.input_channels, embed_dim=args.embed_dim,
                     depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                     num_heads=args.heads_number, window_size=args.window_size).to(args.device)

    test_file = np.load("test.npz")
    test_data = test_file["arr_0"]
    if test_data.shape[0] > 4:
        test_data = test_data[0:4,...]

    test_data = np.expand_dims(np.expand_dims(np.squeeze(test_data, axis=-1), axis=0), axis=2)
    test_tensor = torch.Tensor(test_data)
    print(test_data.shape)

    model.load_state_dict(torch.load('./{}/trained_model_state_dict'.format(args.model_dir), map_location=torch.device('cpu')))

    prediction = test_np(args, logger, 0, model, test_tensor, None, cache_dir, 20)

    with open("prediction.npz", "wb") as fp:
        np.save(fp, prediction)
        print("wrote file 'prediction.npz'")

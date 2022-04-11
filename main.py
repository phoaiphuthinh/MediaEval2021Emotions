import argparse

import torch

from data import load_data
from trainer import Trainer

def main(args):
    torch.set_num_threads(7)
    torch.manual_seed(1234)

    train_data, dev_df, test_df, path_audio = load_data(args)

    trainer = Trainer(args, train_dataset=train_data, val_df=dev_df, test_df=test_df, path_audio=path_audio)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print('KeyboardInterrupt. Stop training...')
    finally:
        trainer.load_model()
        trainer.evaluate("test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default=None, required=True, type=str, help="Path to data dir")
    parser.add_argument("--train", default=None, required=True, type=str, help="Train data")
    parser.add_argument("--valid", default=None, required=True, type=str, help="Valid data")
    parser.add_argument("--test", default=None, required=True, type=str, help="Test data")

    parser.add_argument("--name", nargs='+', required=True, help="List of model names")
    parser.add_argument("--size", default=640, required=True, type=int, help="Size")
    parser.add_argument("--forget_rate", default=0.05, required=True, type=int, help="Forget rate")
    
    parser.add_argument("--chunk_size", default=16, required=True, type=int, help="Chunk size")
    parser.add_argument("--cut_size", default=960, required=True, type=int, help="Cut size")

    parser.add_argument("--batch_size", default=32, required=True, type=int, help="Batch size")
    parser.add_argument("--epoch", default=200, required=True, type=int, help="Batch size")
    

    args = parser.parse_args()
    main(args)
from train_imagenet import ImageNetTrainer, make_config
from tqdm import tqdm
import torch as ch

class NoUpdateTrainer(ImageNetTrainer):
    def train_loop(self, epoch):
        print(f'Entering epoch: {epoch}')
        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            if ch.isnan(images).any():
                raise ValueError(f'{epoch}:{ix} --- nan images!')

            if ix > 20:
                break

if __name__ == "__main__":
    make_config()
    NoUpdateTrainer.launch_from_args()

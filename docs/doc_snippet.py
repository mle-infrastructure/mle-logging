from mle_logging import MLELogger
import torch.nn as nn
import matplotlib.pyplot as plt


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.fc1(x)
        return x


model = DummyModel()
fig, ax = plt.subplots()
extra = {"hi": "there"}


def run():
    log = MLELogger(time_to_track=['num_updates', 'num_epochs'],
                    what_to_track=['train_loss', 'test_loss'],
                    experiment_dir="experiment_dir/",
                    model_type="torch",
                    config_fname="config_1.json",
                    seed_id=1,
                    verbose=True)
    log.update({'num_updates': 10, 'num_epochs': 1},
               {'train_loss': 0.1234, 'test_loss': 0.1235},
               model, fig, extra, save=False)



if __name__ == "__main__":
    run()

import torch
from tqdm import tqdm

from cold_diffusion.utils import cold_diffusion_utils as utils
from cold_diffusion.utils.plotters import plot
from cold_diffusion.utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.evaluate_model()
            self.train_one_epoch()
            self.save_checkpoint(epoch)

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, x0 in pbar:
            t = utils.get_random_t(self.config)
            x0 = x0.to(self.device)
            t = t.to(self.device)
            x_noisy = utils.get_gaussian_blur_image(x0, t, self.config)
            x_denoise = self.model(x_noisy, t)
            loss, loss_dict = self.loss_fn(x_noisy, x_denoise)
            self.write_dict_to_tb(loss_dict, self.total_iters_train, prefix="train")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_iters_train += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            self.write_float_to_tb(self.optimizer.param_groups[0]["lr"], "train/lr", self.total_iters_train)
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self, plot_to_image=False):
        self.model.eval()
        T = self.config["NOISE"]["T"]
        num_samples = self.config["MISC"]["n_denoise_imgs"]
        for i in range(num_samples):
            xt = utils.sample_from_gmm(self.train_dataset.channel_mean, self.train_dataset.channel_std, self.config)
            xt = xt.unsqueeze(0).to(self.device)
            for t in tqdm(range(T - 1, 0, -1), desc=f"Sampling image {i+1}/{num_samples}"):
                t = torch.tensor(t).to(self.device)
                x0_pred = self.model(xt, t)
                xt = (
                    xt
                    - utils.get_gaussian_blur_image(x0_pred, t, self.config)
                    + utils.get_gaussian_blur_image(x0_pred, t - 1, self.config)
                )
                if t % self.config["MISC"]["plot_every_t_steps"] == 0:
                    denoised_img = plot(
                        xt[0],
                        save_figure=plot_to_image,
                        output_path=f"{self.config['IMG_OUT_DIR']}/{str(i).zfill(4)}/{str(t).zfill(5)}.png",
                    )
                    if not plot_to_image:
                        self.write_images_to_tb(
                            denoised_img, t, f"img/{str(self.total_iters_train).zfill(4)}/sample_{i}"
                        )
            self.logger.info(
                f"[Sample {i}] Final xt stats -> t: {t}, mean: {xt.mean()}, min: {xt.min()}, max: {xt.max()}"
            )

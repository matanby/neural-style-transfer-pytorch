from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
# noinspection PyPep8Naming
from torch.nn import functional as F
from tqdm import tqdm

import image_utils
from vgg import Vgg19


@dataclass
class StylizerConfig:
    # the weight of the content term in the total loss.
    lambda_content: float = 1

    # the weight of the style term in the total loss.
    # empirically good range: 10 - 100_000
    lambda_style: float = 100

    # the weight of the generated image's total variation
    # in the total loss. empirically good range: 0 - 1_000.
    lambda_tv: float = 10

    # the size of each step of the optimization process.
    step_size: float = 0.1

    # number of optimization iterations.
    iterations: int = 500

    # the weight of each convolutional block in the content loss.
    # These five numbers refer to the following five activations of
    # the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
    content_block_weights: Tuple[float] = (0.0, 0.0, 0.0, 1.0, 0.0)

    # the weight of each convolutional block in the style loss.
    # These five numbers refer to the following five activations of
    # the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
    style_block_weights: Tuple[float] = (1/5, 1/5, 1/5, 1/5, 1/5)

    # whether or not the optimization process should start with a
    # random initial image (True), or the input content image (False).
    random_initial_image: bool = False

    # the maximal allowed input image dimension. input images of
    # which max(H,W) is larger than this number will be downscaled appropriately.
    # this also defines the dimension of the generated stylized image.
    # Raising this value will allow creating larger stylized images, but
    # will also require more time and memory.
    max_input_dim: int = 512

    # the interval (number of iterations) after which an intermediate
    # result of the stylized image will be saved to the disk.
    save_interval: int = 50

    def update(self, **kwargs) -> 'StylizerConfig':
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise KeyError(f'Unknown configuration value: "{key}"')
        return self


class Stylizer:
    """
    A class that generates stylized images using the method presented in
    "A Neural Algorithm of Artistic Style" by Gatys et. al (2015)
    Paper: https://arxiv.org/abs/1508.06576.
    """

    def __init__(self, use_gpu: bool = True):
        gpu_available = torch.cuda.is_available()
        self._device = 'cuda' if use_gpu and gpu_available else 'cpu'
        self._vgg = Vgg19(use_avg_pooling=True).to(self._device)
        self._opt = None

    def stylize(
        self,
        content: np.ndarray,
        style: np.ndarray,
        config: Optional[StylizerConfig] = None,
    ) -> np.ndarray:
        """
        Creates a stylized image in which the content is taken from the input
        content image, and the style is taken from the input style image.
        :param content: The content image: np.ndarray of shape (h, w, 3) in range [0, 1].
        :param style: The style image: np.ndarray of shape (h, w, 3) in range [0, 1].
        :param config: (optional) an instance of `StyleTransferConfig`.
        :return: The generated stylized image: np.ndarray of shape (h, w, 3) in range [0, 1].
        """

        config = config or StylizerConfig()
        print(config)

        content_t = self._preprocess(content, config.max_input_dim)
        style_t = self._preprocess(style, config.max_input_dim)

        if config.random_initial_image:
            opt_t = torch.rand(
                size=content_t.shape,
                dtype=torch.float32,
                device=self._device,
                requires_grad=True,
            )
        else:
            opt_t = content_t.clone().requires_grad_(True)

        with torch.no_grad():
            content_features = self._vgg(content_t)
            style_features = self._vgg(style_t)

        self._opt = torch.optim.Adam([opt_t], lr=config.step_size)

        prog_bar = tqdm(range(1, config.iterations + 1))
        for i in prog_bar:
            loss = self._step(content_features, style_features, opt_t, config)
            mean_grad = opt_t.grad.abs().mean().item()
            prog_bar.set_description(f'Loss: {loss:.2f}, mean grad: {mean_grad:.7f}')

            if i % config.save_interval == 0:
                opt = self._postprocess(opt_t)
                image_utils.save(opt, f'images/progress/opt_{i}.jpg')

        opt = self._postprocess(opt_t)
        return opt

    def _preprocess(self, image: np.ndarray, max_dim: int) -> Tensor:
        h, w = image.shape[:-1]
        if max(h, w) > max_dim:
            resize_factor = max_dim / max(h, w)
            size = int(w * resize_factor), int(h * resize_factor)
            image = image_utils.resize(image, size)

        image_t = torch.tensor(image, device=self._device)
        image_t = image_t.unsqueeze(0).permute(0, 3, 1, 2)

        return image_t

    @staticmethod
    def _postprocess(image_t: Tensor) -> np.ndarray:
        image_t = image_t.permute(0, 2, 3, 1).squeeze(0)
        image = image_t.detach().cpu().numpy()
        return image

    def _step(
        self,
        content_features: List[Tensor],
        style_features: List[Tensor],
        opt_t: Tensor,
        config: StylizerConfig,
    ) -> float:

        opt_features = self._vgg(opt_t)
        content_loss = self._content_loss(opt_features, content_features, config.content_block_weights)
        style_loss = self._style_loss(opt_features, style_features, config.style_block_weights)
        tv_loss = self._tv_loss(opt_t)
        loss = content_loss * config.lambda_content + style_loss * config.lambda_style + config.lambda_tv * tv_loss
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        loss_f = loss.item()
        with torch.no_grad():
            opt_t.clamp_(0.0, 1.0)
        return loss_f

    @staticmethod
    def _content_loss(features_input: List[Tensor],
                      features_target: List[Tensor],
                      weights: Tuple[float]) -> Tensor:
        assert len(features_input) == len(features_target) == len(weights)
        device = features_input[0].device
        total = torch.zeros(1, dtype=torch.float32, device=device)

        num_features = len(features_input)
        for i in range(num_features):
            if weights[i] > 0:
                block_loss = F.mse_loss(features_input[i], features_target[i])
                block_loss = block_loss
                total = total + block_loss * weights[i]

        return total

    @staticmethod
    def _style_loss(features_input: List[Tensor],
                    features_target: List[Tensor],
                    weights: Tuple[float]) -> Tensor:
        assert len(features_input) == len(features_target) == len(weights)
        device = features_input[0].device
        total = torch.zeros(1, dtype=torch.float32, device=device)

        num_features = len(features_input)
        for i in range(num_features):
            if weights[i] > 0:
                gram_input = Stylizer._gram_matrix(features_input[i])
                gram_target = Stylizer._gram_matrix(features_target[i])
                block_loss = F.mse_loss(gram_input, gram_target)
                total = total + block_loss * weights[i]

        return total

    @staticmethod
    def _tv_loss(image: Tensor) -> Tensor:
        tv_loss = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean() + \
                  (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean()

        return tv_loss

    @staticmethod
    def _gram_matrix(features: Tensor) -> Tensor:
        n, c, h, w = features.shape
        x = features.view(n, c, h * w)
        y = features.view(n, c, h * w).permute(0, 2, 1)
        gram = torch.bmm(x, y)
        gram = gram / (h * w)
        return gram

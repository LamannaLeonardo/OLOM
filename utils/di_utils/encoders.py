from typing import Optional
import torch
import torch.nn as nn
from ding.model import ConvEncoder

from ding.torch_utils import Flatten, ResBlock
from ding.torch_utils.network.dreamer import Conv2dSame, DreamerLayerNorm
from ding.utils import SequenceType

class SmallGrayScaleEncoder(nn.Module):
    """
    Overview:
        The Convolution Encoder is used to encode 2-dim image observations.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [8, 16],
            # output_size: int = 256,
            activation: Optional[nn.Module] = nn.ReLU(),
            kernel_size: SequenceType = [4, 4],
            stride: SequenceType = [2, 2],
            # kernel_size: SequenceType = [2],
            # stride: SequenceType = [2],
            padding: Optional[SequenceType] = None,
            layer_norm: Optional[bool] = False,
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Initialize the ``Convolution Encoder`` according to the provided arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, plus one or more ``input size``.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent conv layers \
                and the final dense layer.
            - activation (:obj:`nn.Module`): Type of activation to use in the conv ``layers`` and ``ResBlock``. \
                Default is ``nn.ReLU()``.
            - kernel_size (:obj:`SequenceType`): Sequence of ``kernel_size`` of subsequent conv layers.
            - stride (:obj:`SequenceType`): Sequence of ``stride`` of subsequent conv layers.
            - padding (:obj:`SequenceType`): Padding added to all four sides of the input for each conv layer. \
                See ``nn.Conv2d`` for more details. Default is ``None``.
            - layer_norm (:obj:`bool`): Whether to use ``DreamerLayerNorm``, which is kind of special trick \
                proposed in DreamerV3.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResBlock`` \
                for more details. Default is ``None``.
        """
        super(SmallGrayScaleEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list
        if padding is None:
            padding = [0 for _ in range(len(kernel_size))]

        layers = []
        input_size = obs_shape[0]  # in_channel
        # input_size = obs_shape[-1]  # in_channel
        for i in range(len(kernel_size)):
            if layer_norm:
                layers.append(
                    Conv2dSame(
                        in_channels=input_size,
                        out_channels=hidden_size_list[i],
                        kernel_size=(kernel_size[i], kernel_size[i]),
                        stride=(2, 2),
                        bias=False,
                    )
                )
                layers.append(DreamerLayerNorm(hidden_size_list[i]))
                layers.append(self.act)
            else:
                layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i], padding[i]))
                layers.append(self.act)
            input_size = hidden_size_list[i]
        if len(self.hidden_size_list) >= len(kernel_size) + 2:
            assert self.hidden_size_list[len(kernel_size) - 1] == self.hidden_size_list[
                len(kernel_size)], "Please indicate the same hidden size between conv and res block"
        assert len(
            set(hidden_size_list[len(kernel_size):-1])
        ) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(len(kernel_size), len(self.hidden_size_list) - 1):
            layers.append(ResBlock(self.hidden_size_list[i - 1], activation=self.act, norm_type=norm_type))
        layers.append(Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.output_size = self.hidden_size_list[-1]  # outside to use
        self.mid = nn.Linear(flatten_size, self.output_size)

    def _get_flatten_size(self) -> int:
        """
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Returns:
            - outputs (:obj:`torch.Tensor`): Size ``int`` Tensor representing the number of ``in-features``.
        Shapes:
            - outputs: :math:`(1,)`.
        Examples:
            >>> conv = ConvEncoder(
            >>>    obs_shape=(4, 84, 84),
            >>>    hidden_size_list=[32, 64, 64, 128],
            >>>    activation=nn.ReLU(),
            >>>    kernel_size=[8, 4, 3],
            >>>    stride=[4, 2, 1],
            >>>    padding=None,
            >>>    layer_norm=False,
            >>>    norm_type=None
            >>> )
            >>> flatten_size = conv._get_flatten_size()
        """
        test_data = torch.randn(1, *self.obs_shape)
        # test_data = test_data.permute(0, 3, 1, 2)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output 1D embedding tensor of the env's 2D image observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Raw 2D observation of the environment.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - x : :math:`(B, C, H, W)`, where ``B`` is batch size, ``C`` is channel, ``H`` is height, ``W`` is width.
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]`` .
        Examples:
            >>> conv = ConvEncoder(
            >>>    obs_shape=(4, 84, 84),
            >>>    hidden_size_list=[32, 64, 64, 128],
            >>>    activation=nn.ReLU(),
            >>>    kernel_size=[8, 4, 3],
            >>>    stride=[4, 2, 1],
            >>>    padding=None,
            >>>    layer_norm=False,
            >>>    norm_type=None
            >>> )
            >>> x = torch.randn(1, 4, 84, 84)
            >>> output = conv(x)
        """
        x = self.main(x)
        x = self.mid(x)
        return x

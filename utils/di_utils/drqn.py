from typing import Union, Optional, Dict
import torch
import torch.nn as nn
from ding.model import FCEncoder, DuelingHead, DiscreteHead, MultiHead
from ding.model.template.q_learning import parallel_wrapper

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze

from utils.di_utils.encoders import SmallGrayScaleEncoder

@MODEL_REGISTRY.register('mydrqn')
class MyDRQN(nn.Module):
    """
    Overview:
        DQN + RNN = DRQN
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            lstm_type: Optional[str] = 'normal',
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            res_link: bool = False
    ) -> None:
        r"""
        Overview:
            Init the DRQN Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to ``Head``.
            - lstm_type (:obj:`Optional[str]`): Version of rnn cell, now support ['normal', 'pytorch', 'hpc', 'gru']
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
            - res_link (:obj:`bool`): use the residual link or not, default to False
        """
        super(MyDRQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        # FC Encoder
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = SmallGrayScaleEncoder(
                obs_shape,
                encoder_hidden_size_list,
                activation=activation,
                norm_type=norm_type,
                kernel_size=[4, 4],
                stride=[2, 2]
            )
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DRQN".format(obs_shape)
            )
        # LSTM Type
        self.rnn = get_lstm(lstm_type, input_size=head_hidden_size, hidden_size=head_hidden_size)
        self.res_link = res_link
        # Head Type
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def forward(self, inputs: Dict, inference: bool = False, saved_state_timesteps: Optional[list] = None) -> Dict:
        r"""
        Overview:
            Use observation tensor to predict DRQN output.
            Parameter updates with DRQN's MLPs forward setup.
        Arguments:
            - inputs (:obj:`Dict`):
            - inference: (:obj:'bool'): if inference is True, we unroll the one timestep transition,
                if inference is False, we unroll the sequence transitions.
            - saved_state_timesteps: (:obj:'Optional[list]'): when inference is False,
                we unroll the sequence transitions, then we would save rnn hidden states at timesteps
                that are listed in list saved_state_timesteps.

       ArgumentsKeys:
            - obs (:obj:`torch.Tensor`): Encoded observation
            - prev_state (:obj:`list`): Previous state's tensor of size ``(B, N)``

        Returns:
            - outputs (:obj:`Dict`):
                Run ``MLP`` with ``DRQN`` setups and return the result prediction dictionary.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit tensor with same size as input ``obs``.
            - next_state (:obj:`list`): Next state's tensor of size ``(B, N)``
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N=obs_space)`, where B is batch size.
            - prev_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - next_state(:obj:`torch.FloatTensor list`): :math:`[(B, N)]`

        Examples:
            >>> # Init input's Keys:
            >>> prev_state = [[torch.randn(1, 1, 64) for __ in range(2)] for _ in range(4)] # B=4
            >>> obs = torch.randn(4,64)
            >>> model = DRQN(64, 64) # arguments: 'obs_shape' and 'action_shape'
            >>> outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
            >>> # Check outputs's Keys
            >>> assert isinstance(outputs, dict)
            >>> assert outputs['logit'].shape == (4, 64)
            >>> assert len(outputs['next_state']) == 4
            >>> assert all([len(t) == 2 for t in outputs['next_state']])
            >>> assert all([t[0].shape == (1, 1, 64) for t in outputs['next_state']])
        """

        x, prev_state = inputs['obs'], inputs['prev_state']

        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        if inference:
            # x = x.permute(0, 3, 1, 2)  # move channel from last to first
            x = self.encoder(x)
            if self.res_link:
                a = x
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match head network input
            if self.res_link:
                x = x + a
            x = self.head(x)
            x['next_state'] = next_state
            return x
        else:
            assert len(x.shape) in [3, 5], x.shape
            # x = x.permute(0, 1, 4, 2, 3)  # move channel from last to first
            x = parallel_wrapper(self.encoder)(x)  # (T, B, N)
            if self.res_link:
                a = x
            lstm_embedding = []
            # TODO(nyz) how to deal with hidden_size key-value
            hidden_state_list = []
            if saved_state_timesteps is not None:
                saved_state = []
            for t in range(x.shape[0]):  # T timesteps
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, head_hidden_size)
                if saved_state_timesteps is not None and t + 1 in saved_state_timesteps:
                    saved_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(lstm_embedding, 0)  # (T, B, head_hidden_size)
            if self.res_link:
                x = x + a
            x = parallel_wrapper(self.head)(x)  # (T, B, action_shape)
            # the last timestep state including the hidden state (h) and the cell state (c)
            # shape: {list: B{dict: 2{Tensor:(1, 1, head_hidden_size}}}
            x['next_state'] = prev_state
            # all hidden state h, this returns a tensor of the dim: seq_len*batch_size*head_hidden_size
            # This key is used in qtran, the algorithm requires to retain all h_{t} during training
            x['hidden_state'] = torch.cat(hidden_state_list, dim=-3)
            if saved_state_timesteps is not None:
                # the selected saved hidden states, including the hidden state (h) and the cell state (c)
                x['saved_state'] = saved_state
            return x

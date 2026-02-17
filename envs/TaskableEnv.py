import abc
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Dict, Any, List, Optional, Tuple

import gymnasium as gym
# import gym
import numpy as np
import pygame

from PAL.Modeling.Operator import Operator
from utils.config import get_dom_cfg

ObsType = TypeVar("ObsType")
StateType = TypeVar("StateType")

# TODO: add inner class for defining the state?


@dataclass
class TaskableEnv(gym.Env, abc.ABC):
    """
    A taskable environment, i.e., an environment with a possibly changing goal
    """

    # Environment state
    _state: any

    # Set of goal states
    goal_states: set = field(default_factory=set)

    # Operators
    operators: List[Operator] = field(init=False)  # set in __post_init__

    # Environment initial seed
    seed: int | None = None

    # Current step number
    current_step: int = 0

    # Maximum number of steps, i.e. episode length
    max_steps: int = 100

    # For rendering
    render_mode: str = None
    _window = None
    _window_size = 280  # Size of the rendering window

    # Domain configuration
    cfg: Dict[str, Any] = field(default_factory=get_dom_cfg)

    def __post_init__(self):
        """
        Set environment state and seed through :meth: reset for reproducibility
        """

        # Set random seed for reproducibility
        if self.seed is not None:
            super().reset(seed=self.seed)

        assert self.operators is not None and len(self.operators) > 0, \
            (NotImplementedError(f"{self.__class__.__name__} must define a list of 'operators'"))

    @abstractmethod
    def _get_obs(self) -> ObsType:
        """
        Return an observation of the current environment state.
        Note observations are returned after executing :meth: step and :meth: reset.
        :return: observation of the current environment state
        """
        pass

    def _get_info(self) -> Dict[Any, Any]:
        """
        Return an additional information dictionary after
        executing :meth: step and :meth: reset.
        :return: additional information about the environment
        """
        return dict()

    @abstractmethod
    def _reward_fn(self, s, a, sp) -> float:
        """
        Reward function :math: R: S \times A \times S \rightarrow \mathbb{R}^+.
        Note this method should take into account the value of self.goal_states and work
        with possibly different sets of goal states.
        :param s: previous state
        :param a: executed action
        :param sp: current state
        :return: reward for executing action 'a' in state 's' and reaching state 'sp'
        """
        pass

    @abstractmethod
    def _randomize_state(self) -> None:
        """
        Randomize the current environment state. Note this is necessary when :meth: reset is executed
        """
        pass

    @abstractmethod
    def distance_to_success(self) -> float:
        """
        Distance from the current environment state to a goal state, i.e. optimal plan cost.
        :return: optimal plan cost
        """
        pass

    def close(self) -> None:
        """
        Cleanup when the environment is closed.
        """
        if self._window is not None:
            pygame.quit()
            pygame.display.quit()

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Resets the environment initial and goal states, and returns the observation of the initial state
        and the (possibly empty) additional information dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Set initial state
        if options is None:
            self._randomize_state()
            assert self.goal_states is not None, "No goal state, cannot compute reward."
        else:
            state = options.get('_state', None)
            if state is None:
                self._randomize_state()
            else:
                self._state = state

            # Set goal state
            goal_states = options.get('goal_states', None)
            if goal_states is not None:
                self.goal_states = goal_states
            assert self.goal_states is not None, "No goal state, cannot compute reward."

        return self._get_obs(), self._get_info()  # Observation, info dict

    def render(self):
        if self.render_mode == "human":  # TODO: manage different rendering modes
            return self._render_human()
        elif self.render_mode == "rgb_array":
            raise NotImplementedError

    def _render_human(self) -> None:
        """
        Render the environment into a pygame window (when rendering in 'human' mode)
        """
        assert self.render_mode == 'human'

        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((self._window_size, self._window_size))

        # Clear the screen
        self._window.fill((0, 0, 0))  # White background

        # Draw the currently visible digit
        img = pygame.surfarray.make_surface(np.transpose(self._get_obs(), (1, 0, 2)))  # Transpose axes

        # Resize the surface to fit the window (optional)
        image_surface = pygame.transform.scale(img, (self._window_size, self._window_size))

        # Blit the image surface onto the screen
        self._window.blit(image_surface, (0, 0))

        # Update display
        pygame.display.flip()

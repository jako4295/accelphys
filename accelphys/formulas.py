from typing import List, Union
import numpy as np
from scipy.constants import c


class Relativistic:
    @classmethod
    def gamma(
        cls,
        v: Union[float, np.ndarray] = None,
        beta: Union[float, np.ndarray] = None,
        q: float = None,
        p: float = None,
        e_0: float = None,
    ) -> Union[float, np.ndarray]:
        """
        The combinations of inputs that can be provided are:
        - v
        - beta
        - q, p, e_0

        :param v: velocity in m/s
        :param beta: relativistic beta. Velocity divided by speed of light.
        :param q: charge in elementary charge
        :param p: momentum in GeV/c
        :param e_0: rest energy in GeV
        """
        valid_combinations = [[v], [beta], [q, p, e_0]]
        index = cls._check_only_one_valid_comb(valid_combinations)

        match index:
            case 0:
                return 1 / np.sqrt(1 - (v**2 / c**2))
            case 1:
                return 1 / np.sqrt(1 - beta**2)
            case 2:
                return np.sqrt(((p * q) / e_0) ** 2 + 1)
            case _:
                raise ValueError("Error in valid_combinations definition")

    @classmethod
    def beta(
        cls,
        v: Union[float, np.ndarray] = None,
        gamma: Union[float, np.ndarray] = None,
        b_rho: Union[float, np.ndarray] = None,
        m: float = None,
        q: float = None,
    ) -> Union[float, np.ndarray]:
        """
        The combinations of inputs that can be provided are:
        - v
        - gamma
        - b_rho, m, q

        :param v: velocity in m/s
        :param gamma: Lorentz factor
        :param b_rho: magnetic rigidity in Tm
        :param m: atomic mass in atomic mass unit
        :param q: charge in elementary charge
        """

        valid_combinations = [[v], [gamma], [b_rho, q, m]]
        index = cls._check_only_one_valid_comb(valid_combinations)

        match index:
            case 0:
                return v / c
            case 1:
                return np.sqrt(1 - 1 / gamma**2)
            case 2:
                amu = 0.931493614838934  # GeV/c^2
                p = b_rho * 0.33
                e_0 = m * amu  # GeV
                return np.sqrt(1 - 1 / cls.gamma(p=p, q=q, e_0=e_0) ** 2)
            case _:
                raise ValueError("Error in valid_combinations definition")

    @classmethod
    def b_rho(
        p: Union[float, np.ndarray] = None,
        q: float = None,
        m: float = None,
        gamma: Union[float, np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        return p / q

    @classmethod
    def p_momentum(
        cls,
        e_tot: Union[float, np.ndarray] = None,
        e_0: float = None,
        q: float = None,
        b_rho: Union[float, np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        valid_combinations = [[e_tot, e_0], [b_rho, q]]
        index = cls._check_only_one_valid_comb(valid_combinations)

        match index:
            case 0:
                return np.sqrt(e_tot**2 - e_0**2) / c
            case 1:
                return b_rho * q
            case _:
                raise ValueError("Error in valid_combinations definition")

    @classmethod
    def _check_only_one_valid_comb(
        cls, valid_combinations: List[List[Union[float, np.ndarray]]]
    ) -> int:
        """
        Makes sure only one of the valid input combinations is passed.

        Args:
            valid_combinations (List[List[Union[float, np.ndarray]]]): Combin. to check

        Returns:
            int: index of the combination that is valid
        """
        # make sure only one valid combination is passed
        num_valids = 0
        valid_index = None

        for i, comb in enumerate(valid_combinations):
            if all([_param is not None for _param in comb]):
                num_valids += 1
                valid_index = i

        if num_valids != 1:
            raise ValueError("You must specify exactly one combination of parameters")

        return valid_index

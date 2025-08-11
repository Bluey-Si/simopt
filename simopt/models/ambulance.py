"""
Summary
-------
Simulate and optimize the average response time in a multi-base ambulance dispatch system.
The system includes a set of fixed ambulance bases and a set of variable bases with decision-variable coordinates.
The objective is to minimize the expected response time by optimizing the locations of the variable bases.
"""

from __future__ import annotations

from typing import Callable, Final
import numpy as np
from simopt.base import ConstraintType, Model, Problem, VariableType
from mrg32k3a.mrg32k3a import MRG32k3a

AVAILABLE = 0
BUSY = 1

NUM_FIXED: Final[int] = 3
NUM_VARIABLE: Final[int] = 2

class Ambulance(Model):
    @property
    def name(self) -> str:
        return "AMBULANCE"

    @property
    def n_rngs(self) -> int:
        return 4

    @property
    def n_responses(self) -> int:
        return 1

    @property
    def specifications(self) -> dict[str, dict]:
        # In specifications, use NUM_FIXED and NUM_VARIABLE to define default locs.
        return {
            "fixed_base_count": {
                "description": "Number of fixed bases",
                "datatype": int,
                "default": NUM_FIXED,
            },
            "variable_base_count": {
                "description": "Number of variable bases",
                "datatype": int,
                "default": NUM_VARIABLE,
            },
            "fixed_locs": {
                "description": "Fixed base coordinates, [x0, y0, x1, y1, ...]",
                "datatype": list,
                "default": [15, 15, 5, 15, 5, 5],
            },
            "variable_locs": {
                "description": "Variable base coordinates, as [x0, y0, x1, y1, ...]",
                "datatype": list,
                "default": [3, 10, 3, 10],
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "fixed_base_count": self.check_fixed_base_count,
            "variable_base_count": self.check_variable_base_count,
            "fixed_locs": self.check_fixed_locs,
            "variable_locs": self.check_variable_locs,
        }

    def __init__(self, fixed_factors: dict | None = None) -> None:
        # First call parent constructor, this will populate self.factors
        super().__init__(fixed_factors)

    def check_fixed_base_count(self) -> None:
        if self.factors["fixed_base_count"] < 0:
            raise ValueError("fixed_base_count must be >= 0.")

    def check_variable_base_count(self) -> None:
        if self.factors["variable_base_count"] <= 0:
            raise ValueError("variable_base_count must be > 0.")

    def check_fixed_locs(self) -> None:
        expected_length = 2 * self.factors["fixed_base_count"]
        if len(self.factors["fixed_locs"]) != expected_length:
            raise ValueError(
                f"The length of fixed_locs must be {expected_length} (2 * fixed_base_count)."
            )

    def check_variable_locs(self) -> None:
        expected_length = 2 * self.factors["variable_base_count"]
        if len(self.factors["variable_locs"]) != expected_length:
            raise ValueError(
                f"The length of variable_locs must be {expected_length} (2 * variable_base_count)."
            )

    def replicate(self, rng_list: list[MRG32k3a]) -> tuple[dict, dict]:
        """
        Simulate a single replication for the current model factors.

        Returns
        -------
        responses : dict
            Dictionary containing "avg_response_time".
        gradients : dict
            Dictionary containing gradients of "avg_response_time" w.r.t. variable base locations.
        """

        # Unpack model factors
        fixed_base_count = self.factors["fixed_base_count"]
        variable_base_count = self.factors["variable_base_count"]
        fixed_locs = self.factors["fixed_locs"]
        variable_locs = self.factors["variable_locs"]

        # Build base positions as lists of (x, y) coordinates
        fixed_base_positions = [
            [fixed_locs[2 * i], fixed_locs[2 * i + 1]] for i in range(fixed_base_count)
        ]
        variable_bases = [
            [variable_locs[2 * i], variable_locs[2 * i + 1]] for i in range(variable_base_count)
        ]
        BASES = fixed_base_positions + variable_bases
        variable_START_INDEX = len(fixed_base_positions)

        # Simulation parameters
        SQUARE_WIDTH = 20.0  # Size of the service area (square side length)
        AMB_SPEED = 1.0  # Ambulance speed
        MEAN_INTERARRIVAL = 25.0  # Mean inter-arrival time of calls
        MEAN_SCENE_TIME = 10.0  # Mean scene service time
        SIM_LENGTH = 60 * 24.0 * 7  # Total simulation time (7 days in minutes)

        # Initialize random seed
        rng_arrival = rng_list[0]
        rng_scene = rng_list[1]
        rng_x = rng_list[2]
        rng_y = rng_list[3]

        # Initialize event list and simulation state
        event_list = []
        current_time = 0.0

        # Helper function to generate the next call arrival event
        def next_arrival(curr):
            return [
                curr + rng_arrival.expovariate(1.0 / MEAN_INTERARRIVAL),
                1,
                rng_x.uniform(0, SQUARE_WIDTH),
                rng_y.uniform(0, SQUARE_WIDTH),
                rng_scene.expovariate(1.0 / MEAN_SCENE_TIME)
            ]

        # Seed initial events: simulation end and first arrival
        event_list.append([SIM_LENGTH, 0, 0, 0, 0])  # Type 0: end of simulation
        event_list.append(next_arrival(0))  # Type 1: call arrival

        # Ambulance state array: (x, y, status), status = AVAILABLE or BUSY
        ambs = np.array([[bx, by, AVAILABLE] for bx, by in BASES])
        queued_calls = []
        active_calls = 0

        # Performance metrics
        total_response_time = 0.0
        num_calls = 0
        grad_total = np.zeros((variable_base_count, 2))

        # Main simulation loop
        while event_list:
            # Process next event
            event_list.sort(key=lambda e: e[0])
            event = event_list.pop(0)
            current_time = event[0]
            etype = event[1]

            if etype == 0:
                # End of simulation
                break
            elif etype == 1:
                # New call arrival
                active_calls += 1
                if active_calls > len(BASES):
                    # All ambulances busy → queue the call
                    queued_calls.append(event)
                else:
                    # Dispatch nearest available ambulance
                    times = [
                        np.sum(np.abs(amb[:2] - event[2:4])) / AMB_SPEED if amb[2] == AVAILABLE else float("inf")
                        for amb in ambs
                    ]
                    i = int(np.argmin(times))
                    ambs[i, 2] = BUSY
                    response_time = times[i]
                    total_response_time += response_time
                    num_calls += 1

                    # Accumulate gradient if ambulance is variable
                    if i >= variable_START_INDEX and i - variable_START_INDEX < variable_base_count:
                        j = i - variable_START_INDEX
                        dx = np.sign(ambs[i, 0] - event[2])
                        dy = np.sign(ambs[i, 1] - event[3])
                        grad_total[j, 0] += dx / AMB_SPEED
                        grad_total[j, 1] += dy / AMB_SPEED

                    # Schedule call completion
                    done_time = current_time + 2 * response_time + event[4]
                    event_list.append([done_time, 2, i, 0, 0])

                # Schedule next arrival
                event_list.append(next_arrival(current_time))

            elif etype == 2:
                # Ambulance becomes available after completing a call
                i = int(event[2])
                ambs[i, 2] = AVAILABLE
                active_calls -= 1
                if queued_calls:
                    # Dispatch queued call
                    qevent = queued_calls.pop(0)
                    travel = np.sum(np.abs(ambs[i, 0:2] - qevent[2:4])) / AMB_SPEED
                    queue_delay = current_time - qevent[0]
                    total_response_time += travel + queue_delay
                    num_calls += 1

                    # Accumulate gradient if ambulance is variable
                    if i >= variable_START_INDEX and i - variable_START_INDEX < variable_base_count:
                        j = i - variable_START_INDEX
                        dx = np.sign(ambs[i, 0] - qevent[2])
                        dy = np.sign(ambs[i, 1] - qevent[3])
                        grad_total[j, 0] += dx / AMB_SPEED
                        grad_total[j, 1] += dy / AMB_SPEED

                    # Schedule call completion
                    done_time = current_time + 2 * travel + qevent[4]
                    event_list.append([done_time, 2, i, 0, 0])

        # Final results computation
        if num_calls:
            avg_time = total_response_time / num_calls
            grad_avg = grad_total / num_calls
        else:
            avg_time = float("inf")
            grad_avg = np.full((variable_base_count, 2), float("nan"))

        # Pack responses and gradients
        responses = {"avg_response_time": avg_time}
        gradients = {
            "avg_response_time": {
                "variable_locs": grad_avg.flatten().tolist()  # Flatten gradient to 1D list
            }
        }
        return responses, gradients

class AmbulanceMinAvgResponse(Problem):
    @property
    def n_objectives(self) -> int:
        return 1

    @property
    def n_stochastic_constraints(self) -> int:
        return 0

    @property
    def minmax(self) -> tuple[int]:
        return (-1,)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_available(self) -> bool:
        return True
    @property
    def optimal_value(self) -> float | None:
        return None

    @property
    def optimal_solution(self) -> tuple | None:
        return None

    @property
    def model_default_factors(self) -> dict:
        return {}

    @property
    def model_decision_factors(self) -> set[str]:
        return {"variable_locs"}

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": tuple([3, 10, 3, 10]),
            },
            "budget": {
                "description": "max number of evaluations",
                "datatype": int,
                "default": 5000,
                "isDatafarmable": False,
            },
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }

    @property
    def dim(self) -> int:
        return 2 * self.model.factors["variable_base_count"]

    @property
    def lower_bounds(self) -> tuple:
        return tuple(0.0 for _ in range(self.dim))

    @property
    def upper_bounds(self) -> tuple:
        return tuple(20.0 for _ in range(self.dim))

    def __init__(self, name="AMBULANCE-1", fixed_factors=None, model_fixed_factors=None):
        # Now call parent __init__, which will read specifications (no error)
        super().__init__(name=name,
                         fixed_factors=fixed_factors,
                         model_fixed_factors=model_fixed_factors,
                         model=Ambulance)


    def vector_to_factor_dict(self, vector: tuple) -> dict:
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {"variable_locs": vector[:]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict: dict) -> tuple:
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["variable_locs"])
        return vector

    def response_dict_to_objectives(self, response_dict: dict) -> tuple:
        return (response_dict["avg_response_time"],)

    def response_dict_to_stoch_constraints(self, response_dict: dict) -> tuple:
        return ()

    def deterministic_objectives_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        return (0,), (tuple(0.0 for _ in x),)

    def deterministic_stochastic_constraints_and_gradients(self, x: tuple) -> tuple[tuple, tuple]:
        return (), ()

    def check_deterministic_constraints(self, x: tuple) -> bool:
        return all(0 <= xi <= 20 for xi in x)

    def get_random_solution(self, rand_sol_rng) -> tuple:
        return tuple(rand_sol_rng.uniform(0, 20) for _ in range(self.dim))

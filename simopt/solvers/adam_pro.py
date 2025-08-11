"""
Summary
-------
ADAM-Pro
An algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order moments.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/adam.html>`__.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simopt.base import (
    ConstraintType,
    ObjectiveType,
    Problem,
    Solution,
    Solver,
    VariableType,
)

import csv
import os
import random
import string


class ADAM_PRO(Solver):
    """
    An algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """

    @property
    def objective_type(self) -> ObjectiveType:
        return ObjectiveType.SINGLE

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BOX

    @property
    def variable_type(self) -> VariableType:
        return VariableType.CONTINUOUS

    @property
    def gradient_needed(self) -> bool:
        return False

    @property
    def specifications(self) -> dict[str, dict]:
        return {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True,
            },
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 30,
            },
            "beta_1": {
                "description": "exponential decay of the rate for the first moment estimates",
                "datatype": float,
                "default": 0.9,
            },
            "beta_2": {
                "description": "exponential decay rate for the second-moment estimates",
                "datatype": float,
                "default": 0.999,
            },
            "epsilon": {
                "description": "a small value to prevent zero-division",
                "datatype": float,
                "default": 10 ** (-8),
            },
            "sensitivity": {
                "description": "shrinking scale for variable bounds",
                "datatype": float,
                "default": 10 ** (-7),
            },
            "auto_init_steplength": {
                "description": "whether to automatically initialize step length",
                "datatype": bool,
                "default": True,
            },
            "init_alpha": {
                "description": "initial step size for ADAM",
                "datatype": float,
                "default": 0.1,  # Changing the step size matters a lot.
            },
            "init_h": {
                "description": "initial step size for finite difference",
                "datatype": float,
                "default": 0.01, 
            },
            "n_solns": {
                "description": "number of solutions to be generated when automatically initializing step size",
                "datatype": int,
                "default": 1000,
            },
            "init_alpha_fraction": {
                "description": "fraction of variable range to automatically initialize ADAM solver step size",
                "datatype": float,
                "default": 0.02,
            },
            "init_h_fraction": {
                "description": "fraction of variable range to automatically initialize finite difference step size",
                "datatype": float,
                "default": 0.002,
            },
            "use_gradient": {
                "description": "whether to use gradient if available",
                "datatype": bool,
                "default": False,
            },
            "fd_method": {
                "description": "finite difference method, 0 for Standard, 2 for Simplex, 4 for Adaptive Simplex",
                "datatype": int,
                "default": 4,
            },
            "adaptive_simplex_fraction": {
                "description": "Fraction of total simulation replications r to use in adaptive simplex gradient estimation",
                "datatype": float,
                "default": 0.1,
            },
            "adaptive_steplength": {
                "description": "whether to use adaptive step length",
                "datatype": bool,
                "default": False,
            },
            "step_adaptation_rate": {
                "description": "Multiplicative factor used to adapt step sizes alpha and h",
                "datatype": float,
                "default": 1.2,
            },
            "max_step_multiplier": {
                "description": "Maximum multiple allowed for step sizes alpha and h",
                "datatype": float,
                "default": 10.0,
            }
        }

    @property
    def check_factor_list(self) -> dict[str, Callable]:
        return {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "beta_1": self.check_beta_1,
            "beta_2": self.check_beta_2,
            "epsilon": self.check_epsilon,
            "sensitivity": self.check_sensitivity,
            "auto_init_steplength": self.check_auto_init_steplength,
            "init_alpha": self.check_init_alpha,
            "init_h": self.check_init_h,
            "n_solns": self.check_n_solns,
            "init_alpha_fraction": self.check_init_alpha_fraction,
            "init_h_fraction": self.check_init_h_fraction,
            "use_gradient": self.check_use_gradient,
            "fd_method": self.check_fd_method,
            "adaptive_simplex_fraction": self.check_adaptive_simplex_fraction,
            "adaptive_steplength": self.check_adaptive_steplength,
            "step_adaptation_rate": self.check_step_adaptation_rate,
            "max_step_multiplier": self.check_max_step_multiplier,
        }

    def __init__(
        self, name: str = "ADAM_PRO", fixed_factors: dict | None = None
    ) -> None:
        # Let the base class handle default arguments.
        super().__init__(name, fixed_factors)

    def check_r(self) -> None:
        if self.factors["r"] <= 0:
            raise ValueError(
                "The number of replications taken at each solution must be greater than 0."
            )

    def check_beta_1(self) -> None:
        if self.factors["beta_1"] <= 0 or self.factors["beta_1"] >= 1:
            raise ValueError("Beta 1 must be between 0 and 1.")

    def check_beta_2(self) -> None:
        if self.factors["beta_2"] > 0 and self.factors["beta_2"] >= 1:
            raise ValueError("Beta 2 must be less than 1.")
        
    def check_epsilon(self) -> None:
        if self.factors["epsilon"] <= 0:
            raise ValueError("Epsilon must be greater than 0.")

    def check_sensitivity(self) -> None:
        if self.factors["sensitivity"] <= 0:
            raise ValueError("Sensitivity must be greater than 0.")
        
    def check_auto_init_steplength(self) -> None:
        if not isinstance(self.factors["auto_init_steplength"], bool):
            raise ValueError("Auto init step length must be a boolean value.")

    def check_init_alpha(self) -> None:
        if self.factors["init_alpha"] <= 0:
            raise ValueError("Initial Alpha must be greater than 0.")
    
    def check_init_h(self) -> None:
        if self.factors["init_h"] <= 0:
            raise ValueError("Initial H must be greater than 0.")
    
    def check_n_solns(self) -> None:
        if self.factors["n_solns"] <= 0:
            raise ValueError("Number of solutions must be greater than 0.")
        
    def check_init_alpha_fraction(self) -> None:
        if not (0 < self.factors["init_alpha_fraction"] <= 1):
            raise ValueError(
                "Initial alpha fraction must be in the range (0, 1]."
            )

    def check_init_h_fraction(self) -> None:
        if not (0 < self.factors["init_h_fraction"] <= 1):
            raise ValueError(
                "Initial h fraction must be in the range (0, 1]."
            )
        
    def check_use_gradient(self) -> None:
        if not isinstance(self.factors["use_gradient"], bool):
            raise ValueError("Use gradient must be a boolean value.")
            
    def check_fd_method(self) -> None:
        if self.factors["fd_method"] not in [0, 2, 4]:
            raise ValueError(
                "Finite difference method must be one of: 0 (FD), 2 (Simplex), 4 (Adaptive Simplex)."
            )
            
    def check_adaptive_simplex_fraction(self) -> None:
        if not (0 < self.factors["adaptive_simplex_fraction"] <= 1):
            raise ValueError(
                "Adaptive simplex fraction must be in the range (0, 1]."
            )
            
    def check_adaptive_steplength(self) -> None:
        if not isinstance(self.factors["adaptive_steplength"], bool):
            raise ValueError("Adaptive step length must be a boolean value.")
        
    def check_step_adaptation_rate(self) -> None:
        if self.factors["step_adaptation_rate"] <= 1:
            raise ValueError("Step adaptation rate must be greater than 1.")
        
    def check_max_step_multiplier(self) -> None:
        if self.factors["max_step_multiplier"] <= 1:
            raise ValueError("Max step multiplier must be greater than 1.")

    def solve(self, problem: Problem) -> tuple[list[Solution], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

        # Default values.
        r: int = self.factors["r"]
        beta_1: float = self.factors["beta_1"]
        beta_2: float = self.factors["beta_2"]
        # If auto_init_steplength is False, initialize alpha and h.
        if not self.factors["auto_init_steplength"]:
            alpha: float = self.factors["init_alpha"]
            h: float = self.factors["init_h"]
        epsilon: float = self.factors["epsilon"]

        # Designate random number generator for random sampling
        find_next_soln_rng = self.rng_list[0]
        # Designate random number generator for subsubstream sampling
        s_rng = self.rng_list[1]
        d_rng = self.rng_list[2]
                
        if self.factors["auto_init_steplength"]:
            # Generate many dummy solutions without replication only to find a reasonable maximum radius
            dummy_solns: list[tuple[int, ...]] = []
            for _ in range(self.factors["n_solns"]):
                random_soln = problem.get_random_solution(find_next_soln_rng)
                dummy_solns.append(random_soln)

            delta_max_arr: list[float | int] = []
            for i in range(problem.dim):
                delta_max_arr += [
                    min(
                        max([sol[i] for sol in dummy_solns])
                        - min([sol[i] for sol in dummy_solns]),
                        problem.upper_bounds[0] - problem.lower_bounds[0],
                    )
                ]

            delta_max = np.mean(delta_max_arr)
            # Compute alpha: scaled delta_max
            alpha = delta_max * self.factors["init_alpha_fraction"]
            h = delta_max * self.factors["init_h_fraction"]
        
        alpha_max = alpha * self.factors["max_step_multiplier"]
        h_max = h * self.factors["max_step_multiplier"]

        # print(f"Alpha: {alpha}, H: {h}")

        # Upper bound and lower bound.
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Start with the initial solution.
        new_solution = self.create_new_solution(
            problem.factors["initial_solution"], problem
        )
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution

        # Initialize the first moment vector, the second moment vector, and the timestep.
        m = np.zeros(problem.dim)
        v = np.zeros(problem.dim)
        t = 0
        fail_count = 0

        while expended_budget < problem.factors["budget"]:
            # Update timestep.
            t = t + 1
            new_x = new_solution.x
            # Check variable bounds.
            forward = np.isclose(
                new_x, lower_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            backward = np.isclose(
                new_x, upper_bound, atol=self.factors["sensitivity"]
            ).astype(int)
            # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff.
            bounds_check = np.subtract(forward, backward)

            # If budget is nearly exhausted, print current solution in .2f format.
            if expended_budget > 0.98 * problem.factors["budget"]:
                formatted_x = ', '.join(f"{xi:.2f}" for xi in new_solution.x)
                if isinstance(new_solution.objectives_mean, (np.ndarray, list)):
                    formatted_obj = ', '.join(f"{oi:.2f}" for oi in np.atleast_1d(new_solution.objectives_mean))
                else:
                    formatted_obj = f"{new_solution.objectives_mean:.2f}"
                # print(f"Solution: [{formatted_x}], Objective: {formatted_obj}")

            
            if problem.gradient_available and self.factors["use_gradient"]:
                # Use IPA gradient if available.
                grad = (
                    -1
                    * problem.minmax[0]
                    * new_solution.objectives_gradients_mean[0]
                )
            else:
                # Use finite difference to estimate gradient if IPA gradient is not available.
                if self.factors["fd_method"] == 0:
                    grad, budget = self.finite_diff(h, new_solution, bounds_check, problem)
                elif self.factors["fd_method"] == 2:
                    grad, budget = self.simplex_gradient(h, new_solution, problem, direction_rng=d_rng)
                elif self.factors["fd_method"] == 4:
                    frac = self.factors.get("adaptive_simplex_fraction", 0.1)
                    if fail_count == 0:
                        m0 = int(max(1, r * frac))
                    elif fail_count >= 1:
                        m0 = r
                    grad, budget = self.simplex_adaptive_gradient(
                        h, new_solution, problem, m=m0,
                        shuffle_rng=s_rng, direction_rng=d_rng
                    )

                expended_budget += budget

            # Convert new_x from tuple to list.
            new_x = list(new_x)
            # Loop through all the dimensions.
            for i in range(problem.dim):

                # Update biased first moment estimate.
                m[i] = beta_1 * m[i] + (1 - beta_1) * grad[i]
                # Update biased second raw moment estimate.
                v[i] = beta_2 * v[i] + (1 - beta_2) * grad[i] ** 2
                # Compute bias-corrected first moment estimate.
                mhat = m[i] / (1 - beta_1**t)
                # Compute bias-corrected second raw moment estimate.
                vhat = v[i] / (1 - beta_2**t)
                # Update new_x and adjust it for box constraints.
                new_x[i] = min(
                    max(
                        new_x[i] - alpha * mhat / (np.sqrt(vhat) + epsilon),
                        lower_bound[i],
                    ),
                    upper_bound[i],
                )

            # Create new solution based on new x
            new_solution = self.create_new_solution(tuple(new_x), problem)
            # Use r simulated observations to estimate the objective value.
            problem.simulate(new_solution, r)
            expended_budget += r

            success = problem.minmax[0] * new_solution.objectives_mean > problem.minmax[0] * best_solution.objectives_mean
            if success:
                fail_count = 0
                # Increase h and alpha, but cap at 10x their initial values
                if self.factors['adaptive_steplength']:
                    h = min(h * self.factors["step_adaptation_rate"], h_max)
                    alpha = min(alpha * self.factors["step_adaptation_rate"], alpha_max)

                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            else:
                fail_count += 1
                if self.factors['adaptive_steplength']:
                    alpha = alpha / self.factors["step_adaptation_rate"]
                    h = h / self.factors["step_adaptation_rate"]

        # Loop through the budgets and convert any numpy int32s to Python ints.
        for i in range(len(intermediate_budgets)):
            intermediate_budgets[i] = int(intermediate_budgets[i])
            
         # === Create log file ===
        log_filename = "adampro_log.csv"
        file_exists = os.path.isfile(log_filename)

        # Generate random 4-character code: letter + digit + letter + digit
        code = (
                random.choice(string.ascii_uppercase)
                + random.choice(string.digits)
                + random.choice(string.ascii_uppercase)
                + random.choice(string.digits)
        )

        # If file doesn't exist, create and write header
        if not file_exists:
            with open(log_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Code", "Current_Solution", "Incumbent_Solution", "Replication", "Expended_Budget"]
                )

        # Main optimization loop
        while expended_budget < budget:
            k += 1
            (
                final_ob,
                delta_k,
                recommended_solns,
                intermediate_budgets,
                expended_budget,
                incumbent_x,
                kappa,
                incumbent_solution,
                visited_pts_list,
                h_k,
            ) = self.iterate(
                k,
                delta_k,
                delta_max,
                problem,
                visited_pts_list,
                incumbent_x,
                expended_budget,
                budget,
                recommended_solns,
                intermediate_budgets,
                kappa,
                incumbent_solution,
                h_k,  # type: ignore
            )

            # Log this iteration
            with open(log_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        code,
                        "(" + ", ".join(f"{float(v):.10g}" for v in incumbent_x) + ")",
                        "(" + ", ".join(f"{float(v):.10g}" for v in incumbent_x) + ")",
                        "[" + ", ".join(f"{float(v):.10g}" for v in (kappa if hasattr(kappa, "__iter__") else [kappa])) + "]",
                        f"{float(expended_budget):.10g}",
                    ]
                )

        return recommended_solns, intermediate_budgets

    # Finite difference for approximating gradients.
    def finite_diff(
        self, h: float, new_solution: Solution, bounds_check: np.ndarray, problem: Problem
    ) -> tuple[np.ndarray, int]:
        """
        Finite difference gradient estimator with simulate counter (accumulates r).
        """
        r = self.factors["r"]
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        new_x = new_solution.x
        # Store values for each dimension.
        function_diff = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)
        simulate_calls = 0  # Counter for total simulation calls (including r per simulate_up_to)

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = h
            # Backward stepsize.
            steph2 = h

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            if bounds_check[i] == 0:
                function_diff[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + function_diff[i, 2]
                x2[i] = x2[i] - function_diff[i, 2]
            elif bounds_check[i] == 1:
                function_diff[i, 2] = steph1
                x1[i] = x1[i] + function_diff[i, 2]
            else:
                function_diff[i, 2] = steph2
                x2[i] = x2[i] - function_diff[i, 2]

            x1_solution = self.create_new_solution(tuple(x1), problem)
            if bounds_check[i] != -1:
                problem.simulate_up_to([x1_solution], r)
                simulate_calls += r
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                function_diff[i, 0] = (
                    fn1[0] if isinstance(fn1, np.ndarray) else fn1
                )

            x2_solution = self.create_new_solution(tuple(x2), problem)
            if bounds_check[i] != 1:
                problem.simulate_up_to([x2_solution], r)
                simulate_calls += r
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                function_diff[i, 1] = (
                    fn2[0] if isinstance(fn2, np.ndarray) else fn2
                )

            # Calculate gradient.
            fn_divisor = (
                function_diff[i, 2][0]
                if isinstance(function_diff[i, 2], np.ndarray)
                else function_diff[i, 2]
            )
            if bounds_check[i] == 0:
                fn_diff = fn1 - fn2  # type: ignore
                fn_divisor = 2 * fn_divisor
                if isinstance(fn_diff, np.ndarray):
                    grad[i] = fn_diff[0] / fn_divisor
                else:
                    grad[i] = fn_diff / fn_divisor
            elif bounds_check[i] == 1:
                fn_diff = fn1 - fn  # type: ignore
                if isinstance(fn_diff, np.ndarray):
                    grad[i] = fn_diff[0] / fn_divisor
                else:
                    grad[i] = fn_diff / fn_divisor
            elif bounds_check[i] == -1:
                fn_diff = fn - fn2  # type: ignore
                if isinstance(fn_diff, np.ndarray):
                    grad[i] = fn_diff[0] / fn_divisor
                else:
                    grad[i] = fn_diff / fn_divisor

        return grad, simulate_calls
        
        
    @staticmethod
    def generate_regular_simplex(d: int, h: float, center: np.ndarray) -> np.ndarray:
        """
        Generate a regular simplex (d+1 points) centered at `center` in R^d, scaled by step size h.

        Returns:
            points: np.ndarray of shape (d+1, d)
        """
        # Create standard basis simplex in R^d
        # From: Constructing a regular simplex with centroid at the origin
        simplex = np.zeros((d + 1, d))

        for i in range(1, d + 1):
            simplex[i, :i] = 1
            simplex[i, i - 1] = -i
            simplex[i] /= np.sqrt(i * (i + 1))

        # Shift simplex so centroid is at the origin, then move to `center`
        centroid = np.mean(simplex, axis=0)
        simplex -= centroid  # Now centered at origin

        return center + h * simplex


    def simplex_gradient(
        self, h: float, new_solution, problem, direction_rng=None
    ) -> tuple[np.ndarray, int]:
        """
        Least Squares Gradient Estimator using (n+1) regular simplex points plus center point.

        Args:
            h (float): step size.
            new_solution: current solution with known objectives_mean.
            problem: optimization problem instance.

        Returns:
            Tuple of:
                - Estimated gradient vector (np.ndarray)
                - Total number of simulate calls (int)
        """
        r = self.factors["r"]
        dim = problem.dim
        x0 = new_solution.x
        simulate_calls = 0

        # f(x0) is known
        f0 = -1 * problem.minmax[0] * new_solution.objectives_mean
        if isinstance(f0, np.ndarray):
            f0 = f0[0]

        # Generate d+1 regular simplex points centered at x0
        points = self.generate_regular_simplex(dim, h, x0)  # shape: (d+1, d)
        # Clip points to problem bounds
        points = np.clip(points, problem.lower_bounds, problem.upper_bounds)
        # Remove any points that are exactly equal to x0 (the center)
        points = np.array([p for p in points if not np.allclose(p, x0)])

        # If repeated points are generated, add new sample points
        if len(set(map(tuple, points))) < dim + 1:
            lb, ub = problem.lower_bounds, problem.upper_bounds
            uniq = [p for p in points]

            # Local box centered at x0, clipped to [lb, ub]
            w = h / np.sqrt(dim)
            lb_box = np.maximum(lb, x0 - w)
            ub_box = np.minimum(ub, x0 + w)
            side = ub_box - lb_box

            # Fill remaining slots directly
            need = (dim + 1) - len(uniq)
            U = np.array([direction_rng.random() for _ in range(need * dim)]).reshape(need, dim)
            C = lb_box + U * side  # uniformly inside clipped box

            # Drop points equal to x0 or duplicates
            for c in C:
                if not np.allclose(c, x0, atol=1e-10, rtol=0.0) and \
                not any(np.allclose(c, q, atol=1e-10, rtol=0.0) for q in uniq):
                    uniq.append(c)

            points = np.array(uniq[: dim + 1])

        # Store finite difference pairs: x_i - x0 and f(x_i) - f(x0)
        L = []
        df = []

        for i in range(dim + 1):  # index 1 to d+1 (excluding center at index 0)
            xi = np.clip(points[i], problem.lower_bounds, problem.upper_bounds)

            sol = self.create_new_solution(xi, problem)
            problem.simulate(sol, r)
            simulate_calls += r

            fi = -1 * problem.minmax[0] * sol.objectives_mean
            if isinstance(fi, np.ndarray):
                fi = fi[0]

            L.append(xi - x0)
            df.append(fi - f0)

        L = np.array(L)   # shape: (d+1, d)
        df = np.array(df) # shape: (d+1,)

        # Solve least squares: L g ≈ df
        grad = np.linalg.lstsq(L, df, rcond=None)[0]

        return grad, simulate_calls
    

    def simplex_adaptive_gradient(
        self,
        h: float,
        new_solution,
        problem,
        m: int = 10,
        shuffle_rng=None,
        direction_rng=None
    ) -> tuple[np.ndarray, int, int]:
        """
        Adaptive simplex gradient estimator using m samples and mrg32k3a RNG.

        Args:
            h (float): step size.
            new_solution: current solution with known objectives_mean.
            problem: optimization problem instance.
            m (int): number of samples.
            rng: mrg32k3a RNG object.

        Returns:
            Tuple of:
                - Estimated gradient vector (np.ndarray)
                - Total number of simulate calls (int)
                - Last used sample index (int)
        """
        r = self.factors["r"]
        dim = problem.dim
        x0 = new_solution.x
        simulate_calls = 0
        m = min(m, r)

        # Use mrg32k3a RNG to generate a random permutation of [0, r) and take the first m indices
        i_indices = np.arange(r)
        shuffle_rng.shuffle(i_indices)
        i_indices = i_indices[:m]

        f0 = -1 * problem.minmax[0] * new_solution.objectives_mean
        if isinstance(f0, np.ndarray):
            f0 = f0[0]

        grad_samples = []
        k = 0  # iteration counter

        while k < m:
            i = i_indices[k]

            # Generate regular simplex points
            points = self.generate_regular_simplex(dim, h, x0)
            points = np.clip(points, problem.lower_bounds, problem.upper_bounds)
            # Remove any points that are exactly equal to x0 (the center)
            points = np.array([p for p in points if not np.allclose(p, x0)])

            # If repeated points are generated, add new sample points
            if len(set(map(tuple, points))) < dim + 1:
                lb, ub = problem.lower_bounds, problem.upper_bounds
                uniq = [p for p in points]

                # Local box centered at x0, clipped to [lb, ub]
                w = h / np.sqrt(dim)
                lb_box = np.maximum(lb, x0 - w)
                ub_box = np.minimum(ub, x0 + w)
                side = ub_box - lb_box

                # Fill remaining slots directly
                need = (dim + 1) - len(uniq)
                U = np.array([direction_rng.random() for _ in range(need * dim)]).reshape(need, dim)
                C = lb_box + U * side  # uniformly inside clipped box

                # Drop points equal to x0 or duplicates
                for c in C:
                    if not np.allclose(c, x0, atol=1e-10, rtol=0.0) and \
                    not any(np.allclose(c, q, atol=1e-10, rtol=0.0) for q in uniq):
                        uniq.append(c)

                points = np.array(uniq[: dim + 1])
                
            # Collect L and df
            L, df = [], []
            for d_idx in range(dim + 1):
                xi = np.clip(points[d_idx], problem.lower_bounds, problem.upper_bounds)
                sol = self.create_new_solution(xi, problem)
                problem.simulate(sol, 1, i)
                simulate_calls += 1

                fi = -1 * problem.minmax[0] * sol.objectives_mean
                if isinstance(fi, np.ndarray):
                    fi = fi[0]

                # Use the i-th sample of the original solution for f0
                f0_i = -1 * problem.minmax[0] * new_solution.objectives_samples[i]

                L.append(xi - x0)
                df.append(fi - f0_i)

            # Solve least squares system
            L = np.array(L)
            df = np.array(df)
            grad = np.linalg.lstsq(L, df, rcond=None)[0]
            grad_samples.append(grad.flatten())
            k += 1

        grad_array = np.array(grad_samples)
        grad_mean = np.mean(grad_array, axis=0)

        return grad_mean, simulate_calls

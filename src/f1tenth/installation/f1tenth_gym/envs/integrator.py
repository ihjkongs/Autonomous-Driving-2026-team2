from abc import abstractmethod
from enum import Enum
from math import ceil


class IntegratorType(Enum):
    RK4 = 1
    Euler = 2

    @staticmethod
    def from_string(integrator: str):
        if integrator == "rk4":
            return RK4Integrator()
        elif integrator == "euler":
            return EulerIntegrator()
        else:
            raise ValueError(f"Unknown integrator type {integrator}")


class Integrator:
    def __init__(self) -> None:
        self._integrator_type = None

    @abstractmethod
    def integrate(self, f, x, u, dt, params):
        raise NotImplementedError("integrate method not implemented")

    @property
    def type(self) -> str:
        return self._integrator_type


class RK4Integrator(Integrator):
    def __init__(self) -> None:
        super().__init__()
        self._integrator_type = "rk4"

    def integrate(self, f, x, u, dt, params):
        # The ST lateral dynamics are stiff at low speed. With the course's
        # 25 ms control period, a single RK4 step amplifies a stable mode at
        # 1 m/s (lambda ~= -123/s, RK4 amplification ~= 1.54). Keep the
        # external control/scan period, but integrate physics in <=5 ms steps.
        steps = max(1, ceil(dt / 0.005))
        step_dt = dt / steps
        for _ in range(steps):
            x = self._integrate_step(f, x, u, step_dt, params)
        return x

    def _integrate_step(self, f, x, u, dt, params):
        k1 = f(
            x,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        k2_state = x + dt * (k1 / 2)

        k2 = f(
            k2_state,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        k3_state = x + dt * (k2 / 2)

        k3 = f(
            k3_state,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        k4_state = x + dt * k3

        k4 = f(
            k4_state,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        # dynamics integration
        x = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


class EulerIntegrator(Integrator):
    def __init__(self) -> None:
        super().__init__()
        self._integrator_type = "euler"

    def integrate(self, f, x, u, dt, params):
        dstate = f(
            x,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )
        x = x + dt * dstate
        return x

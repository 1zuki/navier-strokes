"""Readable 2D Stable Fluids style solver.

Arrays use one layer of ghost cells around an N x N interior. The first array
axis is x, and the second array axis is y. Public splat coordinates are in
solver-cell space, where 1..N maps to the visible domain.
"""

from __future__ import annotations

import numpy as np


class Fluid:
    """Small educational incompressible fluid solver."""

    def __init__(
        self,
        size: int = 96,
        dt: float = 0.1,
        diffusion: float = 0.0,
        viscosity: float = 0.0001,
        iterations: int = 16,
        density_decay: float = 0.997,
        max_density: float = 120.0,
        max_velocity: float = 8.0,
    ) -> None:
        if size < 8:
            raise ValueError("size must be at least 8")

        self.size = int(size)
        self.dt = float(dt)
        self.diffusion = float(diffusion)
        self.viscosity = float(viscosity)
        self.iterations = int(iterations)
        self.density_decay = float(density_decay)
        self.max_density = float(max_density)
        self.max_velocity = float(max_velocity)

        shape = (self.size + 2, self.size + 2)
        self.density = np.zeros(shape, dtype=np.float32)
        self.u = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)

        self.density_source = np.zeros(shape, dtype=np.float32)
        self.u_source = np.zeros(shape, dtype=np.float32)
        self.v_source = np.zeros(shape, dtype=np.float32)

        self._density0 = np.zeros(shape, dtype=np.float32)
        self._u0 = np.zeros(shape, dtype=np.float32)
        self._v0 = np.zeros(shape, dtype=np.float32)

    def clear(self) -> None:
        """Reset density, velocity, sources, and scratch arrays."""
        for field in (
            self.density,
            self.u,
            self.v,
            self.density_source,
            self.u_source,
            self.v_source,
            self._density0,
            self._u0,
            self._v0,
        ):
            field.fill(0.0)

    def add_density(
        self, x: float, y: float, amount: float, radius: float = 1.5
    ) -> None:
        """Queue a dye source around a solver-space point."""
        self._splat(self.density_source, x, y, amount, radius)

    def add_velocity(
        self,
        x: float,
        y: float,
        amount_u: float,
        amount_v: float,
        radius: float = 1.5,
    ) -> None:
        """Queue a velocity source around a solver-space point."""
        self._splat(self.u_source, x, y, amount_u, radius)
        self._splat(self.v_source, x, y, amount_v, radius)

    def apply_sources(self) -> None:
        """Apply queued source arrays, then clear them."""
        self.sanitize()
        self.density += self.density_source
        self.u += self.u_source
        self.v += self.v_source

        self.density_source.fill(0.0)
        self.u_source.fill(0.0)
        self.v_source.fill(0.0)

        self._set_bnd(0, self.density)
        self._set_bnd(1, self.u)
        self._set_bnd(2, self.v)
        self.sanitize()

    def step(self) -> None:
        """Advance the simulation one Stable Fluids style time step."""
        self.apply_sources()

        self._u0[:] = self.u
        self._diffuse(1, self.u, self._u0, self.viscosity)
        self._v0[:] = self.v
        self._diffuse(2, self.v, self._v0, self.viscosity)
        self._project(self.u, self.v, self._u0, self._v0)

        self._u0[:] = self.u
        self._v0[:] = self.v
        self._advect(1, self.u, self._u0, self._u0, self._v0)
        self._advect(2, self.v, self._v0, self._u0, self._v0)
        self._project(self.u, self.v, self._u0, self._v0)

        self._density0[:] = self.density
        self._diffuse(0, self.density, self._density0, self.diffusion)
        self._density0[:] = self.density
        self._advect(0, self.density, self._density0, self.u, self.v)

        if self.density_decay < 1.0:
            self.density[1:-1, 1:-1] *= self.density_decay
        self.sanitize()

    def sanitize(self) -> None:
        """Clamp bad interactive inputs before they can poison the solver."""
        for field in (self.u, self.v, self.u_source, self.v_source):
            np.nan_to_num(
                field,
                copy=False,
                nan=0.0,
                posinf=self.max_velocity,
                neginf=-self.max_velocity,
            )
            np.clip(field, -self.max_velocity, self.max_velocity, out=field)

        for field in (self.density, self.density_source):
            np.nan_to_num(
                field,
                copy=False,
                nan=0.0,
                posinf=self.max_density,
                neginf=0.0,
            )
            np.clip(field, 0.0, self.max_density, out=field)

    def _splat(
        self, field: np.ndarray, x: float, y: float, amount: float, radius: float
    ) -> None:
        x = float(np.clip(x, 1.0, self.size))
        y = float(np.clip(y, 1.0, self.size))
        radius = max(float(radius), 0.5)

        i0 = max(1, int(np.floor(x - radius)))
        i1 = min(self.size, int(np.ceil(x + radius)))
        j0 = max(1, int(np.floor(y - radius)))
        j1 = min(self.size, int(np.ceil(y + radius)))

        xs = np.arange(i0, i1 + 1, dtype=np.float32)[:, None]
        ys = np.arange(j0, j1 + 1, dtype=np.float32)[None, :]
        dist2 = (xs - x) ** 2 + (ys - y) ** 2
        weights = np.maximum(0.0, 1.0 - dist2 / (radius * radius))

        if not np.any(weights):
            field[int(round(x)), int(round(y))] += amount
            return
        field[i0 : i1 + 1, j0 : j1 + 1] += amount * weights

    def _diffuse(
        self, boundary: int, current: np.ndarray, previous: np.ndarray, rate: float
    ) -> None:
        a = self.dt * max(rate, 0.0) * self.size * self.size
        self._lin_solve(boundary, current, previous, a, 1.0 + 4.0 * a)

    def _lin_solve(
        self,
        boundary: int,
        current: np.ndarray,
        previous: np.ndarray,
        a: float,
        c: float,
    ) -> None:
        inv_c = 1.0 / c
        for _ in range(self.iterations):
            current[1:-1, 1:-1] = (
                previous[1:-1, 1:-1]
                + a
                * (
                    current[:-2, 1:-1]
                    + current[2:, 1:-1]
                    + current[1:-1, :-2]
                    + current[1:-1, 2:]
                )
            ) * inv_c
            self._set_bnd(boundary, current)

    def _project(
        self,
        u: np.ndarray,
        v: np.ndarray,
        pressure: np.ndarray,
        divergence: np.ndarray,
    ) -> None:
        h = 1.0 / self.size
        divergence[1:-1, 1:-1] = -0.5 * h * (
            u[2:, 1:-1]
            - u[:-2, 1:-1]
            + v[1:-1, 2:]
            - v[1:-1, :-2]
        )
        pressure.fill(0.0)

        self._set_bnd(0, divergence)
        self._set_bnd(0, pressure)
        self._lin_solve(0, pressure, divergence, 1.0, 4.0)

        u[1:-1, 1:-1] -= 0.5 * (
            pressure[2:, 1:-1] - pressure[:-2, 1:-1]
        ) / h
        v[1:-1, 1:-1] -= 0.5 * (
            pressure[1:-1, 2:] - pressure[1:-1, :-2]
        ) / h
        self._set_bnd(1, u)
        self._set_bnd(2, v)

    def _advect(
        self,
        boundary: int,
        current: np.ndarray,
        previous: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> None:
        dt0 = self.dt * self.size
        i = np.arange(1, self.size + 1, dtype=np.float32)[:, None]
        j = np.arange(1, self.size + 1, dtype=np.float32)[None, :]

        x = np.clip(i - dt0 * u[1:-1, 1:-1], 0.5, self.size + 0.5)
        y = np.clip(j - dt0 * v[1:-1, 1:-1], 0.5, self.size + 0.5)

        i0 = np.floor(x).astype(np.int32)
        j0 = np.floor(y).astype(np.int32)
        i1 = i0 + 1
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        current[1:-1, 1:-1] = (
            s0 * (t0 * previous[i0, j0] + t1 * previous[i0, j1])
            + s1 * (t0 * previous[i1, j0] + t1 * previous[i1, j1])
        )
        self._set_bnd(boundary, current)

    def _set_bnd(self, boundary: int, field: np.ndarray) -> None:
        n = self.size

        field[0, 1 : n + 1] = -field[1, 1 : n + 1] if boundary == 1 else field[1, 1 : n + 1]
        field[n + 1, 1 : n + 1] = (
            -field[n, 1 : n + 1] if boundary == 1 else field[n, 1 : n + 1]
        )
        field[1 : n + 1, 0] = -field[1 : n + 1, 1] if boundary == 2 else field[1 : n + 1, 1]
        field[1 : n + 1, n + 1] = (
            -field[1 : n + 1, n] if boundary == 2 else field[1 : n + 1, n]
        )

        field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])
        field[0, n + 1] = 0.5 * (field[1, n + 1] + field[0, n])
        field[n + 1, 0] = 0.5 * (field[n, 0] + field[n + 1, 1])
        field[n + 1, n + 1] = 0.5 * (field[n, n + 1] + field[n + 1, n])

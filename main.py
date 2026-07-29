from __future__ import annotations

import sys

import numpy as np
import pygame

from fluid import Fluid


GRID_SIZE = 96
WINDOW_SIZE = 768
FPS = 60


def solver_position(mouse_pos: tuple[int, int], fluid: Fluid) -> tuple[float, float]:
    x, y = mouse_pos
    gx = 1.0 + (x / max(WINDOW_SIZE - 1, 1)) * (fluid.size - 1)
    gy = 1.0 + (y / max(WINDOW_SIZE - 1, 1)) * (fluid.size - 1)
    return gx, gy


def make_density_surface(fluid: Fluid) -> pygame.Surface:
    density = np.clip(fluid.density[1:-1, 1:-1] / fluid.max_density, 0.0, 1.0)
    glow = np.sqrt(density)

    rgb = np.empty((fluid.size, fluid.size, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * np.clip(1.65 * density, 0.0, 1.0)).astype(np.uint8)
    rgb[..., 1] = (255 * np.clip(0.95 * glow, 0.0, 1.0)).astype(np.uint8)
    rgb[..., 2] = (
        255 * np.clip(2.1 * density * (1.0 - 0.25 * density), 0.0, 1.0)
    ).astype(np.uint8)
    return pygame.surfarray.make_surface(rgb)


def draw_velocity_overlay(screen: pygame.Surface, fluid: Fluid) -> None:
    stride = 8
    cell = WINDOW_SIZE / fluid.size
    scale = cell * 2.2

    for i in range(stride // 2, fluid.size, stride):
        for j in range(stride // 2, fluid.size, stride):
            u = float(fluid.u[i + 1, j + 1])
            v = float(fluid.v[i + 1, j + 1])
            speed = min((u * u + v * v) ** 0.5, fluid.max_velocity)
            if speed < 0.03:
                continue

            sx = (i + 0.5) * cell
            sy = (j + 0.5) * cell
            ex = sx + u * scale
            ey = sy + v * scale
            pygame.draw.line(screen, (225, 240, 255), (sx, sy), (ex, ey), 1)
            pygame.draw.circle(screen, (225, 240, 255), (int(ex), int(ey)), 2)


def draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    fluid: Fluid,
    clock: pygame.time.Clock,
    paused: bool,
    show_velocity: bool,
    force: float,
    brush_radius: float,
) -> None:
    lines = [
        (
            f"FPS {clock.get_fps():5.1f} | grid {fluid.size}x{fluid.size} | "
            f"{'paused' if paused else 'running'}"
        ),
        (
            f"viscosity {fluid.viscosity:.2e} | diffusion {fluid.diffusion:.2e} | "
            f"force {force:.2f} | brush {brush_radius:.1f} | vectors {'on' if show_velocity else 'off'}"
        ),
        "Left mouse dye+push | Right mouse dye | Wheel brush | Space pause | S step",
        "C clear | V vectors | [/] viscosity | -/= force | Esc quit",
    ]

    height = 10 + len(lines) * 20
    panel = pygame.Surface((WINDOW_SIZE, height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 155))
    screen.blit(panel, (0, 0))

    for row, line in enumerate(lines):
        text = font.render(line, True, (235, 240, 245))
        screen.blit(text, (10, 8 + row * 20))


def handle_keys(event: pygame.event.Event, state: dict[str, object], fluid: Fluid) -> None:
    if event.key == pygame.K_ESCAPE:
        state["running"] = False
    elif event.key == pygame.K_SPACE:
        state["paused"] = not bool(state["paused"])
    elif event.key == pygame.K_c:
        fluid.clear()
    elif event.key == pygame.K_v:
        state["show_velocity"] = not bool(state["show_velocity"])
    elif event.key == pygame.K_s and state["paused"]:
        fluid.step()
    elif event.key == pygame.K_LEFTBRACKET:
        fluid.viscosity = max(0.0, fluid.viscosity / 1.5)
    elif event.key == pygame.K_RIGHTBRACKET:
        fluid.viscosity = min(0.02, max(1e-7, fluid.viscosity * 1.5))
    elif event.key == pygame.K_MINUS:
        state["force"] = max(0.05, float(state["force"]) / 1.25)
    elif event.key == pygame.K_EQUALS:
        state["force"] = min(8.0, float(state["force"]) * 1.25)


def inject_from_mouse(
    fluid: Fluid,
    mouse_pos: tuple[int, int],
    previous_pos: tuple[int, int] | None,
    buttons: tuple[bool, bool, bool],
    force: float,
    brush_radius: float,
) -> bool:
    left, _, right = buttons
    if not (left or right):
        return False

    gx, gy = solver_position(mouse_pos, fluid)
    density_amount = 22.0 if left else 14.0
    fluid.add_density(gx, gy, density_amount, brush_radius)

    if left and previous_pos is not None:
        dx = (mouse_pos[0] - previous_pos[0]) / max(WINDOW_SIZE / fluid.size, 1.0)
        dy = (mouse_pos[1] - previous_pos[1]) / max(WINDOW_SIZE / fluid.size, 1.0)
        fluid.add_velocity(gx, gy, dx * force, dy * force, brush_radius)

    return True


def main() -> int:
    pygame.init()
    pygame.display.set_caption("Navier Strokes - Stable Fluids Demo")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    fluid = Fluid(size=GRID_SIZE)
    state: dict[str, object] = {
        "running": True,
        "paused": False,
        "show_velocity": False,
        "force": 1.2,
    }
    brush_radius = 3.0
    previous_mouse_pos: tuple[int, int] | None = None

    while state["running"]:
        interacted = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["running"] = False
            elif event.type == pygame.KEYDOWN:
                handle_keys(event, state, fluid)
            elif event.type == pygame.MOUSEWHEEL:
                brush_radius = float(np.clip(brush_radius + event.y * 0.5, 1.0, 12.0))

        mouse_pos = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed(3)
        if 0 <= mouse_pos[0] < WINDOW_SIZE and 0 <= mouse_pos[1] < WINDOW_SIZE:
            interacted = inject_from_mouse(
                fluid,
                mouse_pos,
                previous_mouse_pos,
                buttons,
                float(state["force"]),
                brush_radius,
            )
        previous_mouse_pos = mouse_pos if any(buttons) else None

        if state["paused"]:
            if interacted:
                fluid.apply_sources()
        else:
            fluid.step()

        density_surface = make_density_surface(fluid)
        screen.blit(pygame.transform.smoothscale(density_surface, screen.get_size()), (0, 0))

        if state["show_velocity"]:
            draw_velocity_overlay(screen, fluid)

        draw_overlay(
            screen,
            font,
            fluid,
            clock,
            bool(state["paused"]),
            bool(state["show_velocity"]),
            float(state["force"]),
            brush_radius,
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())

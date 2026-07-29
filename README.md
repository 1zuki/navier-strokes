# Navier Strokes

My implementation of 2D fluid demo using a Jos Stam "Stable Fluids" style solver.

The solver lives in `fluid.py` and only depends on NumPy. The Pygame interface lives in `main.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Controls

- Left mouse: add dye and push the fluid
- Right mouse: add dye without force
- Mouse wheel: adjust brush size
- Space: pause or resume
- S: advance one step while paused
- C: clear the simulation
- V: toggle velocity overlay
- `[` / `]`: decrease / increase viscosity
- `-` / `=`: decrease / increase mouse force
- Esc: quit

## Notes

The simulation order is:

1. Add queued density and velocity sources.
2. Diffuse velocity.
3. Project velocity to reduce divergence.
4. Advect velocity.
5. Project again.
6. Diffuse and advect density.

The implementation uses simple boundary conditions, density clamping, velocity
clamping, and finite-value cleanup so the interactive demo can recover from
large mouse inputs. Numerical diffusion is expected, especially at low grid
sizes or high viscosity.

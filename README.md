# valgrad

First-principles autograd in pure C, inspired by micrograd.

This is very WIP. I am building it mostly to understand reverse-mode autodiff, memory ownership, computation graphs, and eventually tensors without hiding behind a big framework.

`valgrad` means values + gradients. The name does not fully explain itself yet, but the idea is simple: start with scalar `Value` nodes, make gradients work, then try to grow that into a tiny tensor engine.

## Goals

- Build scalar autograd in C.
- Train tiny examples like curve fitting and XOR.
- Add finite-difference gradient checks.
- Move from scalars to tensors without hand-waving the hard parts.
- Eventually make it importable from Python.

## Non-goals

- Not trying to beat PyTorch.
- Not starting with CUDA, fancy optimizers, or huge abstractions.
- Not pretending this is useful before it actually is.

For the longer project plan, see [PLAN.md](PLAN.md).

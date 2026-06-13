# Valgrad

This document is the north star for Valgrad.

It is written for me first. It should help me decide what to build, what not to build, and how to keep the project useful without losing the learning challenge. It is also structured so it can double as an `AGENTS.md`-style guide for future coding agents, but that is secondary.

## One-Sentence Vision

Valgrad is a tiny pure-C reverse-mode autodiff engine that starts with scalar `Value` objects, grows into tensors, and eventually becomes importable from Python for small learning experiments and optimization problems.

## Why This Exists

The point is not to build a smaller PyTorch.

The point is to understand the machinery underneath autodiff and neural networks by implementing it in a language where memory, ownership, shape metadata, and graph lifetime cannot be hand-waved away.

Valgrad should be useful as:

- A learning project for reverse-mode autodiff.
- A tiny embeddable C library for differentiating small scalar and tensor computations.
- A playground for fitting simple models and formulas.
- A Python-importable toy engine for experiments where inspectability matters more than speed.
- A project that forces me to learn systems-level design without hiding behind a large framework.

## Name

The name is `valgrad`.

Meaning:

- `val`: starts from scalar `Value` objects.
- `grad`: reverse-mode gradient accumulation.

It should read well in both C and Python:

```c
#include <valgrad.h>
```

```python
import valgrad as vg
```

Recommended C symbol prefix:

```c
vg_
```

Examples:

```c
vg_value_t *x = vg_value(ctx, 2.0);
vg_value_t *y = vg_tanh(ctx, vg_mul(ctx, x, w));
vg_backward(y);
```

```python
import valgrad as vg

x = vg.value(2.0)
y = (x * w).tanh()
y.backward()
```

## Core Bet

The core should be written in C.

Python should become the friendly experiment layer later. C should remain the source of truth for autodiff, tensor storage, backward passes, optimizers, and graph lifetime.

Why C:

- It makes memory ownership explicit.
- It keeps the ABI simple.
- It is easy to embed.
- It makes the hard parts visible.
- It matches the "build the machinery myself" goal.

Why not Go for the core:

- The runtime and GC are not the learning target here.
- Python extension interop is less natural.
- Tensor memory/layout work fits C better.

Why not Rust for v0:

- Rust is a good future option, but learning Rust and building autodiff at the same time is two projects.

Why not C++ for v0:

- C++ would make many things nicer, but part of the exercise is designing the core without RAII, templates, or operator overloading.

## Project Identity

Valgrad should feel like:

- Small.
- Inspectable.
- Honest.
- Educational.
- Usable from C first.
- Convenient from Python later.

Valgrad should not feel like:

- A PyTorch clone.
- A framework.
- A compiler project.
- A CUDA project.
- A benchmark game.
- A pile of abstractions with no examples.

## Non-Goals

Do not chase these early:

- GPU support.
- CUDA kernels.
- Automatic mixed precision.
- Distributed training.
- JIT compilation.
- Dynamic graph optimizations.
- Full NumPy compatibility.
- Full broadcasting semantics from day one.
- A huge neural network module system.
- Performance before correctness.
- Fancy optimizers before SGD works.
- Python polish before the C core is real.

## Definition of Useful

Valgrad is useful if it can do these things reliably:

- Fit a line or polynomial from C.
- Train a scalar MLP on XOR from C.
- Export a computation graph to Graphviz DOT.
- Run finite-difference gradient checks for every operation.
- Train a tiny tensor MLP on a toy dataset.
- Be imported from Python and used for small experiments.
- Let me inspect every important piece of the engine.

The first public-useful version does not need to be fast. It needs to be correct, small, and clear.

## The Challenge

The scalar-to-tensor jump is the real challenge.

Scalar autograd teaches:

- Graph construction.
- Topological ordering.
- Local backward rules.
- Gradient accumulation.
- Manual training loops.

Tensor autograd adds:

- Shape checking.
- Storage layout.
- Strides.
- Views versus copies.
- Gradient tensors.
- Matmul backward.
- Reductions.
- Broadcasting gradients.
- Memory pressure from graph construction.

That transition is where the project becomes more than a micrograd port.

## Milestones

### Milestone 0: Project Skeleton

Create a plain C project that is easy to build and test.

Suggested layout:

```text
valgrad/
  include/
    valgrad.h
  src/
    context.c
    value.c
    graph.c
    optim.c
    tensor.c
  tests/
    test_value.c
    test_gradcheck.c
    test_tensor.c
  examples/
    scalar_curve_fit.c
    scalar_xor.c
    tensor_xor.c
  python/
    valgrad/
      __init__.py
    bindings/
  docs/
    notes.md
  VALGRAD.md
```

Start with the simplest build that keeps momentum. A `Makefile` is fine at the beginning. CMake can come later if packaging and Python bindings need it.

Minimum setup:

- Compile with warnings enabled.
- Have one command that runs all C tests.
- Have one command that runs all examples.
- Use sanitizers while developing if available.

Recommended compiler flags during development:

```text
-std=c11 -Wall -Wextra -Wpedantic -Werror
```

Useful debug flags:

```text
-fsanitize=address,undefined -g
```

### Milestone 1: Scalar Autograd Core

Build the scalar engine first.

Target concept:

```c
typedef struct vg_context vg_context_t;
typedef struct vg_value vg_value_t;

vg_context_t *vg_context_create(void);
void vg_context_free(vg_context_t *ctx);

vg_value_t *vg_value(vg_context_t *ctx, double data);
vg_value_t *vg_add(vg_context_t *ctx, vg_value_t *a, vg_value_t *b);
vg_value_t *vg_mul(vg_context_t *ctx, vg_value_t *a, vg_value_t *b);
vg_value_t *vg_tanh(vg_context_t *ctx, vg_value_t *x);
vg_value_t *vg_relu(vg_context_t *ctx, vg_value_t *x);

double vg_data(const vg_value_t *v);
double vg_grad(const vg_value_t *v);
void vg_set_label(vg_value_t *v, const char *label);

void vg_zero_grad(vg_value_t **params, size_t n);
void vg_backward(vg_value_t *root);
void vg_sgd_step(vg_value_t **params, size_t n, double lr);
```

Important design decision:

Use an explicit context or arena from the start.

Do not let every value become an independent unmanaged allocation. The graph is temporary, and training loops create many graphs. A context gives the project a place to own nodes and clean them up.

For v0, it is acceptable if:

- A context owns all values.
- `vg_context_free(ctx)` frees everything.
- A training example creates a fresh graph each iteration.

Later, separate persistent parameters from temporary graph nodes.

Scalar operations to support first:

- `add`
- `sub`
- `mul`
- `div`
- `neg`
- `pow` for scalar exponent if useful
- `tanh`
- `relu`
- `exp`
- `log`

Do not add all operations at once. Each operation must come with tests.

Scalar demos:

- Fit `y = ax + b`.
- Fit a quadratic.
- Train XOR with a scalar MLP.
- Export the graph to DOT.

Done means:

- `backward()` works from any scalar root.
- Gradients accumulate correctly through shared nodes.
- Re-running backward behavior is clearly defined.
- Finite-difference checks pass for every scalar op.
- At least one scalar training demo actually reduces loss.

### Milestone 2: Tiny Neural Network Helpers

Only after scalar autodiff works, build small helper abstractions:

- Neuron.
- Layer.
- MLP.
- Parameter collection.
- SGD step.

These helpers should be examples or thin utilities, not the center of the project.

The center is autodiff.

Target C usage:

```c
vg_mlp_t *model = vg_mlp_create(ctx, layers, nlayers);

for (size_t step = 0; step < 1000; step++) {
    vg_value_t *loss = xor_loss(ctx, model, xs, ys);
    vg_backward(loss);
    vg_sgd_step(params, nparams, 0.05);
    vg_zero_grad(params, nparams);
    vg_context_reset_graph(ctx);
}
```

This example will probably force a better graph lifetime model.

That is good. Let the example reveal the design pressure.

### Milestone 3: Tensor Storage Without Autograd

Do not jump directly from scalar autograd to tensor autograd.

First build tensors as plain data structures with forward-only operations.

Initial tensor type:

```c
typedef struct vg_tensor vg_tensor_t;
```

Initial constraints:

- `double` only.
- CPU only.
- Contiguous storage only.
- No broadcasting at first.
- No views at first.
- Explicit shape checking.

Basic API sketch:

```c
vg_tensor_t *vg_tensor_new(vg_context_t *ctx, const size_t *shape, size_t ndim);
vg_tensor_t *vg_tensor_from_data(vg_context_t *ctx, const double *data, const size_t *shape, size_t ndim);

vg_tensor_t *vg_tensor_add(vg_context_t *ctx, vg_tensor_t *a, vg_tensor_t *b);
vg_tensor_t *vg_tensor_mul(vg_context_t *ctx, vg_tensor_t *a, vg_tensor_t *b);
vg_tensor_t *vg_tensor_matmul(vg_context_t *ctx, vg_tensor_t *a, vg_tensor_t *b);
vg_tensor_t *vg_tensor_tanh(vg_context_t *ctx, vg_tensor_t *x);
vg_tensor_t *vg_tensor_relu(vg_context_t *ctx, vg_tensor_t *x);
vg_tensor_t *vg_tensor_sum(vg_context_t *ctx, vg_tensor_t *x);
vg_tensor_t *vg_tensor_mean(vg_context_t *ctx, vg_tensor_t *x);
```

Forward-only tensor work should teach:

- Shape representation.
- Index calculation.
- Row-major layout.
- Matrix multiplication.
- Reductions.
- Error handling.

Done means:

- Tensor creation is reliable.
- Shape errors are caught.
- Matmul works.
- Reductions work.
- Tests cover shape and numeric behavior.

### Milestone 4: Tensor Autograd

Once tensor forward ops work, add gradients.

Each tensor needs:

- Data storage.
- Optional gradient storage.
- Requires-grad flag.
- Operation metadata.
- Parent pointers.
- Backward function.

Initial tensor autodiff ops:

- `add`
- `mul`
- `matmul`
- `tanh`
- `relu`
- `sum`
- `mean`
- `mse`

Backward rules to understand deeply:

```text
z = a + b
da += dz
db += dz

z = a * b
da += dz * b
db += dz * a

z = matmul(a, b)
da += dz @ b.T
db += a.T @ dz

z = sum(x)
dx += ones_like(x) * dz

z = mean(x)
dx += ones_like(x) * dz / numel(x)
```

Broadcasting should come after these work without broadcasting.

Broadcasting is not just forward shape convenience. The backward pass has to reduce gradients along broadcasted dimensions. That is a separate lesson.

Done means:

- Tensor gradient checks pass.
- Matmul backward is tested.
- Reductions backward are tested.
- A tiny tensor MLP can reduce loss.

### Milestone 5: Python Import

Make Python a frontend, not the source of truth.

Initial Python goals:

```python
import valgrad as vg

x = vg.tensor([[1.0, 2.0]])
w = vg.randn((2, 1), requires_grad=True)
y = x @ w
loss = y.mean()
loss.backward()
```

Possible binding paths:

1. `ctypes` or `cffi` for fast early experiments.
2. CPython extension module when the API stabilizes.
3. NumPy interop after basic Python import works.

Do not start with packaging polish.

First goal:

- `import valgrad as vg`
- Create a scalar or tensor.
- Run a forward op.
- Run backward.
- Read `.grad`.

Then improve ergonomics.

### Milestone 6: Real Demos

Valgrad should prove itself with examples.

Good demos:

- Scalar line fitting.
- Scalar polynomial fitting.
- XOR scalar MLP.
- Tensor XOR MLP.
- Sine-wave regression.
- Tiny logistic regression on a CSV.
- Tiny MNIST MLP as a stretch goal.

The demos should be small enough to understand in one sitting.

Each demo should answer:

- What is being optimized?
- What are the parameters?
- What is the loss?
- Does the loss go down?
- Are gradients checked somewhere nearby?

## Design Principles

### Correctness Before Speed

Every operation needs:

- Forward test.
- Backward test.
- Finite-difference gradient check.

If an op does not have a gradient check, assume it is wrong.

### Small Public API

Expose fewer things.

Internals can change. Public APIs become promises.

Prefer:

```c
vg_tensor_matmul(ctx, a, b)
```

over clever abstractions too early.

### Explicit Ownership

Every object should have an obvious owner.

Questions to answer for every type:

- Who allocates it?
- Who frees it?
- Can it outlive the graph?
- Can it be reused across training steps?
- Does it own its data or borrow it?

### No Hidden Global Graph

Avoid a global graph.

Use an explicit context:

```c
vg_context_t *ctx = vg_context_create();
```

That context can own graph nodes, temporary buffers, and error state.

### Parameters Are Special

Parameters need to persist across iterations.

Computation graph nodes are often temporary.

This tension should be represented in the design. If everything lives in one context forever, training loops will leak memory conceptually even if the OS frees it at program exit.

Possible later design:

- Model parameters live in a long-lived context.
- Forward graph nodes live in a short-lived tape/context.
- Gradients accumulate into parameters.
- Temporary graph is reset after each optimization step.

### Error Handling Should Be Boring

C has no exceptions.

Pick one simple strategy and use it consistently:

- Return `NULL` on allocation or shape failure.
- Store an error code/message in `vg_context_t`.
- Provide `vg_context_error(ctx)`.

Do not mix five styles.

### Do Not Hide the Math

The project should make backward rules visible.

Avoid abstractions that make it impossible to inspect:

- Which op created this node?
- Who are its parents?
- What backward function will run?
- Where did this gradient come from?

## Testing Strategy

Tests are not optional. This project is math-heavy, and math bugs can look plausible.

Required test categories:

- Unit tests for each scalar op.
- Unit tests for each tensor op.
- Shape tests.
- Gradient accumulation tests.
- Shared-subgraph tests.
- Finite-difference gradient checks.
- Training smoke tests.
- Memory/sanitizer runs when possible.

Finite-difference check:

```text
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Use central difference where possible.

Suggested tolerance:

```text
1e-6 for scalar double checks, adjusted when needed
```

Important cases:

- A value used twice.
- A parameter affecting loss through multiple paths.
- Zero gradients.
- Negative values through ReLU.
- Matmul shapes that are not square.
- Reductions over nontrivial shapes.

## The First Magic Moment

The first magic moment is not tensor support.

The first magic moment is this:

```c
vg_value_t *loss = xor_loss(ctx, model, xs, ys);
vg_backward(loss);
vg_sgd_step(params, nparams, 0.05);
```

and the loss actually goes down.

Do not skip this moment.

## The Second Magic Moment

The second magic moment is:

```python
import valgrad as vg

x = vg.tensor([[0.0, 1.0], [1.0, 0.0]])
y = model(x)
loss = vg.mse(y, target)
loss.backward()
```

and Python is calling into the C engine.

That proves the library is both educational and usable.

## Dangerous Rabbit Holes

Avoid these until the core works:

- Building a complete Pythonic API before C examples work.
- Designing a full module system.
- Adding broadcasting before basic tensor backward works.
- Supporting many dtypes.
- Supporting GPU.
- Rewriting in Rust midstream.
- Switching to C++ because C feels painful.
- Optimizing matmul before verifying matmul gradients.
- Adding Adam before SGD examples work.
- Building a package website.

The project wins by finishing small, correct layers.

## Personal Rules

These rules are for me.

1. Do not move to tensors until scalar autograd has gradient checks.
2. Do not move to Python until C examples can train something.
3. Do not add an op without a backward test.
4. Do not add broadcasting until non-broadcasted tensor autograd works.
5. Do not add performance optimizations unless there is a benchmark and a correctness test.
6. Do not let API design become a substitute for implementing the hard part.
7. Keep examples tiny and runnable.
8. Prefer deleting a confusing abstraction over explaining it.
9. Keep the README simple until the library earns more words.
10. Every milestone should produce a working demo.

## Possible Version Map

### v0.1: Scalar Core

- Scalar `Value`.
- Reverse-mode backward.
- Basic ops.
- SGD.
- DOT export.
- Scalar gradient checks.
- Curve fitting demo.

### v0.2: Scalar MLP

- Neuron/layer/MLP helpers.
- XOR training.
- Cleaner parameter handling.
- Better graph lifetime story.

### v0.3: Tensor Forward

- Tensor struct.
- Shape metadata.
- Contiguous storage.
- Forward tensor ops.
- Matmul.
- Reductions.

### v0.4: Tensor Autograd

- Tensor requires-grad.
- Tensor backward.
- Matmul backward.
- MSE loss.
- Tensor gradient checks.
- Tensor XOR demo.

### v0.5: Python Preview

- Python import.
- Minimal scalar/tensor wrapper.
- `.backward()`.
- `.grad`.
- Tiny Python example.

### v0.6: Better Tensor Semantics

- Broadcasting.
- Views or reshape.
- Safer memory model.
- More examples.

### v1.0: Small But Honest

- Stable C API.
- Stable Python preview API.
- Good tests.
- Good examples.
- Clear docs.
- No claims beyond what the library actually does.

## Glossary

`Value`
: A scalar node in the computation graph.

`Tensor`
: A multi-dimensional array with shape metadata and optional gradient.

`Tape`
: The recorded computation graph used for reverse-mode autodiff.

`Backward pass`
: The process of walking the graph in reverse topological order and accumulating gradients.

`Gradient accumulation`
: Adding gradient contributions into `.grad` when a node affects the output through multiple paths.

`Parameter`
: A value or tensor intended to be updated by an optimizer.

`Context`
: The owner of allocations, graph nodes, temporary buffers, and error state.

`Broadcasting`
: Forward shape expansion where smaller tensors behave as if repeated. Backward requires reducing gradients back to the original shape.

`Finite difference`
: Numerical approximation used to verify analytical gradients.

## Example C API Direction

This is not final. It is a shape to aim toward.

```c
#include <valgrad.h>

int main(void) {
    vg_context_t *ctx = vg_context_create();

    vg_value_t *x = vg_value(ctx, 2.0);
    vg_value_t *w = vg_value(ctx, -3.0);
    vg_value_t *b = vg_value(ctx, 1.0);

    vg_value_t *y = vg_add(ctx, vg_mul(ctx, x, w), b);
    vg_value_t *loss = vg_mul(ctx, y, y);

    vg_backward(loss);

    printf("loss=%f\n", vg_data(loss));
    printf("dw=%f\n", vg_grad(w));
    printf("db=%f\n", vg_grad(b));

    vg_context_free(ctx);
    return 0;
}
```

## Example Python API Direction

This is also not final.

```python
import valgrad as vg

x = vg.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = vg.tensor([[0.0], [1.0], [1.0], [0.0]])

model = vg.MLP([2, 8, 1])

for step in range(1000):
    pred = model(x)
    loss = vg.mse(pred, y)
    loss.backward()
    model.step(lr=0.05)
    model.zero_grad()
```

The Python API can be nice, but it should not require the C core to become complicated too early.

## What To Build First

The first real implementation target:

```text
Scalar reverse-mode autodiff in C with finite-difference tests and a curve-fitting demo.
```

That is the seed.

Everything else grows from it.

## Agent Guidance

This section is for future AI coding agents or for me when I forget the project constraints. The human-facing north star above is more important than this section.

### Project Intent

Valgrad is a pure-C autodiff learning project. Keep the implementation small, inspectable, and test-driven. Do not turn it into a broad ML framework.

### Default Technical Choices

- Use C for the core.
- Use C11 unless the project later chooses otherwise.
- Use the `vg_` prefix for public C symbols.
- Prefer an explicit `vg_context_t` over globals.
- Prefer simple contiguous tensors before strides, views, and broadcasting.
- Keep Python as a wrapper layer, not the core implementation.
- Add tests with every math operation.

### Before Editing

- Inspect the existing files and current build/test commands.
- Preserve the current direction unless the user explicitly changes it.
- Do not silently introduce C++, Go, Rust, CUDA, or large dependencies.
- Do not redesign the whole project to solve a narrow issue.

### Coding Style

- Keep APIs boring and explicit.
- Prefer clear structs and functions over clever macros.
- Use comments only where they explain non-obvious math, memory ownership, or graph behavior.
- Keep examples runnable.
- Keep public headers readable.

### Testing Expectations

For any autodiff operation:

- Test forward value.
- Test backward gradient.
- Add or update finite-difference checks.
- Include shared-graph or accumulation tests when relevant.

For tensor changes:

- Test shape errors.
- Test non-square matmul where relevant.
- Test reductions.
- Test gradient shape and values.

### Things Agents Should Avoid

- Do not optimize before correctness.
- Do not add a neural network abstraction before the lower-level autodiff API works.
- Do not add broadcasting unless the basic tensor backward path is already correct.
- Do not make Python bindings the main design driver too early.
- Do not hide memory ownership behind vague helper functions.
- Do not add generated code, build-system churn, or formatting churn unless needed.

### Completion Standard

A change is not done until:

- The relevant test or example passes.
- New math has gradient coverage.
- The public API remains small.
- The change supports the scalar-first, tensor-second, Python-later roadmap.


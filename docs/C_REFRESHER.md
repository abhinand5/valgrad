# C Refresher for Valgrad

A focused, few-hours refresher for someone who knows C (and C++) but has been writing
Python for a year. It covers **exactly** the C you need to build Valgrad — scalar and
tensor reverse-mode autodiff — and nothing you won't use.

Every section has a Python contrast and short exercises. Solutions are at the end.
The exercises deliberately build toward a tiny scalar autograd engine, so finishing
this doc warms you up for Milestone 1.

How to use this: read a section, do its exercises in a scratch file, move on. Budget
~3–4 hours. Don't skip the memory and pointer sections even if they feel familiar —
that's where the bugs live.

Setup once:

```sh
mkdir -p /tmp/cref && cd /tmp/cref
```

Compile every exercise with warnings and sanitizers on. This is non-negotiable for a
math/memory project — the sanitizer catches the bugs that "look plausible":

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -g ex.c -o ex && ./ex
```

Make an alias so you stop thinking about it:

```sh
alias ccdev='cc -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -g'
# usage: ccdev ex.c -o ex && ./ex
```

---

## 0. The mental shifts from Python

Internalize these five before touching pointers. Most of your bugs will be a Python
habit leaking through.

1. **Nothing is automatic.** No GC, no bounds checking, no `len()`, no resizing. If you
   `malloc` it, you `free` it. If you index past the end, the program is *wrong* — it
   may still appear to work, which is worse than crashing.

2. **Variables are boxes, not labels.** In Python `b = a` makes two names for one
   object. In C `b = a` *copies the bytes*. To share, you pass a pointer. This single
   distinction drives the whole language.

3. **Everything has a fixed, known size at compile time** (except heap allocations).
   `sizeof(struct vg_value)` is a number the compiler knows. Arrays don't carry their
   length — you pass it alongside, always.

4. **Types are not optional and not dynamic.** A `double` is 8 bytes forever. There's no
   duck typing; there's `void *` (a pointer to anything) and manual casts, used sparingly.

5. **Undefined behavior (UB) is real and silent.** Reading uninitialized memory, freeing
   twice, signed overflow, out-of-bounds access — the compiler assumes you never do these
   and optimizes accordingly. The sanitizer is how you find them. Treat any sanitizer
   report as a real bug, never noise.

---

## 1. The build pipeline (do this first)

You need to be fluent in *compile → link → run* and a Makefile, because you'll rebuild
hundreds of times.

A C program becomes an executable in stages:

- **Preprocess** — `#include`, `#define`, `#ifdef` are textually expanded.
- **Compile** — each `.c` (a *translation unit*) becomes an object file `.o`.
- **Link** — object files + libraries are stitched into one executable. "Undefined
  reference" errors happen here: you declared a function but never linked its definition.

```sh
cc -c value.c -o value.o      # compile only
cc -c context.c -o context.o
cc value.o context.o -lm -o demo   # link (-lm = math library, for tanh/exp/log)
```

> **Gotcha that will bite you:** `tanh`, `exp`, `log`, `pow`, `sqrt` live in libm. On
> Linux you must link `-lm` or you get "undefined reference to `tanh`". Python never
> made you think about this.

### A starter Makefile

This is enough for the whole project's early life. Copy it into the repo later.

```make
CC      := cc
CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -g
CFLAGS  += -fsanitize=address,undefined      # comment out for release builds
LDFLAGS := -lm
INCLUDES:= -Iinclude

SRC     := $(wildcard src/*.c)
OBJ     := $(SRC:.c=.o)

demo: $(OBJ) examples/scalar_curve_fit.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test: $(OBJ) tests/test_value.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o run_tests $(LDFLAGS) && ./run_tests

clean:
	rm -f src/*.o demo run_tests

.PHONY: test clean
```

Make essentials: a *rule* is `target: prerequisites` then a TAB-indented recipe.
`$@` = target, `$<` = first prerequisite, `$^` = all prerequisites. Tabs, not spaces —
Make is famously strict about this.

**Exercise 1.** Write `hello.c` that prints `tanh(1.0)` using `#include <math.h>`.
Compile it *without* `-lm` and read the linker error. Then add `-lm`. Make sure you
recognize that specific error on sight.

---

## 2. Types, sizes, and the gotchas

For Valgrad you mostly use: `double` (all numeric data), `size_t` (sizes, counts,
indices), `int` (small loop counters, flags), `char` (strings, labels), and `bool`
(`#include <stdbool.h>`).

```c
#include <stddef.h>   // size_t
#include <stdint.h>   // int64_t etc. if you ever need fixed widths
#include <stdbool.h>  // bool, true, false
```

- **`size_t`** is unsigned and the type of `sizeof`. Use it for shapes, `ndim`,
  `numel`, loop indices over arrays. The plan's API uses it everywhere (`size_t n`,
  `const size_t *shape`).

- **Unsigned underflow is a classic trap.** This loops forever:

  ```c
  for (size_t i = n - 1; i >= 0; i--) { /* i >= 0 is ALWAYS true */ }
  ```

  Count down correctly:

  ```c
  for (size_t i = n; i-- > 0; ) { /* use i */ }   // post-decrement trick
  ```

- **Integer division truncates.** `1 / 2 == 0`. `1.0 / 2 == 0.5`. When you compute
  `dz / numel(x)` for a mean's backward, make sure the numerator is `double`.

- **`printf` format specifiers** must match the type or it's UB:
  `%d` int, `%zu` size_t, `%f` double, `%p` pointer, `%s` C-string.

  ```c
  printf("ndim=%zu data=%f ptr=%p\n", ndim, data, (void *)p);
  ```

- **Floating point compares.** Never `a == b` for doubles. For gradient checks use a
  tolerance:

  ```c
  #include <math.h>
  bool close(double a, double b, double tol) { return fabs(a - b) <= tol; }
  ```

**Exercise 2.** Print `sizeof` of `double`, `size_t`, `int`, and a pointer on your
machine. Then write a countdown loop from `n-1` to `0` over `size_t` that doesn't
infinite-loop, and verify with `n = 5`.

---

## 3. Pointers — the core skill

A pointer is a variable holding a memory address. This is how C shares and mutates data
across function boundaries — the thing Python does invisibly.

```c
int x = 10;
int *p = &x;     // & = "address of"; p points at x
*p = 20;         // * = "dereference"; write through p -> x is now 20
printf("%d\n", x);  // 20
```

### Why it matters here: pass-by-pointer to mutate

C passes everything *by value* (copies). To let a function change your variable, pass
its address. This is exactly how `vg_backward` fills in gradients and how `vg_sgd_step`
updates parameters.

```c
void set_grad(double *g, double v) { *g = v; }   // writes through the pointer

double grad = 0.0;
set_grad(&grad, 1.0);   // grad is now 1.0
```

### Pointer to pointer: `vg_value_t **params`

The plan's API takes `vg_value_t **params, size_t n`. Read it as "an array of `n`
pointers to values." You'll iterate it in the optimizer:

```c
void vg_sgd_step(vg_value_t **params, size_t n, double lr) {
    for (size_t i = 0; i < n; i++) {
        params[i]->data -= lr * params[i]->grad;   // -> is (*params[i]).field
    }
}
```

- `params[i]` is a `vg_value_t *`.
- `params[i]->data` dereferences and accesses a struct field (`->` is `(*p).field`).

### NULL

`NULL` is the "points at nothing" pointer. Dereferencing it crashes (segfault). The
project's error convention is *return `NULL` on failure*, so you'll check it constantly:

```c
vg_value_t *v = vg_value(ctx, 2.0);
if (v == NULL) { /* allocation or error; bail out */ }
```

**Exercise 3.** Write `void swap(double *a, double *b)` that swaps two doubles via
pointers. Call it on two variables and print before/after. Then write
`double sum_array(const double *xs, size_t n)` using pointer indexing `xs[i]`.

---

## 4. Arrays, pointer arithmetic, and row-major layout

This section is the foundation of the tensor milestones. Get it cold.

Arrays decay to pointers to their first element. `xs[i]` is literally `*(xs + i)`.
Arrays carry no length — you always pass `n` alongside.

```c
double xs[4] = {1, 2, 3, 4};
double *p = xs;        // decays to &xs[0]
printf("%f\n", p[2]);  // 3.0  == *(p + 2)
```

### Row-major / strides — the tensor mental model

A 2-D tensor is stored as one flat `double` array. For shape `(R, C)` in row-major
(C's convention), element `(i, j)` lives at flat index `i*C + j`. That `C` is the
*stride* of dimension 0; the stride of dimension 1 is `1`.

```c
// tensor a is (rows x cols), contiguous, row-major
double get(const double *data, size_t cols, size_t i, size_t j) {
    return data[i * cols + j];
}
```

Generalizing to N-D: precompute a `strides` array so index `(i0, i1, ...)` maps to
`sum(idx[k] * strides[k])`. For contiguous row-major, `strides[ndim-1] = 1` and
`strides[k] = strides[k+1] * shape[k+1]`. This is precisely the indexing machinery
Milestone 3 asks you to build.

Matmul `C = A @ B` where A is `(m,k)` and B is `(k,n)`:

```c
for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
        double acc = 0.0;
        for (size_t p = 0; p < k; p++)
            acc += A[i*k + p] * B[p*n + j];
        C[i*n + j] = acc;
    }
```

> **The #1 C bug:** writing out of bounds. `double a[4]; a[4] = 0;` is UB — there is no
> index 4. ASan catches this instantly, which is why you always build with it.

**Exercise 4.** Store a 2×3 matrix in a flat `double[6]` in row-major order. Write a
function that prints it as a grid using `i*cols + j`. Then write `transpose` into a new
`double[6]` (now 3×2) — you'll need this for matmul backward (`a.T`, `b.T`).

---

## 5. Structs and opaque types

Structs group fields. They're your `Value` and `Tensor`. Coming from Python, think of a
struct as a class with only data (no methods) — the "methods" are free functions taking
a pointer to the struct.

```c
struct vg_value {
    double  data;
    double  grad;
    struct vg_value *parents[2];   // up to 2 inputs (add, mul are binary)
    size_t  n_parents;
    void  (*backward)(struct vg_value *self);  // function pointer, see §7
    char   *label;
};
```

Access fields with `.` on a value, `->` on a pointer:

```c
struct vg_value v;
v.data = 2.0;            // direct

struct vg_value *p = &v;
p->data = 3.0;           // through pointer; same as (*p).data
```

### typedef and opaque pointers

The project uses `typedef struct vg_value vg_value_t;` so callers write `vg_value_t *`.
Better still, the *public header* can declare the type without defining its fields —
an **opaque type**. Callers get a pointer they can't peek inside, which enforces the
"small public API / hide internals" principle from the plan.

```c
// valgrad.h  (public)
typedef struct vg_value vg_value_t;   // declared, not defined
double vg_data(const vg_value_t *v);  // accessors instead of field access

// value.c  (private)
struct vg_value { double data; double grad; /* ... */ };  // full definition here
double vg_data(const vg_value_t *v) { return v->data; }
```

Outside `value.c`, `sizeof(vg_value_t)` is unknown and `v->data` won't compile — exactly
the encapsulation you want.

**Exercise 5.** Define a `struct point { double x, y; }`. Write
`double dist(const struct point *a, const struct point *b)` using `->`. Then `typedef` it
to `point_t` and rewrite the signature.

---

## 6. Memory management and ownership

This is the heart of the project. The plan mandates an explicit `vg_context` arena that
owns all nodes — so you mostly *won't* be freeing individual values, but you must
understand the primitives to build the arena.

### The malloc family

```c
#include <stdlib.h>

double *a = malloc(n * sizeof *a);   // uninitialized memory
double *b = calloc(n, sizeof *b);    // zero-initialized (great for grads = 0)
double *c = realloc(a, m * sizeof *a); // grow/shrink; may move the block
free(b);                             // release; never use b after this
```

Idioms that prevent bugs:

- **`sizeof *ptr`, not `sizeof(type)`.** `malloc(n * sizeof *a)` stays correct even if
  you change `a`'s type. (Tip: in C you don't cast `malloc`'s return.)
- **Always check for `NULL`.** `malloc` can fail. The project's convention is to
  propagate that as a `NULL` return + context error.
- **`calloc` for gradients.** Zero-initialized is exactly what `grad` wants.
- **Every `malloc` has exactly one `free`.** Double-free and use-after-free are UB;
  ASan flags both.

### Ownership: the question you must answer for every allocation

The plan's "Explicit Ownership" principle is the real curriculum. For each pointer ask:
*who allocates it, who frees it, does it own its data or borrow it?* Encode the answer
in your API. Borrowed pointers (e.g. `params[]` passed to the optimizer) are *not* freed
by the callee.

### A minimal arena (this is basically `vg_context`)

An arena owns many allocations and frees them all at once. This matches "context owns all
values; `vg_context_free(ctx)` frees everything." Simplest version — a growable array of
pointers:

```c
typedef struct {
    void  **blocks;   // every allocation we handed out
    size_t  count;
    size_t  cap;
} arena_t;

void *arena_alloc(arena_t *a, size_t bytes) {
    if (a->count == a->cap) {                    // grow the bookkeeping array
        size_t newcap = a->cap ? a->cap * 2 : 8;
        void **grown = realloc(a->blocks, newcap * sizeof *grown);
        if (!grown) return NULL;                 // caller checks NULL
        a->blocks = grown;
        a->cap = newcap;
    }
    void *p = calloc(1, bytes);
    if (!p) return NULL;
    a->blocks[a->count++] = p;
    return p;
}

void arena_free(arena_t *a) {
    for (size_t i = 0; i < a->count; i++) free(a->blocks[i]);
    free(a->blocks);
    a->count = a->cap = 0;
    a->blocks = NULL;
}
```

Now every `vg_value` comes from `arena_alloc(ctx, sizeof(vg_value_t))` and you never
hand-free a single node. This is *the* pattern that makes the training loop's
"new graph every iteration" sane.

**Exercise 6.** Implement the `arena_t` above. Allocate 1000 `double`s through it (each
a separate `arena_alloc`), write to each, then `arena_free`. Build with ASan and confirm
zero leaks and zero errors. (Leak check: `ASAN_OPTIONS=detect_leaks=1 ./ex`.)

---

## 7. Function pointers — how backward closures work

Reverse-mode autodiff stores, on each node, *the function that propagates its gradient
to its parents*. In C that's a **function pointer**. This is the one "advanced" feature
the project genuinely needs.

Syntax (read inside-out): `void (*backward)(vg_value_t *self)` is "a pointer named
`backward` to a function taking `vg_value_t *` and returning void."

```c
// each op sets node->backward to its local rule
static void add_backward(vg_value_t *self) {
    // z = a + b  =>  da += dz;  db += dz
    self->parents[0]->grad += self->grad;
    self->parents[1]->grad += self->grad;
}

static void mul_backward(vg_value_t *self) {
    // z = a * b  =>  da += dz*b;  db += dz*a
    vg_value_t *a = self->parents[0], *b = self->parents[1];
    a->grad += self->grad * b->data;
    b->grad += self->grad * a->data;
}

vg_value_t *vg_add(vg_context_t *ctx, vg_value_t *a, vg_value_t *b) {
    vg_value_t *out = arena_alloc(ctx, sizeof *out);
    out->data = a->data + b->data;
    out->parents[0] = a; out->parents[1] = b; out->n_parents = 2;
    out->backward = add_backward;   // store the rule
    return out;
}
```

The backward pass then just walks nodes in reverse topological order and calls
`node->backward(node)`. C has no closures, so the node struct *is* the closure: it
carries the parents and data the function needs via `self`.

> **Note `static` on the backward fns:** `static` at file scope means "internal linkage"
> — the symbol isn't visible to other translation units. Use it for all your private
> helpers so the public API stays small (the plan's exact goal).

**Exercise 7.** Define `typedef double (*binop)(double, double);`. Write `add` and `mul`
matching it, put them in an array `binop ops[] = {add, mul};`, and call each on `(3, 4)`
in a loop. This is the same mechanism as `node->backward`, simplified.

---

## 8. Dynamic arrays (growable buffers)

You'll need growable arrays repeatedly: collecting parameters, building the topological
order for backward, accumulating nodes. There's no `list.append`; you manage
`{data, count, capacity}` and `realloc` on growth — the same grow-by-doubling pattern as
the arena.

```c
typedef struct {
    vg_value_t **data;
    size_t count, cap;
} value_vec;

bool vec_push(value_vec *v, vg_value_t *item) {
    if (v->count == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 8;
        vg_value_t **g = realloc(v->data, nc * sizeof *g);
        if (!g) return false;
        v->data = g; v->cap = nc;
    }
    v->data[v->count++] = item;
    return true;
}
```

You'll use exactly this to build the topo order in `vg_backward`: DFS from the root,
push each node after visiting its parents, then iterate the vector in reverse calling
`backward`.

**Exercise 8.** Implement `value_vec` with `vec_push` and `vec_free`. Push 20 integers'
worth of dummy pointers (or change the element type to `int` for practice), print them,
free. Confirm clean under ASan.

---

## 9. Strings (just enough for labels)

A C string is a `char *` to a NUL-terminated (`'\0'`) byte array. The plan's
`vg_set_label` stores a name on a node. The subtlety is *ownership*: if you keep a
caller's `const char *`, you're borrowing (it may change/disappear); if you want to own
it, copy with `strdup` and `free` it later.

```c
#include <string.h>

void vg_set_label(vg_value_t *v, const char *label) {
    free(v->label);             // free any previous (free(NULL) is safe)
    v->label = strdup(label);   // malloc + copy; we now own this
}
```

(If the arena owns everything, you might instead store labels in the arena. Either way,
*decide and document who owns the bytes* — that's the whole lesson.)

`strdup` is POSIX; if `-std=c11` hides it, compile with `-D_POSIX_C_SOURCE=200809L` or
write your own (`malloc(strlen(s)+1)` then `strcpy`).

**Exercise 9.** Write `char *my_strdup(const char *s)` using `strlen`, `malloc`, and
`memcpy`. Test it, then `free` the result. Verify clean under ASan.

---

## 10. const-correctness, headers, and separate compilation

### const

`const` documents and enforces "I won't modify this." The plan uses it on read-only
accessors: `double vg_data(const vg_value_t *v)`. Read pointer-to-const right-to-left:
`const vg_value_t *v` = "pointer to a const value" (you can't write `v->data`). Use it
for every input you only read — it's free documentation the compiler checks.

### Headers and include guards

A header declares the public interface; the `.c` defines it. Guard every header so it's
safe to include twice:

```c
// valgrad.h
#ifndef VALGRAD_H
#define VALGRAD_H

#include <stddef.h>

typedef struct vg_context vg_context_t;
typedef struct vg_value   vg_value_t;

vg_context_t *vg_context_create(void);
void          vg_context_free(vg_context_t *ctx);
vg_value_t   *vg_value(vg_context_t *ctx, double data);
double        vg_data(const vg_value_t *v);

#endif // VALGRAD_H
```

- **Declaration vs definition:** the header *declares* (promises a function exists); the
  `.c` *defines* (provides the body). Calling a declared-but-not-linked function is the
  "undefined reference" linker error.
- **`(void)` parameter list** means "takes no arguments." Empty `()` in C means
  "unspecified" — always write `(void)` for no-arg functions.
- Include `<stddef.h>` in the header if it mentions `size_t`.

**Exercise 10.** Split Exercise 3's `sum_array` into `mathutil.h` (declaration +
guard) and `mathutil.c` (definition). Write `main.c` that includes the header and calls
it. Compile as two units and link: `ccdev -c mathutil.c -o mathutil.o && ccdev main.c
mathutil.o -o app`. Introduce a typo in the call to *see* the linker error.

---

## 11. Error handling, the boring way

C has no exceptions. The plan picks one strategy: **return `NULL` on failure, stash an
error in the context.** Use it everywhere; don't mix styles.

```c
struct vg_context { /* ... arena ... */ const char *error; };

vg_value_t *vg_value(vg_context_t *ctx, double data) {
    vg_value_t *v = arena_alloc(ctx, sizeof *v);
    if (!v) { ctx->error = "out of memory"; return NULL; }
    v->data = data;
    return v;
}

const char *vg_context_error(const vg_context_t *ctx) { return ctx->error; }
```

For shape mismatches in tensor ops, same shape: return `NULL`, set `ctx->error =
"shape mismatch in matmul"`. Callers check the return; tests assert that bad shapes
yield `NULL`. Use `assert()` (`#include <assert.h>`) only for *programmer* invariants
that should never happen, not for user-facing errors.

**Exercise 11.** Write `vg_value_t *checked_div(ctx, a, b)` that returns `NULL` and sets
an error string when `b->data == 0.0`, otherwise returns a value node. Test both paths
and print the error message on failure.

---

## 12. Debugging toolkit

Know these three; they replace the `print`/`pdb` reflexes from Python.

- **AddressSanitizer / UBSan** (compile-time `-fsanitize=address,undefined`): catches
  out-of-bounds, use-after-free, double-free, leaks, signed overflow, misaligned access.
  Your first line of defense. Read the report's first stack frame — it's almost always
  the real culprit.

- **Valgrind** (`valgrind --leak-check=full ./demo`): alternative leak/memory checker, no
  recompile needed. ASan is faster and usually enough; reach for Valgrind if ASan can't
  build.

- **gdb** (`gdb ./demo`, then `run`, `bt` for backtrace on crash, `break vg_backward`,
  `print v->grad`, `next`/`step`). Compile with `-g` for symbols. For a segfault, just
  `run` then `bt` — it points at the offending line.

The finite-difference gradient check from the plan is itself a debugging tool: if
analytic and numeric gradients disagree beyond tolerance, your `backward` rule is wrong.

```c
// central difference: df/dx ≈ (f(x+h) - f(x-h)) / 2h
double numeric_grad(double (*f)(double), double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}
```

**Exercise 12.** Write a function with a deliberate out-of-bounds write
(`double a[3]; a[3] = 1.0;`). Compile with ASan, run it, and read the report until you
can point at the exact line. This trains your eye for the report format.

---

## 13. Capstone: a 60-line scalar autograd

Tie it all together. This is Milestone 1 in miniature and uses every section above:
structs (§5), arena (§6), function pointers (§7), dynamic array for topo order (§8),
pointer-to-pointer params (§3).

**Exercise 13 (the real one).** Build a single-file `mini_autograd.c` that:

1. Defines `struct value { double data, grad; struct value *parents[2]; size_t np;
   void (*backward)(struct value*); };`
2. Has an arena (§6) owning all `value`s, via `value *make(arena*, double data)`.
3. Implements `v_add`, `v_mul`, `v_tanh` (forward + a `static` backward each).
4. Implements `backward(value *root)`: build reverse-topo order with a `value_vec`
   (§8) via DFS over `parents`, set `root->grad = 1.0`, then walk the order in reverse
   calling each node's `backward`.
5. Verifies one expression — e.g. `L = tanh(x*w + b)` — against finite differences
   (§12) for `dL/dx`, `dL/dw`, `dL/db`. They should match to ~1e-6.
6. Frees the arena. Builds clean under `-fsanitize=address,undefined`.

If you can do Exercise 13 from scratch in under an hour, you're ready for the real
`src/value.c`. If you get stuck on a piece, that piece is the section to reread.

---

## Quick reference card

```
&x            address of x
*p            value p points to
p->f          (*p).f   field through pointer
a[i]          *(a + i)
NULL          the null pointer; check after every alloc
sizeof *p     size of what p points to (preferred over sizeof(type))

malloc(n*sizeof *a)  uninit heap      calloc(n, sizeof *a)  zeroed heap
realloc(a, m)        grow/shrink      free(a)               release once
strdup(s)            owned copy       memcpy(d, s, n)        raw copy

size_t        unsigned size/index/count   %zu
double        all numeric data            %f
const T *p    read-only input
void f(void)  takes no args
static fn     file-private (keep API small)

build: cc -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -g x.c -lm -o x
```

### Top bugs to watch (all caught by ASan/warnings)

1. Forgetting `-lm` → "undefined reference to tanh".
2. Out-of-bounds array write (off-by-one in matmul loops).
3. `size_t` countdown loop never terminating (`i >= 0`).
4. Use-after-free / double-free (especially when arena and manual free mix — pick one).
5. Uninitialized `grad` (use `calloc`, or zero it explicitly).
6. Integer division where you meant float (`dz / numel`).
7. Returning a pointer to a stack local (it dies when the function returns).
8. Comparing doubles with `==` instead of a tolerance.

---

## Solutions

Try each exercise before reading. Solutions are intentionally terse.

**1.** `#include <math.h>` then `printf("%f\n", tanh(1.0));`. Without `-lm`: `undefined
reference to 'tanh'` at link time. With `-lm`: prints `0.761594`.

**2.**
```c
printf("%zu %zu %zu %zu\n", sizeof(double), sizeof(size_t), sizeof(int), sizeof(void*));
for (size_t i = 5; i-- > 0; ) printf("%zu ", i);   // 4 3 2 1 0
```

**3.**
```c
void swap(double *a, double *b) { double t = *a; *a = *b; *b = t; }
double sum_array(const double *xs, size_t n) {
    double s = 0; for (size_t i = 0; i < n; i++) s += xs[i]; return s;
}
```

**4.**
```c
void print_mat(const double *m, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) printf("%6.1f ", m[i*cols + j]);
        putchar('\n');
    }
}
void transpose(const double *in, double *out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            out[j*rows + i] = in[i*cols + j];   // out is (cols x rows)
}
```

**5.**
```c
typedef struct point { double x, y; } point_t;
double dist(const point_t *a, const point_t *b) {
    double dx = a->x - b->x, dy = a->y - b->y;
    return sqrt(dx*dx + dy*dy);   // needs -lm
}
```

**6.** See the `arena_t` in §6. Driver:
```c
arena_t a = {0};
for (int i = 0; i < 1000; i++) { double *d = arena_alloc(&a, sizeof *d); *d = i; }
arena_free(&a);
```
`{0}` zero-initializes all fields — important so `blocks`/`cap`/`count` start clean.

**7.**
```c
typedef double (*binop)(double, double);
static double add(double a, double b) { return a + b; }
static double mul(double a, double b) { return a * b; }
binop ops[] = { add, mul };
for (int i = 0; i < 2; i++) printf("%f\n", ops[i](3, 4));  // 7, 12
```

**8.** See `value_vec`/`vec_push` in §8; `vec_free` is
`free(v->data); v->data = NULL; v->count = v->cap = 0;`.

**9.**
```c
char *my_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);   // copies the trailing '\0' too
    return p;
}
```

**10.** `mathutil.h`: guard + `#include <stddef.h>` + `double sum_array(const double *, size_t);`.
`mathutil.c`: `#include "mathutil.h"` + the definition. The typo (`sum_arry`) yields a
link-time `undefined reference to 'sum_arry'`.

**11.**
```c
vg_value_t *checked_div(vg_context_t *ctx, vg_value_t *a, vg_value_t *b) {
    if (b->data == 0.0) { ctx->error = "division by zero"; return NULL; }
    return vg_value(ctx, a->data / b->data);   // (real version records parents/backward)
}
```

**12.** ASan prints `ERROR: AddressSanitizer: stack-buffer-overflow`, a `WRITE of size 8`,
and a stack trace whose top frame is your `a[3] = 1.0;` line. That top frame is the skill
to internalize.

**13.** Sketch of the non-obvious parts:
```c
static void tanh_backward(value *s) {
    double t = s->data;                       // s->data already = tanh(input)
    s->parents[0]->grad += (1.0 - t*t) * s->grad;
}
static void build_topo(value *v, value_vec *order, /* visited set */ ...) {
    if (already_visited(v)) return;
    mark_visited(v);
    for (size_t i = 0; i < v->np; i++) build_topo(v->parents[i], order, ...);
    vec_push(order, v);                        // post-order => parents before child
}
void backward(value *root) {
    value_vec order = {0};
    build_topo(root, &order, ...);
    root->grad = 1.0;
    for (size_t i = order.count; i-- > 0; )    // reverse topo
        if (order.data[i]->backward) order.data[i]->backward(order.data[i]);
    vec_free(&order);
}
```
For "visited," a simple approach for small graphs is a `bool visited` flag on each node
(reset per backward) or a second `value_vec` you linear-scan. Don't over-engineer it yet.

---

That's the whole surface area Valgrad needs. When something feels rusty mid-build, jump
back to the relevant section — they're ordered the way the project will exercise them:
build → types → pointers → arrays → structs → memory → function pointers → the rest.
```

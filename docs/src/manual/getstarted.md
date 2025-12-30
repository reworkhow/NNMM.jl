# Get Started

## Installation

To install Julia, please go to the [official Julia website](https://julialang.org/downloads/).
Please see [platform specific instructions](https://julialang.org/downloads/platform.html)
if you have trouble installing Julia.

To install the NNMM package, use the following command inside the Julia REPL (or IJulia Notebook):
```julia
using Pkg
Pkg.add("NNMM")
```

To load the NNMM package:

```julia
using NNMM
```

### Development Version

To use the latest/beta features under development:
```julia
Pkg.add(url="https://github.com/reworkhow/NNMM.jl")
```

### Jupyter Notebook

If you prefer "reproducible research", an interactive Jupyter Notebook interface is available
for Julia (and therefore NNMM). The Jupyter Notebook is an open-source web application for creating
and sharing documents that contain live code, equations, visualizations and explanatory text.
To install IJulia for Jupyter Notebook, please go to [IJulia](https://github.com/JuliaLang/IJulia.jl).

## Multi-threaded Parallelism

NNMM supports multi-threaded parallelism for faster computation. The number of threads can be checked by:
```julia
Threads.nthreads()
```

To start Julia with multiple threads (requires Julia 1.5+):
```bash
julia --threads 4
```

## Access Documentation

!!! warning

    Please load the NNMM package first.

To show basic information about NNMM in REPL or IJulia notebook, use `?NNMM` and press enter.

For help on a specific function, type `?` followed by its name, e.g., `?runNNMM` and press enter.

The full documentation is available [here](http://reworkhow.github.io/NNMM.jl/latest/index.html).

## Run Your Analysis

There are several ways to run your analysis:

### Interactive Session (REPL)

Start an interactive session by double-clicking the Julia executable or running `julia` from the command line:

```julia
julia> using NNMM
julia> # your analysis code here
```

To evaluate code written in a file `script.jl` in REPL:

```julia
julia> include("script.jl")
```

To exit the interactive session, type `^D` (control + d) or `quit()`.

### Command Line

To run code in a file non-interactively from the command line:

```bash
julia script.jl
```

If you want to pass arguments to your script:
```bash
julia script.jl arg1 arg2
```
where arguments `arg1` and `arg2` are passed as `ARGS[1]` and `ARGS[2]` of type *String*.

### Jupyter Notebook

To run code in Jupyter Notebook, please see [IJulia](https://github.com/JuliaLang/IJulia.jl).

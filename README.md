# ClimaInterpolations.jl

A collection of interpolation tools for packages in the CliMA software ecosystem.

## Code Formatting with `JuliaFormatter.jl`

One of the tests consists in checking that the code is uniformly formatted. We
use [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) to achieve
consistent formatting. Here's how to use it:

You can either install in your base environment with
``` sh
julia -e 'using Pkg; Pkg.add("JuliaFormatter")'
```
or use it from within the `TestEnv` (or base) environments (see previous section).

Then, you can format the package running:
``` julia
using JuliaFormatter; format(".")
```
or just with `format(".")` if the package is already imported.

The rules for formatting are defined in the `.JuliaFormatter.toml`.

If you are used to formatting from the command line instead of the REPL, you can
install `JuliaFormatter` in your base environment and call
``` sh
julia -e 'using JuliaFormatter; format(".")'
```
You could also define a shell alias
``` sh
alias julia_format_here="julia -e 'using JuliaFormatter; format(\".\")'"
```

> :note: Please, open an issue if you find workflow problems/friction with this
> system.

module Datasets

using Printf

"""
    dataset(file_name::AbstractString; dataset_name::AbstractString="")

Get the path to a built-in dataset file.

# Arguments
- `file_name::AbstractString`: The name of the file to retrieve
- `dataset_name::AbstractString=""`: Optional subdirectory name within the data folder

# Returns
- `String`: Full path to the requested data file

# Examples
```julia
phenofile = dataset("phenotypes.csv")
genofile = dataset("genotypes.txt", dataset_name="example")
```
"""
function dataset(file_name::AbstractString; dataset_name::AbstractString="")
    basename = joinpath(dirname(@__FILE__), "data", dataset_name)
    rdaname = joinpath(basename, string(file_name))
    if isfile(rdaname)
        return rdaname
    else
        error(@sprintf "Unable to locate file %s in %s\n" file_name dataset_name)
    end
end

export dataset

end # module Datasets

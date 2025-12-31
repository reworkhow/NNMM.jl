using Documenter, NNMM

makedocs(
    modules = [NNMM, NNMM.Datasets, NNMM.PedModule],
    doctest = false,
    clean = true,
    sitename = "NNMM.jl",
    authors = "Hao Cheng, Tianjing Zhao, Rohan Fernando, Dorian Garrick and contributors.",
    pages = Any[
        "Home" => "index.md",
        "Mixed Effects Neural Networks (NNMM)" => Any[
            "Part 1. Introduction" => "nnmm/Part1_introduction.md",
            "Part 2. NNMM" => "nnmm/Part2_NNMM.md",
            "Part 3. NNMM with intermediate omics" => "nnmm/Part3_NNMMwithIntermediateOmics.md",
            "Part 4. Partial connected neural network" => "nnmm/Part4_PartialConnectedNeuralNetwork.md",
            "Part 5. User-defined nonlinear function" => "nnmm/Part5_UserDefinedNonlinearFunction.md",
            "Part 6. Traditional genomic prediction" => "nnmm/Part6_TraditionalGenomicPrediction.md",
        ],
        "Manual" => Any[
            "Get Started" => "manual/getstarted.md",
            "Public Functions" => "manual/public.md",
        ],
        "Citing" => "citing/citing.md",
    ],
)

deploydocs(
    repo = "github.com/reworkhow/NNMM.jl.git",
    target = "build",
)

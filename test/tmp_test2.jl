
using NNMM,DataFrames,CSV,Statistics,NNMM.Datasets,DelimitedFiles,LinearAlgebra,Random
########################################################
#test NNMM
########################################################
data_dir = joinpath(@__DIR__, "tmp_test2_data")
mkpath(data_dir)

geno_path = dataset("genotypes0.csv")
g_path = [geno_path]

Random.seed!(123)
geno = NNMM.nnmm_get_genotypes(geno_path)
nind = length(geno.obsID)

omics_df = DataFrame(ID=geno.obsID)
for i in 1:5
    omics_df[!, Symbol("o$(i)")] = randn(nind)
end
omics_df.x1 = randn(nind)
omics_df.x2 = randn(nind)
omics_df.x3 = rand(["m", "f"], nind)
o_path = joinpath(data_dir, "omics.csv")
CSV.write(o_path, omics_df; missingstring="NA")

pheno_df = CSV.read(dataset("phenotypes.csv"), DataFrame)
pheno_df.x4 = randn(nrow(pheno_df))
pheno_df.x5 = rand(1:3, nrow(pheno_df))
y_path = joinpath(data_dir, "phenotypes.csv")
CSV.write(y_path, pheno_df[:, [:ID, :y1, :x4, :x5]]; missingstring="NA")

#if polygenic effects are included in the model, the pedigree relationship should be provided
pedfile    = dataset("pedigree.csv")
pedigree = get_pedigree(pedfile,separator=",",header=true);

#other user-defined relationship matrix
GRM = readdlm(dataset("GRM.csv"),',')
grm_names=GRM[:,1]
GRM=Float32.(Matrix(GRM[:,2:end]))+ I*0.0001

G2=0.5
G3=0.1


layers = [
    Layer(layer_name="geno", data_path=g_path), #g_path -> list if partial connected
    Layer(layer_name="omics", data_path=o_path, missing_value="NA"), # should also include other random&covariate data
    Layer(layer_name="phenotypes",data_path=y_path, missing_value="NA") # should also include other random&covariate data
];

equations = [ 
    Equation(
        from_layer_name="geno",
        to_layer_name="omics",
        equation="omics = intercept + x1 + x2 + x3 + ID + geno",
        omics_name = ["o1","o2","o3","o4","o5"],
        covariate=["x1","x2"],
        random=[
            (name="ID", pedigree=pedigree), 
            (name="x3",)],
        method="BayesC"),
    Equation(
        from_layer_name="omics",
        to_layer_name ="phenotypes",
        equation ="phenotypes = intercept + ID + x4 + x5 + omics", 
        phenotype_name = ["y1"],
        covariate=["x4"],
        random=[
            (name="x5", G=G3),
            (name="ID", pedigree=pedigree)],
        method="BayesC",
        activation_function="sigmoid")
];

tt=runNNMM(layers, equations; chain_length=10);

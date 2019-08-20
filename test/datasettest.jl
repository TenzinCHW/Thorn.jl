fname = "hello.jld2"
dset = Dataset(fname, "train")
@test isa(dset["hello", 1], typeof(rand(5)))

for item in dset
    @test isa(item, typeof(rand(5)))
end

swaptraintest!(dset, "test")
@test length(dset.activeset) == 2
resizeset!(dset, 50) # Make dataset 50% the full size
@test length(dset.activeset) == 1
shufflebyclass!(dset)

dloader = Dataloader(dset, false)
for item in dloader
    @test isa(item, typeof(rand(5)))
end


fname = "hello.jld2"
dset = Dataset(fname, "train")
@test isa(dset["hello", 1], typeof(rand(5)))

for item in dset
    @test isa(item, typeof(rand(5)))
end

swaptraintest!(dset, "test")
resizeset!(dset, 100)
shufflebyclass!(dset)

dloader = Dataloader(dset, false)
for item in dloader
    @test isa(item, typeof(rand(5)))
end


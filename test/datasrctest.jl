fname = "hello.jld2"
dsrc = Datasource(fname, "w")
value = rand(5)
datasrcwrite!(dsrc, "train/hello/1", value)
out = datasrcread(dsrc, "train/hello/1")
@test out == value
items = datasrcitems(dsrc, "train/hello")
@test items == ["1"]

datasrcwrite!(dsrc, "test/hello/1", value .+ 3)
datasrcwrite!(dsrc, "test/hello/2", value .+ 2)
close(dsrc)


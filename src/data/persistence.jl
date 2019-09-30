using JLD2

struct Datasource
    file::JLD2.JLDFile

    function Datasource(srcpath::String, mode::String="r")
        fname = validsrcname(srcpath)
        f = jldopen(fname, mode)
        new(f)
    end
end

function validsrcname(srcpath::String)
    isdir(srcpath) ? joinpath(srcpath, "datasource.jld2") : srcpath
end

function datasrcread(datasrc::Datasource, path::String)
    datasrc.file[path]
end

function datasrcreadbits(datasrc::Datasource, path::String)::Array{UInt8, 2}
    datasrcread(datasrc, path)
end

function datasrcwrite!(datasrc::Datasource, path::String, data)
    datasrc.file[path] = data
end

function datasrcwritebits!(datasrc::Datasource, path::String, data::Array{UInt8, Any})
    datasrcwrite!(datasrc, BitArray(data))
end

function datasrcitems(src::Datasource, pth::String)
    keys(src.file[pth])
end

Base.close(src::Datasource) = close(src.file)


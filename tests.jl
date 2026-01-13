ids = Dict()
for i in (1,1)
    id = objectid(i)
    if !isnothing(get(ids, id, nothing))
        print("copy found: both $i and $(ids[id]) have id $id !")
    else
        ids[id] = i
    end
end
print("done!")
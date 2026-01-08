# A POMDPs.jl implementation of the custom ABC model, as defined in the paper.

module ABCModel
using POMDPs, POMDPTools, QuickPOMDPs
export ABC, ABC_delayed

# function

function T(s,a)
    s=="terminal" && (return SparseCat(["terminal"], [1]))
    (a=="a" || a=="b") && (return SparseCat(["terminal"], [1]))
    # (s=="A") && (return SparseCat(["A", "B"], [0.79, 0.21]))
    # (s=="B") && (return SparseCat(["A", "B"], [0.19, 0.81]))
    (s=="A") && (return SparseCat(["A", "B"], [0.8, 0.2]))
    (s=="B") && (return SparseCat(["A", "B"], [0.2, 0.8]))
end

R(s,a) = ( (s=="A" && a=="a") || (s=="B" && a=="b")) ? 1 : 0
O(a,sp) = SparseCat(["nothing"],[1])

ABC(;discount=0.95) = QuickPOMDP(
    states = ["A","B","terminal"],
    actions=["a","b","c"],
    observations=["nothing"],
    discount=discount,

    transition = T,
    observation = O,
    reward = R,
    initialstate = SparseCat(["A", "B"], [0.5, 0.5]),
    isterminal = s -> s=="terminal"
)

function Tp(s,a)
    s=="terminal" && (return SparseCat(["terminal"], [1.0]))
    (a=="a" || a=="b") && (return SparseCat(["terminal"], [1.0]))
    s=="Ap" && (return SparseCat(["A"], [1.0]))
    s=="Bp" && (return SparseCat(["B"], [1.0]))
    s=="A" ? not_s = "B" : not_s = "A"
    return SparseCat([s,not_s], [0.8, 0.2])
    return SparseCat([s,not_s], [0.8, 0.2])
end

ABC_delayed(;discount=0.95) = QuickPOMDP(
    states = ["Ap", "Bp", "A","B","terminal"],
    actions=["a","b","c"],
    observations=["nothing"],
    discount=discount,

    transition = Tp,
    observation = O,
    reward = R,
    initialstate = SparseCat(["Ap", "Bp"], [0.5, 0.5]),
    isterminal = s -> s=="terminal"
)

end
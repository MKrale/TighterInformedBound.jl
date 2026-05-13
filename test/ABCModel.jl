"""
Running example used to explain TIB and variants.
"""

module ABCModel
using POMDPs, POMDPTools, QuickPOMDPs
export ABC

# function

function T(s,a)
    s=="terminal" && (return SparseCat(["terminal"], [1.0]))
    (a=="a" || a=="b") && (return SparseCat(["terminal"], [1.0]))
    s=="A" ? not_s = "B" : not_s = "A"
    return SparseCat([s,not_s], [0.8, 0.2])
end

R(s,a) = ( (s=="A" && a=="a") || (s=="B" && a=="b")) ? 1.0 : 0.0
O(a,sp) = SparseCat(["nothing"],[1.0])

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

end
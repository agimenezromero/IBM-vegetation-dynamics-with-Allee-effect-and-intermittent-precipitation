using Random
using Distributions
using StatsBase

# ============================================================
# PARAMETERS + VARIABLES
# ============================================================

mutable struct Parameters{T<:Real}
    # γ_c, σ_c, γ_p, σ_p, σ_d, c, p, d, f
    constants::Vector{T}
    total_rates::Vector{T}    # [D_total, P_total]
    L::T                      # system size
end

mutable struct Variables{T<:Real}
    x::Vector{T}
    y::Vector{T}

    D2::Vector{T}             # death squared rate (vars.rates[1,:])
    P::Vector{T}              # proliferation rate (vars.rates[2,:])
    D::Vector{T}              # competition contribution (vars.rates[3,:])

    N::Int
    switching_times::Vector{T}
end

# ============================================================
# PERIODIC BOUNDARY
# ============================================================

@inline function periodic(x, L)
    if x < -L / 2
        x + L
    elseif x >= L / 2
        x - L
    else
        x
    end
end

# ============================================================
# KERNELS
# ============================================================

@inline function competition_kernel(ξ2, params::Parameters)
    γc = params.constants[1]
    σc = params.constants[2]
    return γc * exp(-ξ2 / (2 * σc^2))
end

@inline function proliferation_kernel(ξ2, params::Parameters)
    γp = params.constants[3]
    σp = params.constants[4]
    return γp * exp(-ξ2 / (2 * σp^2))
end

# ============================================================
# INITIALIZATION + RATE RECOMPUTATION
# ============================================================

function initialize_vars!(vars::Variables, params::Parameters)
    N = vars.N
    fill!(vars.D, 0.0)
    fill!(vars.P, 0.0)
    fill!(vars.D2, 0.0)

    @inbounds for i in 1:N
        xi = vars.x[i]
        yi = vars.y[i]

        Di = 0.0
        Pi = 0.0

        @inbounds for j in 1:N
            j == i && continue
            dx = periodic(vars.x[j] - xi, params.L)
            dy = periodic(vars.y[j] - yi, params.L)
            ξ2 = dx * dx + dy * dy

            Di += competition_kernel(ξ2, params)
            Pi += proliferation_kernel(ξ2, params)
        end

        vars.D[i] = Di
        vars.P[i] = Pi + params.constants[7]          # + p
        vars.D2[i] = params.constants[6] + Di^2        # c + D^2
    end
end

function recompute_rates!(vars::Variables, params::Parameters)
    initialize_vars!(vars, params)
end

# ============================================================
# ADD PARTICLE (MIRRORING YOUR ORDER)
# ============================================================

function add_particle!(idx::Int, vars::Variables, params::Parameters)
    N_old = vars.N
    N_new = N_old + 1
    vars.N = N_new

    # ensure capacity
    if length(vars.x) < N_new
        newlen = max(2N_new, length(vars.x) * 2)
        resize!(vars.x, newlen)
        resize!(vars.y, newlen)
        resize!(vars.D, newlen)
        resize!(vars.P, newlen)
        resize!(vars.D2, newlen)
    end

    # offspring displacement: same as rand(MultivariateNormal([0,0], σ_d))
    σd = params.constants[5]
    dist = rand(MultivariateNormal([0.0, 0.0], σd))

    newx = periodic(vars.x[idx] + dist[1], params.L)
    newy = periodic(vars.y[idx] + dist[2], params.L)

    vars.x[N_new] = newx
    vars.y[N_new] = newy

    # mirror your loop/order:
    Di_new = 0.0
    Pi_new = 0.0

    @inbounds for i in 1:N_new
        xi = vars.x[i]
        yi = vars.y[i]

        dx = periodic(xi - newx, params.L)
        dy = periodic(yi - newy, params.L)
        ξ2 = dx * dx + dy * dy

        if i != N_new
            ρ_c = competition_kernel(ξ2, params)
            ρ_p = proliferation_kernel(ξ2, params)

            # new particle accumulates
            Di_new += ρ_c
            Pi_new += ρ_p

            # existing particle i updated
            vars.D[i] += ρ_c
            vars.D2[i] = params.constants[6] + vars.D[i]^2
            vars.P[i] += ρ_p
        end
    end

    vars.D[N_new] = Di_new
    vars.D2[N_new] = params.constants[6] + Di_new^2
    vars.P[N_new] = Pi_new + params.constants[7]
end

# ============================================================
# REMOVE PARTICLE (MIRRORING YOUR ORDER)
# ============================================================

function remove_particle!(idx::Int, vars::Variables, params::Parameters)
    N_old = vars.N
    N_new = N_old - 1

    oldx = vars.x[idx]
    oldy = vars.y[idx]

    # update other particles' rates
    @inbounds for i in 1:N_old
        i == idx && continue
        dx = periodic(vars.x[i] - oldx, params.L)
        dy = periodic(vars.y[i] - oldy, params.L)
        ξ2 = dx * dx + dy * dy

        ρ_c = competition_kernel(ξ2, params)
        ρ_p = proliferation_kernel(ξ2, params)

        vars.D[i] -= ρ_c
        vars.D2[i] = params.constants[6] + vars.D[i]^2
        vars.P[i] -= ρ_p
    end

    # compact arrays (move last into idx)
    if idx != N_old
        vars.x[idx] = vars.x[N_old]
        vars.y[idx] = vars.y[N_old]
        vars.D[idx] = vars.D[N_old]
        vars.P[idx] = vars.P[N_old]
        vars.D2[idx] = vars.D2[N_old]
    end

    vars.N = N_new
end

# ============================================================
# GILLESPIE COMPONENTS
# ============================================================

function compute_waiting_time(vars::Variables, params::Parameters)
    Dtot = 0.0
    Ptot = 0.0

    @inbounds for i in 1:vars.N
        Dtot += vars.D2[i]
        Ptot += vars.P[i]
    end

    params.total_rates[1] = Dtot
    params.total_rates[2] = Ptot

    W = Dtot + Ptot
    τ = -log(rand()) / W
    return τ, W
end

function proliferation_event!(vars::Variables, params::Parameters)
    # cumsum + findfirst, like your original
    cum = cumsum(view(vars.P, 1:vars.N))
    R = rand() * params.total_rates[2]
    idx = findfirst(cum .> R)
    add_particle!(idx, vars, params)
end

function death_event!(vars::Variables, params::Parameters)
    cum = cumsum(view(vars.D2, 1:vars.N))
    R = rand() * params.total_rates[1]
    idx = findfirst(cum .> R)
    remove_particle!(idx, vars, params)
end

function choose_apply_event!(reactions, orders, W, params::Parameters, vars::Variables)
    U = rand() * W

    @inbounds for i in 1:2
        if U < sum(params.total_rates[orders[1:i]])
            reactions[i](vars, params)

            if i != 1
                reactions[i-1], reactions[i] = reactions[i], reactions[i-1]
                orders[i-1], orders[i] = orders[i], orders[i-1]
            end

            break
        end
    end
end

# ============================================================
# MAIN IBM
# ============================================================

function IBM(t_max::Real, vars::Variables, params::Parameters; coupling="competition")
    N_t = Int[]
    t_res = Float64[]

    reactions = [death_event!, proliferation_event!]
    orders = [1, 2]

    t = 0.0
    kernel_type = "LP"
    t_hp = 0.0
    count2 = 1

    initialize_vars!(vars, params)

    push!(N_t, vars.N)
    push!(t_res, t)

    while t < t_max
        if kernel_type == "LP" &&
           count2 <= length(vars.switching_times) &&
           t > vars.switching_times[count2]

            if coupling == "competition"
                params.constants[1] /= params.constants[9]
            elseif coupling == "facilitation"
                params.constants[3] *= params.constants[9]
            end

            recompute_rates!(vars, params)
            kernel_type = "HP"
            count2 += 1

        elseif kernel_type == "HP" && t_hp > params.constants[8]
            if coupling == "competition"
                params.constants[1] *= params.constants[9]
            elseif coupling == "facilitation"
                params.constants[3] /= params.constants[9]
            end

            recompute_rates!(vars, params)
            kernel_type = "LP"
            t_hp = 0.0
        end

        vars.N == 0 && break

        τ, W = compute_waiting_time(vars, params)
        t += τ
        kernel_type == "HP" && (t_hp += τ)

        choose_apply_event!(reactions, orders, W, params, vars)

        push!(N_t, vars.N)
        push!(t_res, t)
    end

    return N_t, t_res
end

function IBM_save_all(t_max::Real, vars::Variables, params::Parameters;
    folder="Results", save_at=1.0, coupling="competition", verbose=false)

    N_t = Int[]
    t_res = Float64[]

    reactions = [death_event!, proliferation_event!]
    orders = [1, 2]

    t = 0.0
    t_save_at = 0.0

    initialize_vars!(vars, params)

    kernel_type = "LP"
    t_hp = 0.0
    count2 = 1

    # Save initial state
    push!(N_t, vars.N)
    push!(t_res, t)

    P_mat = [vars.x[1:vars.N]'; vars.y[1:vars.N]']
    writedlm(folder * "/P_" * string(t) * ".txt", P_mat)

    while t < t_max

        if verbose
            println("t = $t, N = $(vars.N), kernel = $kernel_type")
        end

        # --- Switching logic ---
        if kernel_type == "LP" &&
           count2 <= length(vars.switching_times) &&
           t > vars.switching_times[count2]

            if coupling == "competition"
                params.constants[1] /= params.constants[9]
            elseif coupling == "facilitation"
                params.constants[3] *= params.constants[9]
            end

            recompute_rates!(vars, params)
            kernel_type = "HP"
            count2 += 1

        elseif kernel_type == "HP" && t_hp > params.constants[8]

            if coupling == "competition"
                params.constants[1] *= params.constants[9]
            elseif coupling == "facilitation"
                params.constants[3] /= params.constants[9]
            end

            recompute_rates!(vars, params)
            kernel_type = "LP"
            t_hp = 0.0
        end

        # --- Extinction ---
        vars.N == 0 && break

        # --- Gillespie step ---
        τ, W = compute_waiting_time(vars, params)
        t += τ
        kernel_type == "HP" && (t_hp += τ)

        choose_apply_event!(reactions, orders, W, params, vars)

        # --- Save positions ---
        if t - t_save_at > save_at
            P_mat = [vars.x[1:vars.N]'; vars.y[1:vars.N]']
            writedlm(folder * "/P_" * string(t) * ".txt", P_mat)

            push!(N_t, vars.N)
            push!(t_res, t)

            t_save_at += save_at
        end
    end

    writedlm(folder * "/N_t.txt", N_t)
    writedlm(folder * "/t_res.txt", t_res)
end



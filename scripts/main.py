import numpy as np
from numpy.linalg import inv
import random

np.seterr(divide="ignore")


def parse_var(val):
    x_index = val.find("x")
    coef = val[:x_index]
    var = val[x_index:]
    return float(coef), var


def parse_instance(path):
    lines = []
    with open(path, "r") as file:
        lines = file.readlines()

    lines = [line.strip().split(sep=" ") for line in lines]

    model = {}
    model["problem_objective"] = lines[0][0]

    model["c"] = []
    model["x_obj"] = []
    model["x_vars"] = set()

    for val in lines[0][1:]:
        coef, var = parse_var(val)
        model["c"].append(coef)
        model["x_obj"].append(var)
        model["x_vars"].add(var)

    model["b"] = []
    model["type_equality"] = []
    model["A"] = []
    model["A_xvars"] = []

    count = 1
    for line in lines[1:]:
        if line[0] == "bounds":
            count = count + 1
            break
        coefs = []
        costr_vars = []
        for val in line[:-2]:
            coef, var = parse_var(val)
            coefs.append(coef)
            costr_vars.append(var)
        model["A"].append(coefs)
        model["A_xvars"].append(costr_vars)
        model["b"].append(float(line[-1]))
        model["type_equality"].append(line[-2])
        model["x_vars"].add(var)
        count = count + 1

    model["x_bound"] = []
    model["b_bound"] = []
    model["ineq_bound"] = []

    for line in lines[count:]:
        if len(line) == 3:
            model["x_bound"].append(line[0])
            model["ineq_bound"].append(line[1])
            model["b_bound"].append(line[2])
        else:
            model["x_bound"].append(line[0])
            model["ineq_bound"].append(line[1])
            model["b_bound"].append(None)
    model["count_constr"] = len(model["A"])
    model["count_vars"] = len(model["x_bound"])
    return model


def model_to_standard_form(model):
    great_index = 0
    for val in model["x_vars"]:
        if int(val[1:]) > great_index:
            great_index = int(val[1:])
    great_index = great_index + 1
    aux_model = model

    for i, val in enumerate(model["b"]):
        if val > 0:
            continue
        model["A"][i] = [-1 * x for x in model["A"][i]]
        model["b"][i] = -1 * val
        if model["type_equality"][i] == "<=":
            model["type_equality"][i] = ">="
        elif model["type_equality"][i] == ">=":
            model["type_equality"][i] = "<="

    for i, constr in enumerate(model["type_equality"]):
        if constr == "=":
            continue
        elif constr == "<=":
            model["A"][i].append(1)
        else:
            model["A"][i].append(-1)

        model["A_xvars"][i].append("x" + str(great_index))
        model["x_vars"].add("x" + str(great_index))
        great_index = great_index + 1

    new_model = {}
    new_model["c"] = np.array([0 for x in range(len(model["x_vars"]))])
    new_model["problem_objective"] = model["problem_objective"]
    for i, val in enumerate(model["x_obj"]):
        new_model["c"][int(val[1:]) - 1] = model["c"][i]
    if model["problem_objective"] == "max":
        new_model["c"] = new_model["c"] * -1

    new_model["A"] = np.array(
        [[0.0 for x in range(len(model["x_vars"]))] for y in range(len(model["A"]))]
    )

    for i, constr in enumerate(model["A_xvars"]):
        for j, val in enumerate(constr):
            new_model["A"][i][int(val[1:]) - 1] = model["A"][i][j]
    new_model["b"] = np.array(model["b"])
    new_model["old_model"] = aux_model

    return new_model


def to_dual(model):
    aux_p_vars = ["p" + str(i + 1) for i in range(len(model["A"]))]
    fo = "max " if model["problem_objective"] == "min" else "min "
    for i, val in enumerate(model["A"]):
        if i == 0:
            fo += str(model["b"][i]) + aux_p_vars[i] + " "
        else:
            symb = "+" if model["b"][i] >= 0 else ""
            fo += symb + str(model["b"][i]) + aux_p_vars[i] + " "

    constr_count = model["old_model"]["count_constr"]
    var_count = model["old_model"]["count_vars"]
    At = model["A"][:constr_count, :var_count].transpose()
    constrs = []
    bounds = []

    for i, row in enumerate(At):
        constr = ""

        c = (
            model["c"][i] * -1
            if model["old_model"]["problem_objective"] == "max"
            else model["c"][i]
        )
        for j, col in enumerate(row):
            if j == 0:
                constr += str(col) + aux_p_vars[j] + " "
            else:
                symb = "+" if col >= 0 else ""
                constr += symb + str(col) + aux_p_vars[j] + " "

        if model["problem_objective"] == "min":
            if model["old_model"]["ineq_bound"][i] == ">=":
                constr += " <= " + str(c)
            elif model["old_model"]["ineq_bound"][i] == "<=":
                constr += " >= " + str(c)
            else:
                constr += " = " + str(c)
        elif model["problem_objective"] == "max":
            if model["old_model"]["ineq_bound"][i] == ">=":
                constr += " >= " + str(c)
            elif model["old_model"]["ineq_bound"][i] == "<=":
                constr += " <= " + str(c)
            else:
                constr += " = " + str(c)
        constrs.append(constr)

    for i in range(model["old_model"]["count_constr"]):
        bound = "p" + str(i + 1)
        if model["problem_objective"] == "min":
            if model["old_model"]["type_equality"][i] == ">=":
                bound += " >= 0"
            elif model["old_model"]["type_equality"][i] == "<=":
                bound += " <= 0"
            else:
                bound += " livre"
        elif model["problem_objective"] == "max":
            if model["old_model"]["type_equality"][i] == ">=":
                bound += " <= 0"
            elif model["old_model"]["type_equality"][i] == "<=":
                bound += " >= 0"
            else:
                bound += " livre"
        bounds.append(bound)
    print(fo)
    print("st")
    for constr in constrs:
        print(constr)
    print("bounds")
    for bound in bounds:
        print(bound)


def select_base(var_indexes, size_base, model):
    base = random.sample(var_indexes, size_base)
    N = [x for x in var_indexes if x not in base]

    while 1:
        B = model["A"][:, base]
        B_inverse = inv(B)

        basic_sol = np.dot(B_inverse, model["b"])

        new_base = False
        for i in basic_sol:
            if i < 0:
                new_base = True
                break
        if not new_base:
            break
        base = random.sample(var_indexes, size_base)
        N = [x for x in var_indexes if x not in base]

    return base, N


def simplex(model):
    size_base = len(model["A"])
    var_indexes = [x for x in range(len(model["c"]))]

    base, N = select_base(var_indexes, size_base, model)
    basic_sol = 0
    iteration = 0
    pT = 0
    basic_sol = 0
    B_inverse = 0
    while 1:
        print(f" \n\n---------- ITERATION {iteration} ------------------- \n\n")
        print(f"Base = {base}")
        print(f"N = {N}\n")
        B = model["A"][:, base]
        print(f"B = {B}")
        B_inverse = inv(B)
        print(f"B' = {B_inverse}")
        print(f"b = {model['b']}")
        basic_sol = np.dot(B_inverse, model["b"])

        print(f"basic_sol = {basic_sol}")
        pT = np.dot(np.array([model["c"][x] for x in base]).transpose(), B_inverse)
        print(f"pT = {pT}")
        s_j = [
            (i, x, model["c"][x] - np.dot(pT, model["A"][:, x].reshape((-1, 1))))
            for i, x in enumerate(N)
        ]
        print(f"s_j = {s_j}")
        s_k = min(s_j, key=lambda t: t[2])
        print(f"s_k = {s_k}")
        if s_k[2] >= 0:
            break
        y = np.dot(B_inverse, model["A"][:, s_k[1]])
        print(f"y = {y}")
        pricing = []
        for i, val in enumerate(basic_sol):
            if y[i] == 0 or (val / y[i]) < 0:
                continue
            pricing.append((i, base[i], val / y[i]))
        if len(pricing) == 0:
            print("------ PROBLEMA ILIMITADO ----")
            break
        removed_variable = min(pricing, key=lambda t: t[2])
        print(f"removed_variable = {removed_variable}")
        base[removed_variable[0]] = s_k[1]
        N[s_k[0]] = removed_variable[1]
        iteration = iteration + 1
    custo_f = np.dot(model["c"][base], basic_sol)
    if model["problem_objective"] == "max":
        custo_f = custo_f * -1
        pT = pT * -1
    variables = ["x" + str(x + 1) + " = " + str(y) for (x, y) in zip(base, basic_sol)]

    print(" \n_____ RESULT _____\n")
    print(variables, f"f(x) = {custo_f}")
    print(f"Sol dual = ", pT)

    identity = np.identity(len(model["b"]))
    basic_sol = -1 * basic_sol

    for i, val in enumerate(model["b"]):
        b_star = np.dot(B_inverse, identity[:, i])
        b_star = basic_sol / b_star
        b_star_pos = [x for x in b_star if x >= 0]
        b_star_neg = [x for x in b_star if x < 0]
        b_star_pos = min(b_star_pos) if len(b_star_pos) > 0 else np.inf
        b_star_neg = max(b_star_neg) if len(b_star_neg) > 0 else -np.inf

        print(
            f"gamma_" + str(i + 1) + " |   " + str(b_star_pos) + " | " + str(b_star_neg)
        )


# path = "tests/instance.txt"
path = "tests/instance.txt"
model = parse_instance(path)
model_in_standard_form = model_to_standard_form(model)
simplex(model_in_standard_form)
print(to_dual(model_in_standard_form))

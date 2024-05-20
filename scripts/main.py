import numpy as np
from numpy.linalg import inv
import random
import sys
import copy

np.seterr(divide="ignore")


def parse_var(val):
    x_index = val.find("x")
    coef = val[:x_index]
    var = val[x_index:]
    return float(coef), var


def parse_lp(path):
    lines = []
    with open(path, "r") as file:
        lines = file.readlines()

    lines = [line.strip().split(sep=" ") for line in lines]
    model = {}
    model["problem_objective"] = "min" if lines[0][0] == "Minimize" else "max"
    model["c"] = []
    model["x_obj"] = []
    model["x_vars"] = []
    model["b"] = []
    model["type_equality"] = []
    model["A"] = []
    model["A_xvars"] = []
    model["x_bound"] = []
    model["b_bound"] = []
    model["ineq_bound"] = []

    last_pos = 1
    last_coef = 1
    count_line = 0
    previous = False
    for i, line in enumerate(lines[1:]):
        if line[0] == "Subject":
            break

        for val in line:
            if val.find(":") != -1:
                continue

            try:
                last_coef = float(val) * last_pos
                previous = False
            except ValueError:
                if val == "+":
                    last_pos = 1
                    previous = True
                elif val == "-":
                    last_pos = -1
                    previous = True
                else:
                    if previous:
                        last_coef = last_pos
                    model["c"].append(last_coef)
                    model["x_obj"].append(val)
                    model["x_vars"].append(val)
                    previous = False
        count_line = count_line + 1
    count_line = count_line + 2
    last_pos = 1
    last_coef = 1
    coefs = []
    costr_vars = []
    type_ineq = -1
    b = ""
    end = False
    for i, line in enumerate(lines[count_line:]):
        if (line[0].find(":") != -1 and i != 0) or line[0] in ["Bounds", "End"]:
            model["A"].append(coefs)
            model["A_xvars"].append(costr_vars)
            model["b"].append(b)
            model["type_equality"].append(type_ineq)
            coefs = []
            costr_vars = []
            type_ineq = -1
            b = ""
            last_pos = 1
            last_coef = 1

        if line[0] == "Bounds":
            count_line = count_line + i
            break
        elif line[0] == "End":
            end = True
            break

        previous = False
        for val in line:
            if val.find(":") != -1:
                continue
            try:
                last_coef = float(val) * last_pos
                previous = False
            except ValueError:
                if val == "+":
                    last_pos = 1
                    previous = True
                elif val == "-":
                    last_pos = -1
                    previous = True
                elif val in ["=", ">=", "<="]:
                    type_ineq = val
                    previous = False
                    break
                else:
                    if previous:
                        last_coef = last_pos
                    model["x_vars"].append(val)
                    coefs.append(last_coef)
                    costr_vars.append(val)
        if type_ineq != -1:
            b = float(line[-1])

    if not end:
        last_pos = 1
        last_coef = 1
        coefs = []
        costr_vars = []
        type_ineq = -1
        b = ""
        end = False
        print(len(lines), count_line)
        for i, line in enumerate(lines[count_line + 1 :]):
            if line[0] == "End":
                break

            if len(line) == 5:
                model["A"].append([1])
                model["A"].append([1])
                model["A_xvars"].append([line[2]])
                model["A_xvars"].append([line[2]])
                model["b"].append(float(line[0]))
                model["b"].append(float(line[4]))
                model["type_equality"].append(">=")
                model["type_equality"].append("<=")
                model["x_bound"].append(1)
                model["ineq_bound"].append(">=")
                model["b_bound"].append(0)

            elif len(line) == 3:
                try:
                    bound = float(line[2])
                    if bound != 0:
                        model["A"].append([1])
                        model["A_xvars"].append([line[0]])
                        model["b"].append(bound)
                        model["type_equality"].append(line[1])
                except ValueError:
                    model["A"].append([1])
                    model["A_xvars"].append([line[2]])
                    model["b"].append(float(line[0]))
                    model["type_equality"].append(line[1])

                model["x_bound"].append(line[0])
                model["ineq_bound"].append(line[1])
                model["b_bound"].append(0)
            else:
                model["x_bound"].append(line[0])
                model["ineq_bound"].append("livre")
                model["b_bound"].append("")

    model["x_vars"] = {x: i for i, x in enumerate(list(dict.fromkeys(model["x_vars"])))}

    for key, val in model["x_vars"].items():
        if key not in model["x_bound"]:
            model["x_bound"].append(key)
            model["ineq_bound"].append(">=")
            model["b_bound"].append(0)
    model["count_constr"] = len(model["A"])
    model["count_vars"] = len(model["x_bound"])
    print(model)
    return model


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
    model["x_vars"] = {x: i for i, x in enumerate(model["x_vars"])}
    model["count_constr"] = len(model["A"])
    model["count_vars"] = len(model["x_bound"])
    return model


def model_to_standard_form(model):
    great_index = len(model["x_vars"])
    aux_model = copy.deepcopy(model)

    for i, val in enumerate(model["b"]):
        if val >= 0:
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

        model["A_xvars"][i].append("x_folga" + str(great_index))
        model["x_vars"]["x_folga" + str(great_index)] = great_index
        great_index = great_index + 1

    for i, constr in enumerate(model["type_equality"]):
        model["A"][i].append(1)

        model["A_xvars"][i].append("x_artif" + str(great_index))
        model["x_vars"]["x_artif" + str(great_index)] = great_index
        great_index = great_index + 1

    new_model = {}
    new_model["c"] = np.array([np.double(0) for (key, val) in model["x_vars"].items()])
    new_model["problem_objective"] = model["problem_objective"]
    for i, val in enumerate(model["x_obj"]):
        new_model["c"][model["x_vars"][val]] = np.double(model["c"][i])
    if model["problem_objective"] == "max":
        new_model["c"] = np.double(new_model["c"] * -1)

    # for key, val in model["x_vars"].items():
    #     if "artif" in key:
    #         new_model["c"][val] = 0

    new_model["A"] = np.array(
        [
            [np.double(0.0) for x in range(len(model["x_vars"]))]
            for y in range(len(model["A"]))
        ]
    )
    for i, constr in enumerate(model["A_xvars"]):
        for j, val in enumerate(constr):
            new_model["A"][i][model["x_vars"][val]] = np.double(model["A"][i][j])
    print(new_model["A"])
    new_model["b"] = np.array(model["b"])
    new_model["old_model"] = model
    new_model["aux_model"] = aux_model

    return new_model


def to_dual(model):
    aux_p_vars = ["p" + str(i + 1) for i in range(len(model["aux_model"]["A"]))]
    fo = "max " if model["aux_model"]["problem_objective"] == "min" else "min "
    for i, val in enumerate(model["aux_model"]["A"]):
        if i == 0:
            fo += str(model["aux_model"]["b"][i]) + aux_p_vars[i] + " "
        else:
            symb = "+" if model["aux_model"]["b"][i] >= 0 else ""
            fo += symb + str(model["aux_model"]["b"][i]) + aux_p_vars[i] + " "

    constr_count = model["aux_model"]["count_constr"]
    var_count = model["aux_model"]["count_vars"]
    At = model["A"][:constr_count, :var_count].transpose()
    constrs = []
    bounds = []

    for i, row in enumerate(At):
        constr = ""

        c = model["c"][i] * -1 if model["problem_objective"] == "max" else model["c"][i]
        for j, col in enumerate(row):
            if j == 0:
                constr += str(col) + aux_p_vars[j] + " "
            else:
                symb = "+" if col >= 0 else ""
                constr += symb + str(col) + aux_p_vars[j] + " "

        if model["problem_objective"] == "min":
            if model["aux_model"]["ineq_bound"][i] == ">=":
                constr += " <= " + str(c)
            elif model["aux_model"]["ineq_bound"][i] == "<=":
                constr += " >= " + str(c)
            else:
                constr += " = " + str(c)
        elif model["problem_objective"] == "max":
            if model["aux_model"]["ineq_bound"][i] == ">=":
                constr += " >= " + str(c)
            elif model["aux_model"]["ineq_bound"][i] == "<=":
                constr += " <= " + str(c)
            else:
                constr += " = " + str(c)
        constrs.append(constr)

    for i in range(model["aux_model"]["count_constr"]):
        bound = "p" + str(i + 1)
        if model["problem_objective"] == "min":
            if model["aux_model"]["type_equality"][i] == ">=":
                bound += " >= 0"
            elif model["aux_model"]["type_equality"][i] == "<=":
                bound += " <= 0"
            else:
                bound += " livre"
        elif model["problem_objective"] == "max":
            if model["aux_model"]["type_equality"][i] == ">=":
                bound += " <= 0"
            elif model["aux_model"]["type_equality"][i] == "<=":
                bound += " >= 0"
            else:
                bound += " livre"
        bounds.append(bound)
    with open("result_dual.txt", "w") as f:
        f.write(f"{fo}\n")
        f.write("st\n")
        for constr in constrs:
            f.write(f"{constr}\n")
        f.write("bounds\n")
        for bound in bounds:
            f.write(f"{bound}\n")


def select_base(var_indexes, size_base, model):
    base = random.sample(var_indexes, size_base)
    N = [x for x in var_indexes if x not in base]

    while 1:
        B = model["A"][:, base]
        try:
            B_inverse = inv(B)
        except np.linalg.LinAlgError:
            continue

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


def simplex(model, base, c):
    size_base = len(model["A"])
    var_indexes = [x for x in range(len(model["c"]))]

    N = [x for x in var_indexes if x not in base]
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
        print(f"B = {B}\n")
        B_inverse = inv(B)
        # print(f"B' = {B_inverse}\n")
        # print(f"b = {model['b']}\n")
        basic_sol = np.dot(B_inverse, model["b"])

        # print(f"basic_sol = {basic_sol}\n")
        pT = np.dot(c[base].transpose(), B_inverse)
        # print(f"pT = {pT}\n")

        s_j = [
            (i, x, c[x] - np.dot(pT, model["A"][:, x].reshape((-1, 1))))
            for i, x in enumerate(N)
        ]
        print(f"s_j_real = {sorted(s_j, key= lambda x: x[2] )}\n")
        aux = sorted([x for x in s_j if x[2] < -0.00001], key=lambda x: x[1])
        print(f"s_j = {[aux]}")
        s_k = [x for x in s_j if x[2] < -0.00001]
        if len(s_k) == 0:
            break

        s_k = min(s_k, key=lambda x: x[1])
        y = np.dot(B_inverse, model["A"][:, s_k[1]].reshape((-1, 1)))
        print(f"y = {y}\n")
        pricing = []
        for i, val in enumerate(basic_sol):
            if y[i] < 0.00000001:
                continue
            print(y[i])
            cal = val / y[i]
            if abs(cal) <= 0.00001:
                cal = 0
            pricing.append((i, base[i], cal))

        if len(pricing) == 0:
            print("------ PROBLEMA ILIMITADO ----")
            break

        removed_variable = sorted(
            pricing,
            key=lambda t: (t[2], t[1]),
        )
        print("pricing = ", removed_variable)
        removed_variable = removed_variable[0]
        print(f"\ninserted_variable = {s_k}\n")
        print(f"removed_variable = {removed_variable}")
        base[removed_variable[0]] = s_k[1]
        N.remove(s_k[1])
        N.append(removed_variable[1])
        iteration = iteration + 1

    return base, basic_sol, pT, B_inverse


# path = "tests/instance.txt"

instance_file_path = sys.argv[1]
model = parse_lp(instance_file_path)
# model = parse_instance(instance_file_path)
model_in_standard_form = model_to_standard_form(model)
with open("data.txt", "w") as f:
    f.write(f"{model_in_standard_form}")
base = [
    x
    for x in range(
        len(model_in_standard_form["c"]) - len(model_in_standard_form["A"]),
        len(model_in_standard_form["c"]),
    )
]
c = np.array(
    [
        1 if "artif" in key else 0
        for key, val in model_in_standard_form["old_model"]["x_vars"].items()
    ]
)
artif_base = [
    val
    for key, val in model_in_standard_form["old_model"]["x_vars"].items()
    if "artif" in key
]
base, basic_sol, pT, B_inverse = simplex(model_in_standard_form, base, c)
artif = [x for x in base if x in artif_base]
res = np.dot(B_inverse, model_in_standard_form["A"])
print(res)
print(res[:, artif])
# print(c[base])
# print(basic_sol)
# custo_f = np.dot(c[base], basic_sol)
# variables = {
#     val: key for (key, val) in model_in_standard_form["old_model"]["x_vars"].items()
# }
# variables = [
#     variables[x] + " * " + str(c[x]) + " = " + str(y) for (x, y) in zip(base, basic_sol)
# ]
# for i in variables:
#     print(i)
# print(custo_f)
model_in_standard_form["A"][:, artif_base] = 0
(base, basic_sol, pT, B_inverse) = simplex(
    model_in_standard_form, base, model_in_standard_form["c"]
)

custo_f = np.dot(model_in_standard_form["c"][base], basic_sol)
if model_in_standard_form["problem_objective"] == "max":
    custo_f = custo_f * -1
    pT = pT * -1
variables = {
    val: key for (key, val) in model_in_standard_form["old_model"]["x_vars"].items()
}
variables = [variables[x] + " = " + str(y) for (x, y) in zip(base, basic_sol)]
with open("result.txt", "w") as f:
    f.write(" \n_____ RESULT _____\n")
    f.write(f"Vars = {variables}\nf(x) = {custo_f}\n")
    f.write(f"Sol dual = {pT}\n")

    identity = np.identity(len(model_in_standard_form["b"]))
    basic_sol = -1 * basic_sol

    for i, val in enumerate(model_in_standard_form["b"]):
        b_star = np.dot(B_inverse, identity[:, i])
        b_star = basic_sol / b_star
        b_star_pos = [x for x in b_star if x >= 0]
        b_star_neg = [x for x in b_star if x < 0]
        b_star_pos = min(b_star_pos) if len(b_star_pos) > 0 else np.inf
        b_star_neg = max(b_star_neg) if len(b_star_neg) > 0 else -np.inf

        f.write(
            f"gamma_"
            + str(i + 1)
            + " |   "
            + str(b_star_pos)
            + " | "
            + str(b_star_neg)
            + "\n"
        )
    f.close()

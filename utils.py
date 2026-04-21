import os
import sys
import argparse
import pathlib
import numpy as np
import random
import pyscipopt as scp
import torch
import torch.nn as nn
import pickle

import gurobipy as gp
from gurobipy import GRB


device=torch.device("cpu")

TASKS = ["IP","WA","CA","SC"]

def get_a_new2(ins_name):
    epsilon = 1e-6

    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)


    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            # values_spr.append(v)
            values_spr.append(1)
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        # A[cind][-2]=rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            # A[cind][v_indx]=1
            # A[cind][-1]+=1
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars)).to(device)
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    # v_nodes=position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def extract_raw_ilp(ins_name):
    """
    Extract raw ILP data from .lp/.mps file for unsupervised ALM loss computation.

    Returns a dict with:
        obj_coeffs:    [n_vars] raw c_i (sign-adjusted for minimization)
        cons_indices:  [2, n_cons_edges] sparse (constraint_idx, variable_idx)
        cons_values:   [n_cons_edges] raw A_ji values
        rhs:           [n_cons_leq] RHS for all <= constraints (after conversion)
        n_cons:        number of constraints (after sense conversion)
        n_vars:        number of variables
        b_vars_mask:   [n_vars] bool mask for binary variables
        var_names:     list of variable names (sorted)
        obj_sense_min: True if original is minimization
    """
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)
    nvars = len(mvars)

    v_map_raw = {}
    b_vars_mask = torch.zeros(nvars, dtype=torch.bool)
    var_names_raw = []
    for indx, v in enumerate(mvars):
        v_map_raw[v.name] = indx
        var_names_raw.append(v.name)
        if v.vtype() == 'BINARY':
            b_vars_mask[indx] = True

    # Objective sense: convert to minimization
    obj_sense = m.getObjectiveSense()
    obj_sense_min = (obj_sense == 'minimize')
    sign = 1.0 if obj_sense_min else -1.0

    obj = m.getObjective()
    obj_coeffs = torch.zeros(nvars, dtype=torch.float32)
    for e in obj:
        vnm = e.vartuple[0].name
        v_indx = v_map_raw[vnm]
        obj_coeffs[v_indx] = sign * obj[e]

    # Constraints: convert all to <= form
    cons = m.getConss()
    # Filter out empty constraints
    cons = [c for c in cons if len(m.getValsLinear(c)) > 0]

    cons_row = []
    cons_col = []
    cons_val = []
    rhs_list = []
    cons_count = 0

    for c in cons:
        coeff = m.getValsLinear(c)
        rhs_val = m.getRhs(c)
        lhs_val = m.getLhs(c)

        if rhs_val == lhs_val:
            # Equality: split into A_j x <= b_j AND -A_j x <= -b_j
            for k in coeff:
                v_indx = v_map_raw[k]
                cons_row.append(cons_count)
                cons_col.append(v_indx)
                cons_val.append(coeff[k])
            rhs_list.append(rhs_val)
            cons_count += 1

            for k in coeff:
                v_indx = v_map_raw[k]
                cons_row.append(cons_count)
                cons_col.append(v_indx)
                cons_val.append(-coeff[k])
            rhs_list.append(-rhs_val)
            cons_count += 1
        elif rhs_val >= 1e+20:
            # >= constraint: -A_j x <= -lhs
            for k in coeff:
                v_indx = v_map_raw[k]
                cons_row.append(cons_count)
                cons_col.append(v_indx)
                cons_val.append(-coeff[k])
            rhs_list.append(-lhs_val)
            cons_count += 1
        else:
            # <= constraint: A_j x <= rhs
            for k in coeff:
                v_indx = v_map_raw[k]
                cons_row.append(cons_count)
                cons_col.append(v_indx)
                cons_val.append(coeff[k])
            rhs_list.append(rhs_val)
            cons_count += 1

    cons_indices = torch.tensor([cons_row, cons_col], dtype=torch.long)
    cons_values = torch.tensor(cons_val, dtype=torch.float32)
    rhs = torch.tensor(rhs_list, dtype=torch.float32)

    return {
        'obj_coeffs': obj_coeffs,
        'cons_indices': cons_indices,
        'cons_values': cons_values,
        'rhs': rhs,
        'n_cons': cons_count,
        'n_vars': nvars,
        'b_vars_mask': b_vars_mask,
        'var_names': var_names_raw,
        'obj_sense_min': obj_sense_min,
    }
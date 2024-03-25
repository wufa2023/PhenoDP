import json

def is_connected(ref_hpo_list, query_hpo):
    parent_nodes = []
    parent_nodes_second = []
    with open('../source_data/hpo_parents.json', 'r') as f:
        data = json.load(f)
    vocab = list(data.keys())


    for term in ref_hpo_list:
        if term in vocab:
            # 当前节点有父亲节点
            parent_nodes.extend(data[term])
        else:
            # 当前节点无父亲节点
            pass
    # 第二次查询
    for term in parent_nodes:
        if term in vocab:
            parent_nodes_second.extend(data[term])

    parent_query = []
    parent_query_sencond = []

    for term in query_hpo:
        if term in vocab:
            parent_query.extend(data[term])


    for term in parent_query:
        if term in vocab:
            parent_query_sencond.extend(data[term])

    for term in parent_query_sencond:
        if term in parent_nodes_second:
            return True
    return False

def is_simulation(ref_hpo_list, specific_hpo):
    _specific = get_specific(ref_hpo_list, specific_hpo)
    if len(ref_hpo_list) - len(_specific) >= 2 and len(ref_hpo_list) >= 3:
        return True
    else:
        return False

import random
import numpy as np
def random_remove(ref_hpo_list, specific_hpo, if_test=False):
    choice = range(10, 50, 1)
    _per = random.choice(choice) / 100
    _spe = get_specific(ref_hpo_list=ref_hpo_list, specific_hpo=specific_hpo)
    _normal_term = set(ref_hpo_list) - set(_spe)
    # if none normal terms left, return raw sentence
    if len(_normal_term) == 0:
        print('none normal word left, return raw sentences')
        return ref_hpo_list

    _rm_len = np.round(_per * len(_normal_term), 0)

    print('total words', len(ref_hpo_list), 'specific words', len(_spe),
          'normal words', len(_normal_term), 'remove len', _rm_len)

    if if_test:
        print(int(_rm_len))
        print(list(_normal_term))
    for i in range(int(_rm_len)):
        _rm = random.choice(list(_normal_term))
        _normal_term.remove(_rm)
    if if_test:
        print(_spe)
    _sim_hpo = list(_normal_term) + _spe
    if if_test:
        print(_sim_hpo)
    return list(_sim_hpo)


def get_specific(ref_hpo_list, specific_hpo):
    specific_hpo_set = set(specific_hpo)
    return [term for term in ref_hpo_list if term in specific_hpo_set]

def remove_duplicate(save_omim_id, save_sim_hpo):
    unique_omim_id = []
    unique_sim_hpo = []
    seen = list()  # 用于记录已经出现过的元素
    for i in range(len(save_sim_hpo)):
        # 构建一个唯一标识符
        identifier = (save_omim_id[i], save_sim_hpo[i])
        # 如果该标识符已经出现过，则跳过
        if identifier in seen:
            continue
        # 否则将其添加到结果列表中，并记录该标识符
        unique_omim_id.append(save_omim_id[i])
        unique_sim_hpo.append(save_sim_hpo[i])
        seen.append(identifier)
    return unique_omim_id, unique_sim_hpo

def get_change(ref, specific_list, relation):
    new_ref = []
    old_ref = []

    choices = random.choice(range(10, 50, 1)) / 100
    change_len = np.round(len(ref) * choices, 0)


    _specific = get_specific(ref, specific_list)

    print('raw len', len(ref), 'specific len', len(_specific), 'change len', change_len)

    for term in _specific:
        ref.remove(term)
    relation_choice = random.choice([1, 2])


    if len(ref) == 0:
        print('specific only')
        return -1

    count = 0
    while len(old_ref) < change_len:

        # add condition to prevent dead cycle
        if count == 100:
            print('error', ref)

            break
        count += 1

        # select one term from references, and it will be changed to another similar term
        selected_term = random.choice(ref)

        # now, we search relation space
        if selected_term in (relation.keys()):

            # get the first relation space
            parent_level_1 = relation[selected_term]

            # if not relation term left,
            # we will not consider this term, and continue to select another one
            if len(parent_level_1) == 0:
                continue

            # if term has relation terms, we will change it

            # if model choice to change term from space 1, we will end search after first round
            if relation_choice == 1:
                old_ref.append(selected_term)
                new_ref.append(random.choice(parent_level_1))

            # if search two level space
            elif relation_choice == 2:
                parent_level_2 = []
                for term in parent_level_1:
                    if term in list(relation.keys()):
                        parent_level_2.extend(relation[term])

                if len(parent_level_2) != 0:
                    old_ref.append(selected_term)
                    new_ref.append(random.choice(parent_level_2))
                else:
                    continue

    print('select old terms', old_ref)
    print('new terms', new_ref)
    # after we finish selecting new terms, we will replace old terms.
    for term in np.unique(old_ref):
        ref.remove(term)
    ref.extend(new_ref)

    print('______________________')
    return ref
import collections
import random
import copy
from tqdm import tqdm


def get_random_str(choice_list):
    return random.sample(choice_list, 1)[0]


def da_operations(org_samples):
    type_value_dict = collections.defaultdict(set)
    for sample in org_samples:
        types = sample['type']
        values = sample['value']
        n = len(types)
        for i in range(n):
            type_value_dict[types[i]].add(values[i])

    augmented_samples = []
    for sample in tqdm(org_samples):

        l = len(sample['type'])
        alpha = 0.15
        beta = 0.10
        n = int(alpha * l)
        # m = int(beta * l)
        m = n

        sample['sample_label'] = 0
        sample['token_label'] = [0] * l

        # type replacement
        tr_index = random.sample(range(0, l - 1), n)
        tr_sample = copy.deepcopy(sample)
        tr_sample['sample_label'] = 1
        tr_sample['token_label'] = [1 if i in tr_index else 0 for i in range(l)]  # 1
        for idx in tr_index:
            replace_type = get_random_str(list(type_value_dict.keys() - tr_sample['type'][idx]))
            tr_sample['type'][idx] = replace_type

        # value replacement
        vr_index = random.sample(range(0, l - 1), n)
        vr_sample = copy.deepcopy(sample)
        vr_sample['sample_label'] = 0  # label
        vr_sample['token_label'] = [2 if i in vr_index else 0 for i in range(l)]  # 2
        for idx in vr_index:
            cur_type, cur_value = vr_sample['type'][idx], vr_sample['value'][idx]
            choice = type_value_dict[cur_type]
            choice.discard(cur_value)
            if len(choice) == 0:
                replace_value = 'default'
            else:
                replace_value = get_random_str(choice)
            vr_sample['value'][idx] = replace_value

        # pairs Insertion
        random_index = random.sample(range(0, l - 1), m)
        pi_sample = copy.deepcopy(sample)
        pi_sample['sample_label'] = 1  # label
        chosen_types = [pi_sample['type'][i] for i in random_index]
        chosen_values = [pi_sample['value'][i] for i in random_index]
        pi_index = random.sample(range(0, l - 1), m)

        i = 0
        for idx in pi_index:
            pi_sample['type'] = pi_sample['type'][:idx] + [chosen_types[i]] + pi_sample['type'][idx:]
            pi_sample['value'] = pi_sample['value'][:idx] + [chosen_values[i]] + pi_sample['value'][idx:]
            pi_sample['token_label'] = pi_sample['token_label'][:idx] + [3] + pi_sample['token_label'][idx:]  # 3

        augmented_samples.append(sample)
        augmented_samples.append(tr_sample)
        augmented_samples.append(vr_sample)
        augmented_samples.append(pi_sample)

    assert len(augmented_samples) == 4 * len(org_samples)
    # print(len(augmented_samples), len(org_samples))
    # del org_samples
    return augmented_samples

import os
import random

import pandas as pd
import numpy as np


HOME = os.path.expanduser('~')


def calculate_tss(target, output):
    # Confusion Matrix 요소 계산
    tp = sum(1 for t, o in zip(target, output) if t == 1 and o == 1)  # True Positive
    tn = sum(1 for t, o in zip(target, output) if t == 0 and o == 0)  # True Negative
    fp = sum(1 for t, o in zip(target, output) if t == 0 and o == 1)  # False Positive
    fn = sum(1 for t, o in zip(target, output) if t == 1 and o == 0)  # False Negative
    # TPR (True Positive Rate, Sensitivity)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    # FPR (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0    
    # TSS = TPR - FPR
    tss = tpr - fpr
    return tss


def read_csv(file_path):
      df = pd.read_csv(file_path)
      target = df["target"].tolist()
      output = df["output"].tolist()
      pairs = list(zip(target, output))
      return pairs


def flatten(pairs):
    target = []
    output = []
    for pair in pairs :
        t, o = pair
        target.append(t)
        output.append(o)
    return target, output


def undersample(pairs):

    tss_list = []

    negative_pairs = []
    positive_pairs = []

    for pair in pairs :
        target, output = pair
        if target == 0 :
            negative_pairs.append(pair)
        else :
            positive_pairs.append(pair)

    random.shuffle(negative_pairs)
    base_size = len(positive_pairs)
    num_sublist = len(negative_pairs) // base_size
    sublists = []

    start = 0

    for i in range(num_sublist):
        sublists.append(negative_pairs[start:start + base_size])
        start += base_size

    for subset in sublists :
        subpairs = positive_pairs.copy() + subset.copy()
        target, output = flatten(subpairs)
        tss = calculate_tss(target, output)
        tss_list.append(tss)
    
    return tss_list


input_days = range(1, 8)
output_day = 1
epochs = range(100, 1001, 100)
weighted = (False, True)
subsets = range(10)

result_root = f"{HOME}/ap/results"
file_name = "validation_results.csv"

for apply_weighted in weighted :
    for input_day in input_days :
        for epoch in epochs :
            for subset in subsets :
                if apply_weighted is True :
                    experiment_name = f"weighted_days{input_day}_to_day{output_day}_sub_{subset}"
                else :
                    experiment_name = f"days{input_day}_to_day{output_day}_sub_{subset}"

                dir_path = f"{result_root}/{experiment_name}/output/epoch_{epoch}"
                file_path = f"{dir_path}/{file_name}"

                if os.path.exists(file_path):
                    pairs = read_csv(file_path)
                    target, output = flatten(pairs)
                    tss = calculate_tss(target, output)
                    subtss_list = undersample(pairs)

                    print(f"{experiment_name} Total TSS:{tss:.3f}, Sub Avg TSS:{np.mean(subtss_list):.3f}")                




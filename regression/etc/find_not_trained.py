import os

RESULT_ROOT = "/home/hl545/ap/results"

PREFIX = ("reg", "reg_consistency")#, "reg_contrastive")
NUM_SUBSAMPLE_UNDER = 10
NUM_SUBSAMPLE_MIXED = 3

input_days = (1, 2, 3, 4, 5, 6, 7)
output_day = 1

task_list = []

for prefix in PREFIX:

    # Original
    for input_day in input_days :
        dir_path = f"{RESULT_ROOT}/{prefix}_{input_day}_to_{output_day}"
        dir_name = os.path.basename(dir_path)
        model_path = f"{dir_path}/checkpoint/model_final.pth"
        if not os.path.exists(model_path):
            print(f"{prefix}\toriginal\t{input_day}\t{output_day}")
            task_list.append((prefix, "original", input_day, output_day))

    # Oversampling
    for input_day in input_days :
        dir_path = f"{RESULT_ROOT}/{prefix}_{input_day}_to_{output_day}_over"
        dir_name = os.path.basename(dir_path)   
        model_path = f"{dir_path}/checkpoint/model_final.pth"
        if not os.path.exists(model_path):
            print(f"{prefix}\tover\t{input_day}\t{output_day}")
            task_list.append((prefix, "over", input_day, output_day))

    # Undersampling
    for input_day in input_days :
        for subsample_index in range(NUM_SUBSAMPLE_UNDER):
            dir_path = f"{RESULT_ROOT}/{prefix}_{input_day}_to_{output_day}_sub_{subsample_index:02d}"
            dir_name = os.path.basename(dir_path)
            model_path = f"{dir_path}/checkpoint/model_final.pth"
            if not os.path.exists(model_path):
                print(f"{prefix}\tunder\t{input_day}\t{output_day}\t{subsample_index}")
                task_list.append((prefix, "under", input_day, output_day, subsample_index))

    # Mixed
    for input_day in input_days :
        for subsample_index in range(NUM_SUBSAMPLE_MIXED):
            dir_path = f"{RESULT_ROOT}/{prefix}_{input_day}_to_{output_day}_mix_sub_{subsample_index:02d}"
            dir_name = os.path.basename(dir_path)
            model_path = f"{dir_path}/checkpoint/model_final.pth"
            if not os.path.exists(model_path):
                print(f"{prefix}\tmixed\t{input_day}\t{output_day}\t{subsample_index}")
                task_list.append((prefix, "mixed", input_day, output_day, subsample_index))

f = open("not_trained_tasks.txt", "w")
for task in task_list:
    f.write("\t".join([str(x) for x in task]) + "\n")
f.close()
print(f"\nTotal not trained tasks: {len(task_list)}")
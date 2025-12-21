import os

RESULT_ROOT = "/home/hl545/ap/renew/results"
PREFIX = ["REG_MSE"]
NUM_SUBSAMPLE_UNDER = 14
input_days = [1, 2, 3, 4, 5, 6, 7]
output_days = [1, 2, 3]


task_list = []

for prefix in PREFIX:

    for input_day in input_days :
        for output_day in output_days :
            for subsample_index in range(NUM_SUBSAMPLE_UNDER):
                dir_path = f"{RESULT_ROOT}/{prefix}_DATE-{input_day:02d}-TO-{output_day:02d}_UNDER-{subsample_index:02d}"
                dir_name = os.path.basename(dir_path)
                model_path = f"{dir_path}/checkpoint/model_final.pth"
                if not os.path.exists(model_path):
                    print(dir_path)
                    print(f"{prefix}\tunder\t{input_day}\t{output_day}\t{subsample_index}")
                    task_list.append((prefix, "under", input_day, output_day, subsample_index))

f = open("not_trained_tasks.txt", "w")
for task in task_list:
    f.write("\t".join([str(x) for x in task]) + "\n")
f.close()
print(f"\nTotal not trained tasks: {len(task_list)}")
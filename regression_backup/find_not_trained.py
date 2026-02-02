import os

# /home/hl545/ap/renew/results/SINGLE_1_1_00/checkpoint/model_final.pth

RESULT_ROOT = "/home/hl545/ap/renew/results"

def find_single():
    task_list = []
    PREFIX = "SINGLE"
    for I in range(7):
        for T in range(3):
            for S in range(14):
                dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_{T+1}_{S:02d}"
                check_path = f"{dir_path}/checkpoint/model_final.pth"
                if not os.path.exists(check_path):
                    print(f"{dir_path}: not trained")
                    task_list.append((I, T, S))
    f = open("not_trained_single.txt", "w")
    for task in task_list:
        f.write("\t".join([str(x) for x in task]) + "\n")
    f.close()
    return task_list

def find_multi12():
    task_list = []
    PREFIX = "MULTITARGET"
    for I in range(7):
        for S in range(8):
            dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_1-2_{S:02d}"
            check_path = f"{dir_path}/checkpoint/model_final.pth"
            if not os.path.exists(check_path):
                print(f"{dir_path}: not trained")
                task_list.append((I, S))
    f = open("not_trained_multi12.txt", "w")
    for task in task_list:
        f.write("\t".join([str(x) for x in task]) + "\n")
    f.close()
    return task_list

def find_multi13():
    task_list = []
    PREFIX = "MULTITARGET"
    for I in range(7):
        for S in range(6):
            dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_1-3_{S:02d}"
            check_path = f"{dir_path}/checkpoint/model_final.pth"
            if not os.path.exists(check_path):
                print(f"{dir_path}: not trained")
                # task_list.append(dir_path)
                task_list.append((I, S))
    f = open("not_trained_multi13.txt", "w")
    for task in task_list:
        f.write("\t".join([str(x) for x in task]) + "\n")
    f.close()

    return task_list

if __name__ == "__main__":
    task_list = []
    task_list += find_single()
    task_list += find_multi12()
    task_list += find_multi13()
    print(len(task_list))

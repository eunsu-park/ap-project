import os

# /home/hl545/ap/renew/results/SINGLE_1_1_00/checkpoint/model_final.pth

RESULT_ROOT = "/home/hl545/ap/renew/results"

def find_single():
    task_list = []
    PREFIX = "SINGLE"
    for I in range(7):
        for T in range(3):
            for S in range(14):
                for epoch in range(20, 101, 20):
                    dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_{T+1}_{S:02d}/mcd/epoch_{epoch:04d}"
                    check_path = f"{dir_path}/2024122300.h5.npz"
                    if not os.path.exists(check_path):
                        print(f"{dir_path}: not mcded")
                        task_list.append((I, T, S, epoch))
    f = open("not_mcded_single.txt", "w")
    for task in task_list:
        f.write("\t".join([str(x) for x in task]) + "\n")
    f.close()
    return task_list

def find_multi12():
    task_list = []
    PREFIX = "MULTITARGET"
    for I in range(7):
        for S in range(8):
            for epoch in range(20, 101, 20):
                dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_1-2_{S:02d}/mcd/epoch_{epoch:04d}"
                check_path = f"{dir_path}/2024122300.h5.npz"
                if not os.path.exists(check_path):
                    print(f"{dir_path}: not mcded")
                    task_list.append((I, S, epoch))
    f = open("not_mcded_multi12.txt", "w")
    for task in task_list:
        f.write("\t".join([str(x) for x in task]) + "\n")
    f.close()
    return task_list

def find_multi13():
    task_list = []
    PREFIX = "MULTITARGET"
    for I in range(7):
        for S in range(6):
            for epoch in range(20, 101, 20):
                dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_1-3_{S:02d}/mcd/epoch_{epoch:04d}"
                check_path = f"{dir_path}/2024122300.h5.npz"
                if not os.path.exists(check_path):
                    print(f"{dir_path}: not mcded")
                    # task_list.append(dir_path)
                    task_list.append((I, S, epoch))
    f = open("not_mcded_multi13.txt", "w")
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

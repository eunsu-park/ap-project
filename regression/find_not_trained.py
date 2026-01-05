import os

# /home/hl545/ap/renew/results/SINGLE_1_1_00/checkpoint/model_final.pth

RESULT_ROOT = "/home/hl545/ap/renew/results"

def find_single():
    result = []
    PREFIX = "SINGLE"
    for I in range(7):
        for T in range(3):
            for S in range(14):
                dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_{T+1}_{S:02d}"
                check_path = f"{dir_path}/checkpoint/model_final.pth"
                if not os.path.exists(check_path):
                    print(f"{dir_path}: not trained")
                    result.append(dir_path)
    return result

def find_multi_12():
    result = []
    PREFIX = "MULTITARGET"
    for I in range(7):
        for S in range(8):
            dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_1-2_{S:02d}"
            check_path = f"{dir_path}/checkpoint/model_final.pth"
            if not os.path.exists(check_path):
                print(f"{dir_path}: not trained")
                result.append(dir_path)
    return result

def find_multi_13():
    result = []
    PREFIX = "MULTITARGET"
    for I in range(7):
        for S in range(6):
            dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_1-3_{S:02d}"
            check_path = f"{dir_path}/checkpoint/model_final.pth"
            if not os.path.exists(check_path):
                print(f"{dir_path}: not trained")
                result.append(dir_path)
    return result

if __name__ == "__main__":
    result = []
    result += find_single()
    result += find_multi_12()
    result += find_multi_13()
    print(len(result))
import os

# /home/hl545/ap/renew/results/SINGLE_1_1_00/checkpoint/model_final.pth

RESULT_ROOT = "/home/hl545/ap/renew/results"

PREFIX = "SINGLE"

for I in range(7):
    for T in range(3):
        for S in range(14):
            dir_path = f"{RESULT_ROOT}/{PREFIX}_{I+1}_{T+1}_{S:02d}"
            check_path = f"{dir_path}/checkpoint/model_final.pth"

            if not os.path.exists(check_path):
                print(f"{dir_path}: not trained")

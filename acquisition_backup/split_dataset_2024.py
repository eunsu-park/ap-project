import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ====== 경로 설정 ======
INPUT_PATH = r"/opt/projects/ap/datasets/three_twelve_to_one.csv"
TRAIN_PATH = r"/opt/projects/ap/datasets/three_twelve_to_one_train.csv"
VAL_PATH   = r"/opt/projects/ap/datasets/three_twelve_to_one_validation.csv"

# ====== 1. Train/Validation 분리 기준 ======
TRAIN_YEARS = range(2010, 2024)  # 2010-2023
VAL_YEARS   = [2024]             # 2024

def is_validation_row(file_name: str) -> bool:
    """파일명에서 연도 추출 후 validation 여부 판단"""
    year = int(file_name[0:4])
    return year in VAL_YEARS

# ====== 2. 데이터 로드 및 보조 컬럼 추가 ======
df = pd.read_csv(INPUT_PATH)
if "file_name" not in df.columns:
    raise ValueError("CSV에 'file_name' 컬럼이 없습니다.")

# 원본 컬럼 순서를 기억해 두었다가, 저장할 때 그대로 사용
original_cols = df.columns.tolist()

# 분석용 보조 컬럼
df["year"]  = df["file_name"].str[0:4].astype(int)
df["month"] = df["file_name"].str[4:6].astype(int)
df["split"] = df["file_name"].apply(
    lambda x: "validation" if is_validation_row(x) else "train"
)

# ====== 3. Train / Validation 분리 및 저장 (원본 형식 그대로) ======
df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val   = df[df["split"] == "validation"].reset_index(drop=True)

# 저장할 때는 year/month/split 제거하고 원본 컬럼만
os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
df_train[original_cols].to_csv(TRAIN_PATH, index=False)
df_val[original_cols].to_csv(VAL_PATH, index=False)

# ====== 4. 연·월별 통계 (print만) ======
print("=== Year-Month Distribution (전체) ===")
year_month_count = df.groupby(["year", "month"]).size().unstack(fill_value=0)
print(year_month_count)

print("\n=== Year-Month Distribution (Train / Validation) ===")
split_stats = df.groupby(["year", "split"]).size().unstack(fill_value=0)
print(split_stats)

# ====== 5. Heatmap (Year–Month + validation 박스) ======
plt.figure(figsize=(12, 6))

data = year_month_count.values
n_years, n_months = data.shape

# heatmap
plt.imshow(data, aspect='auto', origin='lower')
plt.colorbar(label='Number of data samples')

plt.title("Year–Month Distribution of Data Samples")
plt.xlabel("Month")
plt.ylabel("Year")

months = np.arange(1, 13)
years  = year_month_count.index.values

# tick 위치: 셀 중심 (0~11, 0~N-1)
plt.xticks(np.arange(n_months), [f"{m:02d}" for m in months])
plt.yticks(np.arange(n_years), years)

# 축 범위 (셀 경계에 맞게)
plt.xlim(-0.5, n_months - 0.5)
plt.ylim(-0.5, n_years - 0.5)

# validation year 박스: 2024년 전체 행에 박스 표시
ax = plt.gca()
for row_idx, year in enumerate(years):
    if year in VAL_YEARS:
        # 해당 연도의 모든 월(전체 행)에 박스
        rect = plt.Rectangle((-0.5, row_idx - 0.5),
                             n_months, 1, fill=False,
                             edgecolor='magenta', linewidth=2)
        ax.add_patch(rect)

plt.tight_layout()
plt.show()

# ====== 6. 연도별 Train/Validation stacked bar ======
yearly_counts = df.groupby(["year", "split"]).size().unstack(fill_value=0)
yearly_counts = yearly_counts.reindex(columns=["validation", "train"], fill_value=0)

plt.figure(figsize=(14, 6))
years_plot = yearly_counts.index.values
train_vals = yearly_counts["train"].values
val_vals   = yearly_counts["validation"].values

plt.bar(years_plot, train_vals, label="train")
plt.bar(years_plot, val_vals, bottom=train_vals, label="validation")

plt.title("Yearly Train/Validation Data Samples")
plt.ylabel("Number of data samples")
plt.xlabel("Year")
plt.xticks(years_plot, rotation=45)
plt.legend(title="Split")
plt.tight_layout()
plt.show()

# ====== 7. class_day1 기준 비율 + Train/Val 비율 ======
def print_class_ratios(df_train, df_val):
    def calc_ratio(df):
        total = len(df)
        neg = (df["class_day1"] == 0).sum()
        pos = (df["class_day1"] == 1).sum()
        return total, neg, pos, neg / total * 100, pos / total * 100

    train_total, train_neg, train_pos, train_neg_r, train_pos_r = calc_ratio(df_train)
    val_total,   val_neg,   val_pos,   val_neg_r,   val_pos_r   = calc_ratio(df_val)

    all_total = train_total + val_total
    train_total_r = train_total / all_total * 100
    val_total_r   = val_total   / all_total * 100

    print("\n=== class_day1 Positive/Negative Ratio Summary ===")
    print(f"[Train] 2010-2023")
    print(f"  Total: {train_total} ({train_total_r:.2f}% of all data)")
    print(f"  Negative (0): {train_neg} ({train_neg_r:.2f}%)")
    print(f"  Positive (1): {train_pos} ({train_pos_r:.2f}%)")

    print(f"\n[Validation] 2024")
    print(f"  Total: {val_total} ({val_total_r:.2f}% of all data)")
    print(f"  Negative (0): {val_neg} ({val_neg_r:.2f}%)")
    print(f"  Positive (1): {val_pos} ({val_pos_r:.2f}%)")

print_class_ratios(df_train, df_val)

# ====== 8. 월별 total / positive / negative 분포 plot ======
months_index = np.arange(1, 13)

monthly_total = df.groupby("month").size().reindex(months_index, fill_value=0)
monthly_pos   = df[df["class_day1"] == 1].groupby("month").size().reindex(months_index, fill_value=0)
monthly_neg   = df[df["class_day1"] == 0].groupby("month").size().reindex(months_index, fill_value=0)

plt.figure(figsize=(12, 5))
bar_width = 0.35
x = np.arange(len(months_index))

plt.bar(x - bar_width/2, monthly_pos.values, width=bar_width, label="positive")
plt.bar(x + bar_width/2, monthly_neg.values, width=bar_width, label="negative")
plt.plot(x, monthly_total.values, marker='o', label="total")

plt.xticks(x, [f"{m:02d}" for m in months_index])
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Monthly Distribution (All Years)")
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Done. Files saved:")
print(f" - Train (2010-2023): {TRAIN_PATH}")
print(f" - Validation (2024): {VAL_PATH}")
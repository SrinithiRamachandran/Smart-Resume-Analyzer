import pandas as pd
import re

# Load dataset
file_path = "job_descriptions.csv"   # 🔁 change if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("✅ Dataset Loaded Successfully!\n")

# Check available columns
print("📌 Columns in dataset:")
print(df.columns.tolist(), "\n")

# Check if salary column exists
if 'salary range' not in df.columns:
    print("❌ 'salary range' column not found!")
    exit()

# Show sample salary values
print("🔹 Sample Salary Values:")
print(df['salary range'].head(), "\n")

# Function to extract average salary
def extract_avg_salary(s):
    try:
        nums = re.findall(r'\d+', str(s))
        nums = [int(n) for n in nums]

        if len(nums) >= 2:
            return sum(nums[:2]) / 2
        elif len(nums) == 1:
            return nums[0]
        else:
            return None
    except:
        return None

# Apply extraction
df['avg_salary'] = df['salary range'].apply(extract_avg_salary)

# Drop invalid rows
df = df.dropna(subset=['avg_salary'])

# 🔹 Unique salary ranges
print("🔹 Unique Salary Ranges:")
print(df['salary range'].unique(), "\n")

# 🔹 Count of each range
print("🔹 Salary Range Counts:")
print(df['salary range'].value_counts(), "\n")

# 🔹 Basic statistics
print("📊 Salary Statistics:")
print(df['avg_salary'].describe(), "\n")

# 🔹 Salary buckets
bins = [0, 50, 75, 100, 150, 200, 500]
labels = ["0-50K", "50-75K", "75-100K", "100-150K", "150-200K", "200K+"]

df['salary_bucket'] = pd.cut(df['avg_salary'], bins=bins, labels=labels)

print("📊 Salary Bucket Distribution:")
print(df['salary_bucket'].value_counts(), "\n")

# 🔹 Check if salaries are mostly same
std_dev = df['avg_salary'].std()
print(f"📉 Salary Standard Deviation: {std_dev:.2f}")

if std_dev < 10:
    print("⚠️ WARNING: Salaries have very low variation → Model will not learn properly!")

print("\n✅ Analysis Completed!")
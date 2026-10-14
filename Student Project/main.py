"""
Student Performance Analytics System
Technology: Python, NumPy, Pandas
"""

import numpy as np
import pandas as pd

# ---------- Settings (grading & pass rules) ----------
SUBJECTS = ["Maths", "Physics", "Chemistry", "English", "Computer"]
PASS_MARK = 40          # student must score >= 40 in EVERY subject to pass
TOP_N = 5               # number of top performers to show


# ---------- Reusable functions ----------
def load_data(file_name):
    """Read the student CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_name)


def calculate_total_and_average(df):
    """Add Total and Average columns using NumPy (row-wise)."""
    marks = df[SUBJECTS].to_numpy()                 # DataFrame -> NumPy array
    df["Total"] = np.sum(marks, axis=1)             # sum of each student's row
    df["Average"] = np.round(np.mean(marks, axis=1), 2)
    return df


def get_grade(average):
    """Grading rule based on average marks."""
    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    elif average >= 70:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 40:
        return "D"
    else:
        return "F"


def assign_grades(df):
    """Loop over every average and assign a grade."""
    grades = []
    for avg in df["Average"]:
        grades.append(get_grade(avg))
    df["Grade"] = grades
    return df


def assign_result(df):
    """Pass if marks >= PASS_MARK in all subjects, otherwise Fail."""
    all_passed = np.all(df[SUBJECTS].to_numpy() >= PASS_MARK, axis=1)
    df["Result"] = np.where(all_passed, "Pass", "Fail")
    return df


def subject_analysis(df):
    """Average, highest and lowest marks for every subject."""
    rows = []
    for subject in SUBJECTS:
        scores = df[subject].to_numpy()
        rows.append({
            "Subject": subject,
            "Average": round(np.mean(scores), 2),
            "Highest": np.max(scores),
            "Lowest": np.min(scores),
        })
    return pd.DataFrame(rows)


def print_heading(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------- Main program ----------
def main():
    # Step 1: Read data
    df = load_data("students.csv")
    print_heading("1. DATASET OVERVIEW")
    print("Number of records :", df.shape[0])
    print("Columns           :", list(df.columns))
    print("\nFirst 5 records:")
    print(df.head())

    # Step 2: Process data
    df = calculate_total_and_average(df)
    df = assign_grades(df)
    df = assign_result(df)

    print_heading("2. STUDENT RESULTS")
    print(df[["Student_ID", "Name", "Department", "Total",
              "Average", "Grade", "Result", "Attendance"]].to_string(index=False))

    # Step 3: Analysis
    print_heading("3. CLASS ANALYSIS")
    averages = df["Average"].to_numpy()
    best = df.iloc[np.argmax(averages)]
    worst = df.iloc[np.argmin(averages)]
    print(f"Overall class average : {np.mean(averages):.2f}")
    print(f"Highest average       : {best['Name']} ({best['Average']})")
    print(f"Lowest average        : {worst['Name']} ({worst['Average']})")

    passed = (df["Result"] == "Pass").sum()
    failed = (df["Result"] == "Fail").sum()
    print(f"Students passed       : {passed}")
    print(f"Students failed       : {failed}")
    print(f"Pass percentage       : {passed / len(df) * 100:.1f}%")
    print(f"Average attendance    : {df['Attendance'].mean():.1f}%")

    print_heading("4. GRADE DISTRIBUTION")
    print(df["Grade"].value_counts().sort_index().to_string())

    print_heading("5. SUBJECT-WISE PERFORMANCE")
    subject_df = subject_analysis(df)
    print(subject_df.to_string(index=False))
    top_subject = subject_df.loc[subject_df["Average"].idxmax(), "Subject"]
    weak_subject = subject_df.loc[subject_df["Average"].idxmin(), "Subject"]
    print(f"\nBest subject  : {top_subject}")
    print(f"Weakest subject: {weak_subject}")

    print_heading("6. DEPARTMENT-WISE AVERAGE")
    print(df.groupby("Department")["Average"].mean().round(2).to_string())

    print_heading(f"7. TOP {TOP_N} PERFORMERS")
    top = df.sort_values("Average", ascending=False).head(TOP_N)
    rank = 1
    for _, row in top.iterrows():          # loop to print ranks
        print(f"Rank {rank}: {row['Name']:<16} Avg = {row['Average']:<6} Grade = {row['Grade']}")
        rank += 1

    print_heading("8. FAILED STUDENTS")
    print(df[df["Result"] == "Fail"][["Student_ID", "Name", "Average"]].to_string(index=False))

    # Save the final processed data
    df.to_csv("results.csv", index=False)
    print("\nFull results saved to results.csv")


if __name__ == "__main__":
    main()
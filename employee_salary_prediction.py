import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# EMPLOYEE SALARY PREDICTION SYSTEM
# Tkinter GUI + Linear Regression + Excel Dataset
# ============================================================

DATASET_FILE = "employee_salary_dataset.xlsx"

# ----------------------------- COLORS -------------------------
BG = "#F4F7FB"
CARD = "#FFFFFF"
NAVY = "#132A4A"
BLUE = "#1677FF"
BLUE_DARK = "#0D5ED7"
TEXT = "#172B4D"
MUTED = "#6B7A90"
BORDER = "#DCE3ED"
LIGHT_BLUE = "#EAF3FF"
RESET_BG = "#E9EEF5"
RESET_HOVER = "#D7E0EC"


# ---------------------- LOAD DATASET --------------------------
def load_dataset():
    try:
        df = pd.read_excel(DATASET_FILE)

    except FileNotFoundError:
        messagebox.showerror(
            "Dataset Not Found",
            f"Could not find:\n{DATASET_FILE}\n\n"
            "Keep the Excel file in the same folder as this Python file."
        )
        return None

    except Exception as e:
        messagebox.showerror(
            "Dataset Error",
            f"Could not read the Excel file.\n\n{e}"
        )
        return None

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.lower()
    )

    # Possible names for each required column
    aliases = {
        "age": ["age"],
        "gender": ["gender", "sex"],
        "education": ["education", "educationlevel"],
        "jobtitle": ["jobtitle", "job", "designation"],
        "yearsofexperience": [
            "yearsofexperience",
            "experience",
            "yearsexperience"
        ],
        "location": ["location", "city"],
        "salary": ["salary", "annualsalary", "salaryinr"],
    }

    rename_map = {}

    for standard_name, possible_names in aliases.items():
        for column in df.columns:
            if column in possible_names:
                rename_map[column] = standard_name
                break

    df = df.rename(columns=rename_map)

    required_columns = [
        "age",
        "gender",
        "education",
        "jobtitle",
        "yearsofexperience",
        "location",
        "salary"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        messagebox.showerror(
            "Invalid Dataset",
            "The Excel file is missing these columns:\n\n"
            + "\n".join(missing)
            + "\n\nExpected columns:\n"
            + ", ".join(required_columns)
        )
        return None

    # Convert numerical columns
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["yearsofexperience"] = pd.to_numeric(
        df["yearsofexperience"],
        errors="coerce"
    )
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    # Remove incomplete rows
    df = df.dropna(subset=required_columns).copy()

    # Clean categorical columns
    for column in [
        "gender",
        "education",
        "jobtitle",
        "location"
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    if len(df) < 10:
        messagebox.showerror(
            "Not Enough Data",
            "The dataset needs at least 10 valid rows."
        )
        return None

    return df


# ------------------------- TRAIN MODEL ------------------------
def train_model(df):

    features = [
        "age",
        "gender",
        "education",
        "jobtitle",
        "yearsofexperience",
        "location"
    ]

    target = "salary"

    X = df[features]
    y = df[target]

    numeric_features = [
        "age",
        "yearsofexperience"
    ]

    categorical_features = [
        "gender",
        "education",
        "jobtitle",
        "location"
    ]

    # Support both newer and older scikit-learn versions
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numeric_features
            ),
            (
                "cat",
                encoder,
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                LinearRegression()
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    return model, metrics


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Employee Salary Prediction System"
)

root.geometry(
    "1180x760"
)

root.minsize(
    1000,
    680
)

root.configure(
    bg=BG
)


# --------------------------- STYLES ----------------------------
style = ttk.Style()

style.theme_use("clam")

style.configure(
    "TEntry",
    fieldbackground="white",
    foreground=TEXT,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER,
    padding=10,
    font=("Segoe UI", 11)
)

style.configure(
    "TCombobox",
    fieldbackground="white",
    background="white",
    foreground=TEXT,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER,
    padding=10,
    font=("Segoe UI", 11)
)

style.map(
    "TCombobox",
    fieldbackground=[
        ("readonly", "white")
    ],
    selectbackground=[
        ("readonly", LIGHT_BLUE)
    ],
    selectforeground=[
        ("readonly", TEXT)
    ]
)


# --------------------- LOAD DATA + MODEL -----------------------

df = load_dataset()

if df is None:
    root.destroy()
    raise SystemExit


try:
    model, metrics = train_model(df)

except Exception as e:
    messagebox.showerror(
        "Training Error",
        f"The model could not be trained.\n\n{e}"
    )

    root.destroy()
    raise SystemExit


# ----------------------- HELPER FUNCTIONS ----------------------

def money(value):
    return f"₹{value:,.2f}"


def clear_result():

    result_salary.config(
        text="₹0.00"
    )

    result_subtitle.config(
        text="Enter employee details and click Predict Salary"
    )


# ------------------------- RESET FORM --------------------------

def reset_form():

    age_var.set("")
    gender_var.set("")
    education_var.set("")
    job_var.set("")
    experience_var.set("")
    location_var.set("")

    clear_result()

    status_var.set(
        "Form reset • Ready for prediction"
    )

    age_entry.focus()


# ----------------------- PREDICT SALARY ------------------------

def predict_salary():

    try:

        age = float(
            age_var.get()
        )

        experience = float(
            experience_var.get()
        )

        gender = (
            gender_var.get()
            .strip()
        )

        education = (
            education_var.get()
            .strip()
        )

        job_title = (
            job_var.get()
            .strip()
        )

        location = (
            location_var.get()
            .strip()
        )

        # Check empty fields
        if (
            not gender
            or not education
            or not job_title
            or not location
        ):

            messagebox.showwarning(
                "Missing Information",
                "Please select all dropdown values."
            )

            return

        # Age validation
        if age < 18 or age > 80:

            messagebox.showwarning(
                "Invalid Age",
                "Please enter an age between 18 and 80."
            )

            return

        # Experience validation
        if experience < 0 or experience > 60:

            messagebox.showwarning(
                "Invalid Experience",
                "Please enter experience between 0 and 60 years."
            )

            return

        # Logical experience validation
        if experience > age - 16:

            messagebox.showwarning(
                "Invalid Experience",
                "Years of experience cannot be greater than "
                "the possible working years."
            )

            return

        # Create new employee DataFrame
        new_employee = pd.DataFrame(
            [{
                "age": age,
                "gender": gender,
                "education": education,
                "jobtitle": job_title,
                "yearsofexperience": experience,
                "location": location
            }]
        )

        # Predict
        prediction = float(
            model.predict(new_employee)[0]
        )

        # Prevent negative salary display
        prediction = max(
            0,
            prediction
        )

        # Update GUI
        result_salary.config(
            text=money(prediction)
        )

        result_subtitle.config(
            text="Estimated annual salary based on the trained ML model"
        )

        status_var.set(
            "Prediction completed successfully"
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Age and Years of Experience must contain numbers."
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            f"Could not make the prediction.\n\n{e}"
        )


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=NAVY,
    height=125
)

header.pack(
    fill="x"
)

header.pack_propagate(
    False
)


title = tk.Label(
    header,
    text="EMPLOYEE SALARY PREDICTION",
    bg=NAVY,
    fg="white",
    font=("Segoe UI", 28, "bold")
)

title.pack(
    pady=(25, 3)
)


subtitle = tk.Label(
    header,
    text="Machine Learning Based Salary Prediction System",
    bg=NAVY,
    fg="#B9D7FF",
    font=("Segoe UI", 12)
)

subtitle.pack()


# ============================================================
# BODY
# ============================================================

body = tk.Frame(
    root,
    bg=BG
)

body.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=25
)


# ============================================================
# LEFT CARD
# ============================================================

left_card = tk.Frame(
    body,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

left_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 15)
)


left_inner = tk.Frame(
    left_card,
    bg=CARD
)

left_inner.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=28
)


tk.Label(
    left_inner,
    text="Employee Details",
    bg=CARD,
    fg=TEXT,
    font=("Segoe UI", 20, "bold")
).grid(
    row=0,
    column=0,
    columnspan=2,
    sticky="w"
)


tk.Label(
    left_inner,
    text="Enter employee information below",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 10)
).grid(
    row=1,
    column=0,
    columnspan=2,
    sticky="w",
    pady=(3, 25)
)


# ------------------------- VARIABLES ---------------------------

age_var = tk.StringVar()
gender_var = tk.StringVar()
education_var = tk.StringVar()
job_var = tk.StringVar()
experience_var = tk.StringVar()
location_var = tk.StringVar()


# ------------------------- FIELD FUNCTION ---------------------

def add_field(
    row,
    label_text,
    widget
):

    tk.Label(
        left_inner,
        text=label_text,
        bg=CARD,
        fg=TEXT,
        font=("Segoe UI", 11, "bold")
    ).grid(
        row=row,
        column=0,
        sticky="w",
        padx=(0, 20),
        pady=8
    )

    widget.grid(
        row=row,
        column=1,
        sticky="ew",
        pady=8
    )


left_inner.columnconfigure(
    1,
    weight=1
)


# ---------------------------- AGE ------------------------------

age_entry = ttk.Entry(
    left_inner,
    textvariable=age_var
)

add_field(
    2,
    "Age",
    age_entry
)


# --------------------------- GENDER ----------------------------

gender_combo = ttk.Combobox(
    left_inner,
    textvariable=gender_var,
    values=sorted(
        df["gender"].unique().tolist()
    ),
    state="readonly"
)

add_field(
    3,
    "Gender",
    gender_combo
)


# ------------------------- EDUCATION ---------------------------

education_combo = ttk.Combobox(
    left_inner,
    textvariable=education_var,
    values=sorted(
        df["education"].unique().tolist()
    ),
    state="readonly"
)

add_field(
    4,
    "Education",
    education_combo
)


# -------------------------- JOB TITLE --------------------------

job_combo = ttk.Combobox(
    left_inner,
    textvariable=job_var,
    values=sorted(
        df["jobtitle"].unique().tolist()
    ),
    state="readonly"
)

add_field(
    5,
    "Job Title",
    job_combo
)


# ----------------------- EXPERIENCE ----------------------------

experience_entry = ttk.Entry(
    left_inner,
    textvariable=experience_var
)

add_field(
    6,
    "Years of Experience",
    experience_entry
)


# -------------------------- LOCATION ---------------------------

location_combo = ttk.Combobox(
    left_inner,
    textvariable=location_var,
    values=sorted(
        df["location"].unique().tolist()
    ),
    state="readonly"
)

add_field(
    7,
    "Location",
    location_combo
)


# ============================================================
# BUTTONS
# Reset button is directly beside Predict Salary
# ============================================================

button_frame = tk.Frame(
    left_inner,
    bg=CARD
)

button_frame.grid(
    row=8,
    column=0,
    columnspan=2,
    pady=(28, 10)
)


# --------------------- PREDICT BUTTON -------------------------

predict_button = tk.Button(
    button_frame,
    text="  PREDICT SALARY  ",
    command=predict_salary,
    bg=BLUE,
    fg="white",
    activebackground=BLUE_DARK,
    activeforeground="white",
    relief="flat",
    bd=0,
    cursor="hand2",
    font=("Segoe UI", 12, "bold"),
    padx=18,
    pady=12
)

predict_button.pack(
    side="left",
    padx=6
)


# ------------------------ RESET BUTTON ------------------------

reset_button = tk.Button(
    button_frame,
    text="  RESET  ",
    command=reset_form,
    bg=RESET_BG,
    fg=TEXT,
    activebackground=RESET_HOVER,
    activeforeground=TEXT,
    relief="flat",
    bd=0,
    cursor="hand2",
    font=("Segoe UI", 11, "bold"),
    padx=18,
    pady=12
)

reset_button.pack(
    side="left",
    padx=6
)


# ============================================================
# RIGHT RESULT CARD
# ============================================================

right_card = tk.Frame(
    body,
    bg=NAVY,
    width=330,
    highlightbackground=NAVY,
    highlightthickness=1
)

right_card.pack(
    side="right",
    fill="y"
)

right_card.pack_propagate(
    False
)


right_inner = tk.Frame(
    right_card,
    bg=NAVY
)

right_inner.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=30
)


tk.Label(
    right_inner,
    text="Prediction Result",
    bg=NAVY,
    fg="white",
    font=("Segoe UI", 20, "bold")
).pack(
    anchor="w"
)


# Blue line
tk.Frame(
    right_inner,
    bg=BLUE,
    height=4,
    width=65
).pack(
    anchor="w",
    pady=(12, 35)
)


tk.Label(
    right_inner,
    text="PREDICTED ANNUAL SALARY",
    bg=NAVY,
    fg="#75B5FF",
    font=("Segoe UI", 9, "bold")
).pack()


result_salary = tk.Label(
    right_inner,
    text="₹0.00",
    bg=NAVY,
    fg="white",
    font=("Segoe UI", 26, "bold")
)

result_salary.pack(
    pady=(8, 3)
)


result_subtitle = tk.Label(
    right_inner,
    text="Enter employee details and click Predict Salary",
    bg=NAVY,
    fg="#B7C5D9",
    font=("Segoe UI", 9),
    wraplength=260,
    justify="center"
)

result_subtitle.pack(
    pady=(0, 35)
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

tk.Label(
    right_inner,
    text="MODEL PERFORMANCE",
    bg=NAVY,
    fg="#75B5FF",
    font=("Segoe UI", 9, "bold")
).pack(
    pady=(5, 15)
)


def metric_row(
    label,
    value
):

    frame = tk.Frame(
        right_inner,
        bg=NAVY
    )

    frame.pack(
        fill="x",
        pady=6
    )

    tk.Label(
        frame,
        text=label,
        bg=NAVY,
        fg="#B7C5D9",
        font=("Segoe UI", 10)
    ).pack(
        side="left"
    )

    tk.Label(
        frame,
        text=value,
        bg=NAVY,
        fg="white",
        font=("Segoe UI", 10, "bold")
    ).pack(
        side="right"
    )


metric_row(
    "R² Score",
    f"{metrics['r2']:.4f}"
)

metric_row(
    "MAE",
    money(metrics["mae"])
)

metric_row(
    "RMSE",
    money(metrics["rmse"])
)


# Divider
tk.Frame(
    right_inner,
    bg="#29425F",
    height=1
).pack(
    fill="x",
    pady=22
)


tk.Label(
    right_inner,
    text=f"Training records: {metrics['train_size']}",
    bg=NAVY,
    fg="#B7C5D9",
    font=("Segoe UI", 9)
).pack(
    anchor="w"
)


tk.Label(
    right_inner,
    text=f"Testing records: {metrics['test_size']}",
    bg=NAVY,
    fg="#B7C5D9",
    font=("Segoe UI", 9)
).pack(
    anchor="w",
    pady=(5, 0)
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Frame(
    root,
    bg=BG,
    height=35
)

footer.pack(
    fill="x"
)

footer.pack_propagate(
    False
)


status_var = tk.StringVar(
    value="Model trained successfully • Ready for prediction"
)


tk.Label(
    footer,
    textvariable=status_var,
    bg=BG,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(
    side="left",
    padx=35
)


tk.Label(
    footer,
    text="Employee Salary Prediction • Machine Learning Project",
    bg=BG,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(
    side="right",
    padx=35
)


# ============================================================
# KEYBOARD SHORTCUTS
# ============================================================

root.bind(
    "<Return>",
    lambda event: predict_salary()
)

root.bind(
    "<Escape>",
    lambda event: reset_form()
)


# Start with cursor in Age field
age_entry.focus()


# ============================================================
# RUN APPLICATION
# ============================================================

root.mainloop()
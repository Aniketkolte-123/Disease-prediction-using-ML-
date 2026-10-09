# ------------------ IMPORTS ------------------

from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd


# ------------------ SYMPTOMS LIST ------------------

l1 = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
    'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
    'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
    'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'loss_of_smell', 'bladder_discomfort',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'depression', 'irritability', 'muscle_pain', 'belly_pain'
]


# ------------------ DISEASE LIST ------------------

disease = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
    'Drug Reaction', 'Peptic ulcer disease', 'AIDS', 'Diabetes',
    'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine',
    'Cervical spondylosis', 'Paralysis', 'Jaundice', 'Malaria',
    'Chicken pox', 'Dengue', 'Typhoid', 'Hepatitis A', 'Hepatitis B',
    'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
    'Tuberculosis', 'Common Cold', 'Pneumonia', 'Piles', 'Heart attack',
    'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
    'Osteoarthritis', 'Arthritis', 'Vertigo', 'Acne',
    'Urinary tract infection', 'Psoriasis', 'Impetigo'
]


# ------------------ DATASET ------------------

df = pd.read_csv("Training.csv")

X = df[l1]
y = df["prognosis"]

tr = pd.read_csv("Testing.csv")

X_test = tr[l1]
y_test = tr["prognosis"]


# ============================================================
#                         MAIN WINDOW
# IMPORTANT: root MUST be created BEFORE StringVar()
# ============================================================

root = Tk()

root.title("AI Healthcare Assistant | Disease Prediction")

# Responsive startup size: fit the available screen instead of forcing 1180x850.
root.configure(bg="#F4FAFB")
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
start_w = min(1280, max(760, screen_w - 60))
start_h = min(900, max(600, screen_h - 100))
root.geometry(f"{start_w}x{start_h}+20+20")
root.minsize(760, 600)

# Keep the UI usable with Windows display scaling and different monitor sizes.
try:
    root.tk.call("tk", "scaling", min(1.5, max(1.0, screen_w / 1440)))
except tk.TclError:
    pass


# ============================================================
#                         GUI VARIABLES
# ============================================================

Name = StringVar(
    master=root
)

symptom_vars = [
    StringVar(
        master=root,
        value="Select symptom"
    )
    for _ in range(5)
]

results = [
    StringVar(
        master=root,
        value="Waiting for prediction"
    )
    for _ in range(3)
]

status_var = StringVar(
    master=root,
    value="● System Ready"
)

count_var = StringVar(
    master=root,
    value="Selected symptoms: 0 / 5"
)

agreement_var = StringVar(
    master=root,
    value="No prediction yet"
)


# ------------------ SYMPTOM OPTIONS ------------------

OPTIONS = [
    "Select symptom"
] + sorted(l1)


# ============================================================
#                         FUNCTIONS
# ============================================================

def update_count(*args):

    n = sum(
        v.get() != "Select symptom"
        for v in symptom_vars
    )

    count_var.set(
        f"Selected symptoms: {n} / 5"
    )


# ------------------ CLEAR ------------------

def clear_all():

    Name.set("")

    for v in symptom_vars:
        v.set("Select symptom")

    for v in results:
        v.set("Waiting for prediction")

    count_var.set(
        "Selected symptoms: 0 / 5"
    )

    agreement_var.set(
        "No prediction yet"
    )

    status_var.set(
        "● System Ready"
    )


# ------------------ PREDICT ------------------

def predict_all():

    selected = [
        v.get()
        for v in symptom_vars
        if v.get() != "Select symptom"
    ]

    # Check name
    if not Name.get().strip():

        messagebox.showwarning(
            "Patient Information",
            "Please enter the patient name."
        )

        return

    # Check symptoms
    if not selected:

        messagebox.showwarning(
            "Symptoms Required",
            "Please select at least one symptom."
        )

        return

    # Check duplicate symptoms
    if len(selected) != len(set(selected)):

        messagebox.showwarning(
            "Duplicate Symptoms",
            "Please select different symptoms."
        )

        return


    # ========================================================
    # CREATE INPUT VECTOR
    # ========================================================

    l2 = [0] * len(l1)

    for k in range(len(l1)):

        if l1[k] in selected:
            l2[k] = 1


    # IMPORTANT FIX:
    # Training data has feature names.
    # Prediction input also uses the same feature names.

    inp = pd.DataFrame(
        [l2],
        columns=l1
    )


    # ========================================================
    # ML MODELS
    # ========================================================

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB


    models = [

        DecisionTreeClassifier(
            random_state=42
        ),

        RandomForestClassifier(
            random_state=42
        ),

        GaussianNB()
    ]


    # ========================================================
    # TRAIN AND PREDICT
    # ========================================================

    try:

        out = []


        for model in models:

            # Train
            model.fit(X, y)

            # Predict
            prediction = model.predict(inp)[0]

            out.append(
                str(prediction)
            )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        results[0].set(
            out[0]
        )

        results[1].set(
            out[1]
        )

        results[2].set(
            out[2]
        )


        # ====================================================
        # MODEL AGREEMENT
        # ====================================================

        from collections import Counter

        counter = Counter(out)

        disease_name, number = counter.most_common(1)[0]


        if number > 1:

            agreement_var.set(
                f"✓ {number} / 3 Models Agree  •  {disease_name}"
            )

        else:

            agreement_var.set(
                "⚠ Models produced different predictions"
            )


        status_var.set(
            "● Prediction Complete"
        )


    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

        status_var.set(
            "● Prediction Error"
        )


# ============================================================
#                         COLORS
# ============================================================

BG = "#F4FAFB"
CARD = "#FFFFFF"
PRIMARY = "#1F7A8C"
ACCENT = "#2A9D8F"
DARK = "#24323D"
MUTED = "#6B7280"
LIGHT = "#E8F4F8"
BORDER = "#D8E8EC"


# ============================================================
#                         TEAM
# ============================================================

TEAM = [
    ("Aniket Kolte", "2305192"),
    ("Yuvraj Gadekar", "2305222"),
    ("Pranav Kalam", "2305217"),
    ("Atharv Chavan ", "2305221")  
]


# ============================================================
#                         STYLE
# ============================================================

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "Modern.TCombobox",
    fieldbackground=CARD,
    background=CARD,
    foreground=DARK,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER,
    padding=8,
    font=("Segoe UI", 11)
)


# ============================================================
#                         HEADER
# ============================================================

header = Frame(
    root,
    bg=PRIMARY,
    height=82
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


h = Frame(
    header,
    bg=PRIMARY
)

h.pack(
    side="left",
    padx=28,
    pady=12
)


Label(
    h,
    text="🩺  AI HEALTHCARE ASSISTANT",
    bg=PRIMARY,
    fg="white",
    font=("Segoe UI", 22, "bold")
).pack(
    anchor="w"
)


Label(
    h,
    text="Disease Prediction Using Machine Learning",
    bg=PRIMARY,
    fg="#DDF4F5",
    font=("Segoe UI", 11)
).pack(
    anchor="w"
)


Label(
    header,
    textvariable=status_var,
    bg=PRIMARY,
    fg="white",
    font=("Segoe UI", 11, "bold")
).pack(
    side="right",
    padx=28
)


# ============================================================
#                         MAIN
# ============================================================

# Scrollable application workspace. This prevents the lower cards and buttons
# from being clipped on short displays or when Windows display scaling is high.
workspace = Frame(root, bg=BG)
workspace.pack(fill="both", expand=True)

main_canvas = Canvas(workspace, bg=BG, highlightthickness=0, bd=0)
main_scrollbar = ttk.Scrollbar(workspace, orient="vertical", command=main_canvas.yview)
main_canvas.configure(yscrollcommand=main_scrollbar.set)

main_scrollbar.pack(side="right", fill="y")
main_canvas.pack(side="left", fill="both", expand=True)

main = Frame(main_canvas, bg=BG)
main_window = main_canvas.create_window((0, 0), window=main, anchor="nw")

def _sync_scroll_region(_event=None):
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))

def _fit_main_width(event):
    main_canvas.itemconfigure(main_window, width=event.width)

main.bind("<Configure>", _sync_scroll_region)
main_canvas.bind("<Configure>", _fit_main_width)

# Mouse-wheel scrolling (Windows and Linux).
def _mousewheel_scroll(event):
    if getattr(event, "delta", 0):
        main_canvas.yview_scroll(int(-event.delta / 120), "units")
    elif getattr(event, "num", None) == 4:
        main_canvas.yview_scroll(-3, "units")
    elif getattr(event, "num", None) == 5:
        main_canvas.yview_scroll(3, "units")

def _bind_mousewheel(_event=None):
    main_canvas.bind_all("<MouseWheel>", _mousewheel_scroll)
    main_canvas.bind_all("<Button-4>", _mousewheel_scroll)
    main_canvas.bind_all("<Button-5>", _mousewheel_scroll)

def _unbind_mousewheel(_event=None):
    main_canvas.unbind_all("<MouseWheel>")
    main_canvas.unbind_all("<Button-4>")
    main_canvas.unbind_all("<Button-5>")

main_canvas.bind("<Enter>", _bind_mousewheel)
main_canvas.bind("<Leave>", _unbind_mousewheel)

# Smaller side margins on narrow windows.
def _responsive_margins(event):
    margin = 10 if event.width < 1000 else 24
    main.pack_configure(padx=margin, pady=12 if event.width < 1000 else 20)

main_canvas.bind("<Configure>", _responsive_margins, add="+")


# ============================================================
#                         SIDEBAR
# ============================================================

side = Frame(
    main,
    bg="#173B46",
    width=175
)

side.pack(
    side="left",
    fill="y",
    padx=(0, 12)
)

side.pack_propagate(False)


Label(
    side,
    text="🩺",
    bg="#173B46",
    fg="white",
    font=("Segoe UI", 34)
).pack(
    pady=(25, 0)
)


Label(
    side,
    text="AI HEALTH",
    bg="#173B46",
    fg="white",
    font=("Segoe UI", 16, "bold")
).pack()


Label(
    side,
    text="ASSISTANT",
    bg="#173B46",
    fg="#8ED8D0",
    font=("Segoe UI", 10, "bold")
).pack(
    pady=(0, 30)
)


for item in [
    "🏠  Dashboard",
    "🔍  Prediction",
    "🧠  ML Models",
    "ℹ  About"
]:

    Label(
        side,
        text=item,
        bg="#173B46",
        fg="#EAF7F8",
        font=("Segoe UI", 11, "bold"),
        anchor="w",
        padx=18,
        pady=12
    ).pack(
        fill="x",
        padx=8,
        pady=2
    )


Frame(
    side,
    bg="#315963",
    height=1
).pack(
    fill="x",
    padx=18,
    pady=20
)


Label(
    side,
    text="PROJECT TEAM",
    bg="#173B46",
    fg="#8ED8D0",
    font=("Segoe UI", 9, "bold")
).pack(
    anchor="w",
    padx=18
)


for name, roll in TEAM:

    Label(
        side,
        text=f"{name}\n{roll}",
        bg="#173B46",
        fg="white",
        font=("Segoe UI", 9),
        anchor="w",
        justify="left",
        pady=4
    ).pack(
        fill="x",
        padx=18
    )


Label(
    side,
    text="v1.0  •  Academic Project",
    bg="#173B46",
    fg="#AFC6CC",
    font=("Segoe UI", 8)
).pack(
    side="bottom",
    pady=15
)


# ============================================================
#                         CONTENT
# ============================================================

content = Frame(
    main,
    bg=BG
)

content.pack(
    side="left",
    fill="both",
    expand=True
)


# ============================================================
#                         STATISTICS
# ============================================================

stats = Frame(
    content,
    bg=BG
)

stats.pack(
    fill="x",
    pady=(0, 15)
)


for icon, val, label in [

    ("🧠", "3", "ML Algorithms"),

    ("🦠", str(len(disease)), "Disease Classes"),

    ("🔍", str(len(l1)), "Symptom Features")

]:

    c = Frame(
        stats,
        bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    c.pack(
        side="left",
        fill="x",
        expand=True,
        padx=(0, 10)
    )


    Label(
        c,
        text=icon,
        bg=CARD,
        fg=PRIMARY,
        font=("Segoe UI", 18)
    ).pack(
        side="left",
        padx=(14, 8),
        pady=12
    )


    f = Frame(
        c,
        bg=CARD
    )

    f.pack(
        side="left",
        pady=8
    )


    Label(
        f,
        text=val,
        bg=CARD,
        fg=DARK,
        font=("Segoe UI", 17, "bold")
    ).pack(
        anchor="w"
    )


    Label(
        f,
        text=label,
        bg=CARD,
        fg=MUTED,
        font=("Segoe UI", 9)
    ).pack(
        anchor="w"
    )


# ============================================================
#                     PREDICTION RESULTS
# ============================================================

res = Frame(
    content,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

res.pack(
    fill="x",
    pady=(0, 15)
)


Label(
    res,
    text="🩺  PREDICTION RESULTS",
    bg=CARD,
    fg=DARK,
    font=("Segoe UI", 15, "bold")
).pack(
    anchor="w",
    padx=20,
    pady=(15, 10)
)


rf = Frame(
    res,
    bg=CARD
)

rf.pack(
    fill="x",
    padx=20
)


# ------------------ DECISION TREE ------------------

rc1 = Frame(
    rf,
    bg=LIGHT,
    highlightbackground="#C7E1E6",
    highlightthickness=1
)

rc1.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10),
    pady=(0, 10)
)


Label(
    rc1,
    text="🌳  DECISION TREE",
    bg=LIGHT,
    fg=PRIMARY,
    font=("Segoe UI", 10, "bold")
).pack(
    pady=(12, 5)
)


Label(
    rc1,
    textvariable=results[0],
    bg=LIGHT,
    fg=DARK,
    font=("Segoe UI", 12, "bold"),
    wraplength=180
).pack(
    pady=(2, 15)
)


# ------------------ RANDOM FOREST ------------------

rc2 = Frame(
    rf,
    bg=LIGHT,
    highlightbackground="#C7E1E6",
    highlightthickness=1
)

rc2.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10),
    pady=(0, 10)
)


Label(
    rc2,
    text="🌲  RANDOM FOREST",
    bg=LIGHT,
    fg=PRIMARY,
    font=("Segoe UI", 10, "bold")
).pack(
    pady=(12, 5)
)


Label(
    rc2,
    textvariable=results[1],
    bg=LIGHT,
    fg=DARK,
    font=("Segoe UI", 12, "bold"),
    wraplength=180
).pack(
    pady=(2, 15)
)


# ------------------ NAIVE BAYES ------------------

rc3 = Frame(
    rf,
    bg=LIGHT,
    highlightbackground="#C7E1E6",
    highlightthickness=1
)

rc3.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 0),
    pady=(0, 10)
)


Label(
    rc3,
    text="📊  NAIVE BAYES",
    bg=LIGHT,
    fg=PRIMARY,
    font=("Segoe UI", 10, "bold")
).pack(
    pady=(12, 5)
)


Label(
    rc3,
    textvariable=results[2],
    bg=LIGHT,
    fg=DARK,
    font=("Segoe UI", 12, "bold"),
    wraplength=180
).pack(
    pady=(2, 15)
)


# ------------------ AGREEMENT ------------------

Label(
    res,
    textvariable=agreement_var,
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
).pack(
    pady=(2, 12)
)


# ============================================================
#                  PATIENT & SYMPTOM ANALYSIS
# ============================================================

card = Frame(
    content,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

card.pack(
    fill="x",
    pady=(0, 12)
)


Label(
    card,
    text="👤  PATIENT & SYMPTOM ANALYSIS",
    bg=CARD,
    fg=DARK,
    font=("Segoe UI", 15, "bold")
).pack(
    anchor="w",
    padx=20,
    pady=(17, 12)
)


# ------------------ PATIENT NAME ------------------

row = Frame(
    card,
    bg=CARD
)

row.pack(
    fill="x",
    padx=20
)


Label(
    row,
    text="Patient Name",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
).pack(
    side="left"
)


Entry(
    row,
    textvariable=Name,
    bg="#F9FCFD",
    fg=DARK,
    relief="flat",
    highlightthickness=1,
    highlightbackground=BORDER,
    font=("Segoe UI", 11)
).pack(
    side="left",
    fill="x",
    expand=True,
    padx=(15, 0),
    ipady=8
)


# ------------------ SYMPTOM TITLE ------------------

Label(
    card,
    text="🔍  Select symptoms experienced by the patient",
    bg=CARD,
    fg=DARK,
    font=("Segoe UI", 11, "bold")
).pack(
    anchor="w",
    padx=20,
    pady=(18, 8)
)


# ------------------ SYMPTOMS ------------------

grid = Frame(
    card,
    bg=CARD
)

grid.pack(
    fill="x",
    padx=20
)


for i, v in enumerate(symptom_vars):

    r = Frame(
        grid,
        bg=CARD
    )

    r.pack(
        fill="x",
        pady=4
    )


    Label(
        r,
        text=f"Symptom {i + 1}",
        bg=CARD,
        fg=MUTED,
        width=13,
        anchor="w",
        font=("Segoe UI", 10)
    ).pack(
        side="left"
    )


    cb = ttk.Combobox(
        r,
        textvariable=v,
        values=OPTIONS,
        state="readonly",
        style="Modern.TCombobox"
    )

    cb.pack(
        side="left",
        fill="x",
        expand=True
    )


    v.trace_add(
        "write",
        update_count
    )


# ------------------ COUNT ------------------

Label(
    card,
    textvariable=count_var,
    bg=CARD,
    fg=ACCENT,
    font=("Segoe UI", 9, "bold")
).pack(
    anchor="w",
    padx=20,
    pady=(7, 5)
)


# ------------------ BUTTONS ------------------

br = Frame(
    card,
    bg=CARD
)

br.pack(
    fill="x",
    padx=20,
    pady=(5, 18)
)


Button(
    br,
    text="🔮  PREDICT DISEASE",
    command=predict_all,
    bg=ACCENT,
    fg="white",
    activebackground="#21877B",
    relief="flat",
    cursor="hand2",
    font=("Segoe UI", 11, "bold"),
    padx=25,
    pady=11
).pack(
    side="left"
)


Button(
    br,
    text="↻  CLEAR",
    command=clear_all,
    bg="#EEF5F7",
    fg=DARK,
    activebackground="#DDEBED",
    relief="flat",
    cursor="hand2",
    font=("Segoe UI", 10, "bold"),
    padx=22,
    pady=11
).pack(
    side="left",
    padx=10
)


# ============================================================
#                         WARNING
# ============================================================

Label(
    content,
    text="⚠ Educational prediction only — this application is not a substitute for professional medical diagnosis.",
    bg=BG,
    fg=MUTED,
    font=("Segoe UI", 8)
).pack(
    anchor="w"
)


# ============================================================
#                         START
# ============================================================

root.mainloop()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  datetime import datetime
from rapidfuzz import process, fuzz
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


os.makedirs("image", exist_ok=True)

#%% 1 Data acquisition
path = "data/customer_churn_dataset.csv"
df = pd.read_csv(path)

#%% 2 Data inspection
#step2 inspection
#print("Df Head\n",df.head())
#print("Data info\n",df.info())
#print("Duplicates (sum):",df.duplicated().sum())
#print("Df NAN Value:\n", df.isna().sum())
#print("Df Description:\n", df.describe)
# %% 3 Data Cleaning and (Treatment NAN values)

#df.dropna(inplace=True)

#(df.isna().sum()) #inspenction
# %% 4 Standarzing columns and vales type object(str)

#print(df.columns) #columns inspection
#1 standarzing columns name
df.columns = df.columns.str.strip().str.capitalize()
df.rename(columns={"Tenure":"Ternure(m)"})

#print(df.columns)

#2 standarzing values type object
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.capitalize() if isinstance(x, str) else(x))

# slecting the top/(correct) values
contract_ = df["Contract"].value_counts().head(5).index.tolist()
payment_method_ = df["Payment_method"].value_counts().head(5).index.tolist()
net_service_ = df["Internet_service"].value_counts().head(5).index.tolist()

#Applying rapiduzz
def safe_fuzzy(x, choices, threshold=80):
    if pd.isna(x):
        return x
    
    match = process.extractOne(
        x,
        choices,
        scorer=fuzz.token_sort_ratio
    )
    
    if match and match[1] >= threshold:
        return match[0]
    
    return x

df["Contract"] = df["Contract"].apply(
    lambda x: safe_fuzzy(x, contract_)
)

df["Payment_method"] = df["Payment_method"].apply(
    lambda x: safe_fuzzy(x, payment_method_)
)

df["Internet_service"] = df["Internet_service"].apply(
    lambda x: safe_fuzzy(x, net_service_)
)


#%% 5 Standarzing values type (int/float(64))
#-1 Standarzing col Ternure

Q1 = df["Tenure"].quantile(0.25)
Q3 = df["Tenure"].quantile(0.75)

IQR = Q3 - Q1

max_lim = Q1 - 1.5 * IQR
min_lim = Q3 + 1.5 * IQR

Ternure_out = df[(df["Tenure"] < min_lim)|(df["Tenure"] > max_lim)]
#print("Outerlier",Ternure_out) #Outerlier inspenction

#treating outerliers(col:Ternure)
df["Tenure"] = df["Tenure"].clip(lower= min_lim, upper=max_lim)

#print(df["Tenure"]) #inspenction

#%% 6 standarzing Monthly charges values

Q1 = df["Monthly_charges"].quantile(0.25)
Q3 = df["Monthly_charges"].quantile(0.75)

IQR = Q3 - Q1

min_lim = Q1 - 1.5 * IQR
max_lim = Q3 + 1.5 * IQR

M_charges_out = df[(df["Monthly_charges"] < min_lim) | (df["Monthly_charges"] < max_lim)]

#print(M_charges_out) #ispenction

#standaring "Monthly_charges" values

df["Monthly_charges"] = df["Monthly_charges"].clip(lower=min_lim, upper=max_lim)
#print(df["Monthly_charges"]) # inspenction

#%% 7 Stanarzing Total_charges values

#Stanarzing col["Total_charges"]
Q1 = df["Total_charges"].quantile(0.25)
Q3 = df["Total_charges"].quantile(0.75)

IQR = Q3 - Q1

min_lim = Q1 - 1.5 * IQR
max_lim = Q3 + 1.5 * IQR

T_charges_out = df[(df["Total_charges"] < min_lim) | (df["Total_charges"] < max_lim)]

#print(T_charges_out) #ispenction

#standaring "Total_charges" values

df["Total_charges"] = df["Total_charges"].clip(lower=min_lim, upper=max_lim)
#print(df["Total_charges"]) # inspenction

#%% 8 EDA
#Ternure Category EDA
bins = [0,12,24,36,60,120]
labels =[
    "New (0-12)",
    "Adapting (12-24)",
    "Stable (24-36)",
    "Loyal (36-60)",
    "Veteran (+60)"
]
df["Tenure_category"] = pd.cut(df["Tenure"], bins=bins, labels=labels,right=False)
#print(df["Tenure_category"]) #inspenction


tenure_dist = (
    df["Tenure_category"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)

#print(tenure_dist) #inspenction

tenure_dist.columns = ["Tenure_category", "Percent"]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
sns.despine(top=True, right=True)
plt.figure(figsize=(5,10))

ax = sns.barplot(data=tenure_dist, x="Tenure_category", y="Percent", color="#121C72")
plt.title("Tenure Category Distribution")
plt.xlabel("Tenure Category", loc= "center")
plt.ylabel("Percent(%)", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)
#plt.show()

#%% Tenure convesion
df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
 })

tenure_churn_conv = (
    df.groupby("Tenure_category", observed=True)["Churn"]
    .mean()
    .mul(100)
    .reset_index()
)
tenure_churn_conv.columns = ["Tenure_category", "Conversion_rate"]

print(tenure_churn_conv) #inspenction

#chart
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
sns.despine(top=True, right=True)
plt.figure(figsize=(5,10))

ax = sns.barplot(data=tenure_churn_conv, x="Tenure_category", y="Conversion_rate", color="#121C72")
plt.title("Churn Conversion Per Tenure Category")
plt.xlabel("Tenure Category", loc= "center")
plt.ylabel("Churn Conversion Rate(%)", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)
#plt.show() #inspenction

#%% #Contrat Analysi 

conntract_dist = (
    df["Contract"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
#print(conntract_dist)#inspection

#Proportion Table

cros_t = pd.crosstab(df["Contract"], df["Churn"], normalize="columns")*100

#print(cros_t) #inspection

#print(cros_t.describe()

df_churn_cust = df[df["Churn"] == 1]

contract_count = df_churn_cust["Contract"].value_counts()


fig, ax = plt.subplots(figsize=(5, 10))

ax.pie(
    contract_count.values,#type: ignore
    labels=contract_count.index,#type: ignore
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Type of contract among churned customers")
ax.axis("equal")

#plt.show()
#%% #Tech support status
#print(df.columns)
support_out = df[(df["Tech_support"] == "No") & (df["Support_calls"] > 0)]

#print(support_out["Support_calls"].value_counts(normalize=True).mul(100).sort_index()) #inspenction


cros_t = pd.crosstab(df["Tech_support"], df["Churn"], normalize="columns")*100
#print(cros_t) #inspenction

#%% #Consumption category category
bins = [0.0,20.0,70.0, df["Monthly_charges"].max()]
labels = ["Low","Medium","High"]
df["Consumption_category"] = pd.cut(df["Monthly_charges"], bins=bins, labels=labels)

#print(df["Consumption_category"])

grouped = (df.groupby("Consumption_category")["Monthly_charges"]
           .sum()
           .reset_index()
           )

customers = df["Consumption_category"].value_counts(normalize=True)*100
print(customers)

grouped.columns = ["Consumption_category","Medium_Ticket"]
#print(grouped)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
sns.despine(top=True, right=True)
plt.figure(figsize=(5,10))

ax = sns.barplot(data=grouped, x="Consumption_category", y="Medium_Ticket", color="#121C72")
plt.title("Cosmsumption Per Category(sum)")
plt.xlabel("Consumption category", loc= "center")
plt.ylabel("", loc="center")
plt.xticks(rotation=45, ha="right")
#plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

#%% #LTV Analysis
ltv_agg = df.groupby("Payment_method").agg(
        churn_rate=("Churn", "mean"),
        ltv_mean=("Total_charges", "mean"),
        customer=("Customer_id", "count")
    ).sort_values("ltv_mean", ascending=False)

#print(ltv_agg) #inspenction
#%% #Analising Churn rate per users online_security state
grouped = (df.groupby("Online_security")["Churn"]
           .mean()
           .mul(100)
           .reset_index()
           )
grouped.columns=["Online_security", "Churn_rate"]

#print(grouped) #Isnpection

# %% #Analysing Churn per support_call
max_call = df["Support_calls"].max()
bins = [-1, 2, 4, max(max_call, 10) ]
labels = ["Stable(0-2)","Atention(2-4)","Critic(4+)"]

df["Support_calls_category"] = pd.cut(df["Support_calls"], bins=bins, labels=labels)

grouped =(
    df.groupby("Support_calls_category").agg(
        customer=("Customer_id","count"),
        churn=("Churn", "sum"),
        churn_rate=("Churn", "mean")
    )
)

grouped["churn_rate"] = grouped["churn_rate"]*100
#print(grouped) #inspection

#Chart
sns.set_theme(style="whitegrid", context="notebook", font_scale=1)
sns.despine(top=True, right=True)
plt.figure(figsize=(5,10))

ax = sns.barplot(data=grouped, x="Support_calls_category", y="churn_rate", color="#121C72")
plt.title("Churn Conversion Per Support Calls Category")
plt.xlabel("Support Calls Category", loc= "center")
plt.ylabel("Churn Rate(%)", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

#%% #chart autosave
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
    plt.show(block=False)

for i, fig_num in enumerate(plt.get_fignums(), start=1):
    fig = plt.figure(fig_num)
    fig.savefig(f"images/chart_{i}.png", dpi=300)

  

#%% Treating NAN values:

#print(df.isna().sum()) #inspenction
df["Internet_service"] = df["Internet_service"].fillna("Unknow")

#print(df["Internet_service"])
#print(df.columns)
#%%Selecting Data to test and prevision
categorical_cols =['Tenure_category','Contract','Payment_method','Internet_service','Tech_support','Online_security',
                'Consumption_category','Support_calls_category']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X= df_encoded.drop(['Customer_id','Churn'], axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
model.fit(X_train, y_train)

#%% Aplying Prevision
df_encoded["Churn_predict"] = model.predict(X)
df_encoded["Probability"] = model.predict_proba(X)[:,1]

df_result = df.copy()
df_result["Churn_predict"] = df_encoded["Churn_predict"].values
df_result["Probability"] = df_encoded["Probability"].values
df_result["Risc_seg"] = pd.cut(df_result["Probability"],
                               bins=[0, 0.3, 0.7, 1],
                               labels=["Low","Medium","High"])
#Saving the model
joblib.dump(model,"churn_model_rf.pkl")

#%% Exporting data
# To CSV
df_result.to_csv("churn_prevision.csv", index=False)

# To excel

with pd.ExcelWriter("churn_dashboard.xlsx") as writer:
    df_result.to_excel(writer, sheet_name="Complete_data", index=False)

    metrics = {
        "Metrics": ["Train_acurace", "Test_acurace", "Total_customers","Churn_predict%","High_risk%"],
        "Value":[
            model.score(X_train, y_train),
            model.score(X_test, y_test),
            len(df_result),
            (df_result["Churn_predict"].sum() / len(df_result))*100,
            ((df_result["Risc_seg"] == "High").sum() / len(df_result))*100
        ]
    }
    pd.DataFrame(metrics).to_excel(writer, sheet_name="Model_Metrics", index=False)

    importance = pd.DataFrame({
        "Variables": X.columns,
        "Importance":model.feature_importances_
    }).sort_values("Importance", ascending=False)
    importance.to_excel(writer, sheet_name="Variables_Importance", index=False)

    high_risk = df_result[df_result["Risc_seg"] == "High"].sort_values("Probability",ascending=False)
    high_risk.head(20).to_excel(writer, sheet_name="High_Risk_Customer", index=False)

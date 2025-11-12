# expense_splitter_final.py
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

st.set_page_config(page_title="Expense Splitter", page_icon="💸", layout="centered")

st.title("💸 Expense Splitter — Split Bills with Friends")
st.caption("Easily calculate who owes money and who should get reimbursed after a dinner or trip!")

# ------------------------------------------------------------
# Data Model and Helper Functions
# ------------------------------------------------------------
@dataclass
class Person:
    name: str
    paid: float
    share: float = 0.0

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def compute_balances(people: List[Person], total_amount: float):
    n = len(people)
    if n == 0:
        return []
    per_share = round(total_amount / n, 2)
    for p in people:
        p.share = per_share
    balances = [
        {
            "name": p.name,
            "paid": round(p.paid, 2),
            "share": p.share,
            "balance": round(p.paid - p.share, 2),
        }
        for p in people
    ]
    return balances

def settle_balances(balances: List[Dict]) -> List[Dict]:
    creditors, debtors = [], []
    for b in balances:
        amt = round(b["balance"], 2)
        if amt > 0:
            creditors.append([b["name"], amt])
        elif amt < 0:
            debtors.append([b["name"], -amt])
    creditors.sort(key=lambda x: x[1], reverse=True)
    debtors.sort(key=lambda x: x[1], reverse=True)

    transfers = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        debtor, owe = debtors[i]
        creditor, recv = creditors[j]
        amt = round(min(owe, recv), 2)
        if amt > 0:
            transfers.append({"from": debtor, "to": creditor, "amount": amt})
        debtors[i][1] -= amt
        creditors[j][1] -= amt
        if debtors[i][1] <= 0.005:
            i += 1
        if creditors[j][1] <= 0.005:
            j += 1
    return transfers

def aggregate_transfers(transfers: List[Dict], names: List[str]):
    agg = {n: {"given": 0.0, "received": 0.0, "net": 0.0} for n in names}
    for t in transfers:
        frm, to, amt = t["from"], t["to"], t["amount"]
        agg[frm]["given"] += amt
        agg[to]["received"] += amt
    for n in agg:
        agg[n]["net"] = round(agg[n]["received"] - agg[n]["given"], 2)
    return agg

def make_result_df(balances, agg):
    rows = []
    for b in balances:
        n = b["name"]
        rows.append({
            "Name": n,
            "Paid": b["paid"],
            "Share": b["share"],
            "Balance": b["balance"],
            "Will Give": agg[n]["given"],
            "Will Get": agg[n]["received"],
            "Net After Transfers": agg[n]["net"],
        })
    df = pd.DataFrame(rows)
    return df[["Name","Paid","Share","Balance","Will Give","Will Get","Net After Transfers"]]

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
with st.form("setup_form"):
    col1, col2 = st.columns(2)
    with col1:
        total_amount = st.number_input("💰 Total bill amount", min_value=0.0, value=100.0, step=1.0, format="%.2f")
    with col2:
        num_people = st.number_input("👥 Number of people", min_value=1, value=3, step=1)
    submitted = st.form_submit_button("Update People")

# Persist name and paid fields
for i in range(int(num_people)):
    if f"name_{i}" not in st.session_state:
        st.session_state[f"name_{i}"] = f"Person {i+1}"
    if f"paid_{i}" not in st.session_state:
        st.session_state[f"paid_{i}"] = 0.0

# Remove extra keys if number of people decreased
for key in list(st.session_state.keys()):
    if key.startswith("name_") or key.startswith("paid_"):
        idx = int(key.split("_")[1])
        if idx >= num_people:
            del st.session_state[key]

# ------------------------------------------------------------
# Editable People Section (live updates)
# ------------------------------------------------------------
st.markdown("### ✏️ Enter names and contributions")
people: List[Person] = []
for i in range(int(num_people)):
    col1, col2 = st.columns([2,1])
    name_key, paid_key = f"name_{i}", f"paid_{i}"
    with col1:
        st.text_input(f"Name #{i+1}", key=name_key)
    with col2:
        st.number_input(f"Paid by {st.session_state[name_key]}", min_value=0.0, value=st.session_state[paid_key],
                        step=0.5, format="%.2f", key=paid_key)
    people.append(Person(name=st.session_state[name_key], paid=float(st.session_state[paid_key])))

# ------------------------------------------------------------
# Computation
# ------------------------------------------------------------
balances = compute_balances(people, total_amount)
transfers = settle_balances(balances)
agg = aggregate_transfers(transfers, [p.name for p in people])
result_df = make_result_df(balances, agg)

# ------------------------------------------------------------
# Summary Display
# ------------------------------------------------------------
st.markdown("## 📊 Summary")
per_share = round(total_amount / len(people), 2)
total_paid = round(sum(p.paid for p in people), 2)
diff = round(total_paid - total_amount, 2)

col1, col2, col3 = st.columns(3)
col1.metric("Total Bill", f"{total_amount:.2f}")
col2.metric("Per Person Share", f"{per_share:.2f}")
col3.metric("Total Paid", f"{total_paid:.2f}", delta=f"{diff:+.2f}")

if abs(diff) > 0.01:
    st.warning(f"⚠️ Total paid ({total_paid:.2f}) differs from bill amount ({total_amount:.2f}). Check your entries!")

# ------------------------------------------------------------
# Who Owes / Who Gets
# ------------------------------------------------------------
st.markdown("### 💵 Who Owes Money (Debtors)")
debtors = [b for b in balances if b["balance"] < -0.005]
if debtors:
    for d in debtors:
        st.write(f"- **{d['name']}** owes **₹{abs(d['balance']):.2f}**")
else:
    st.success("No one owes money!")

st.markdown("### 💰 Who Gets Reimbursed (Creditors)")
creditors = [b for b in balances if b["balance"] > 0.005]
if creditors:
    for c in creditors:
        st.write(f"- **{c['name']}** should receive **₹{c['balance']:.2f}**")
else:
    st.info("No reimbursements required.")

# ------------------------------------------------------------
# Transfers Table
# ------------------------------------------------------------
st.markdown("### 🔁 Suggested Transfers")
if not transfers:
    st.success("All settled! Everyone’s even.")
else:
    tr_df = pd.DataFrame(transfers)
    tr_df.columns = ["Payer", "Receiver", "Amount"]
    st.table(tr_df.style.format({"Amount": "{:.2f}"}))

# ------------------------------------------------------------
# Detailed Per-Person Summary
# ------------------------------------------------------------
st.markdown("### 🧍 Per-Person Summary")
st.dataframe(result_df.style.format({"Paid":"{:.2f}","Share":"{:.2f}","Balance":"{:.2f}",
                                     "Will Give":"{:.2f}","Will Get":"{:.2f}","Net After Transfers":"{:.2f}"}))

# Natural language summary
st.markdown("### 🗣️ Summary in Words")
for _, r in result_df.iterrows():
    if r["Balance"] < -0.005:
        st.write(f"- {r['Name']} paid ₹{r['Paid']:.2f}, share ₹{r['Share']:.2f} → owes ₹{abs(r['Balance']):.2f} "
                 f"and will pay a total of ₹{r['Will Give']:.2f}.")
    elif r["Balance"] > 0.005:
        st.write(f"- {r['Name']} paid ₹{r['Paid']:.2f}, share ₹{r['Share']:.2f} → should receive ₹{r['Balance']:.2f} "
                 f"and will get ₹{r['Will Get']:.2f}.")
    else:
        st.write(f"- {r['Name']} paid exactly their share ₹{r['Share']:.2f} — no dues.")

# ------------------------------------------------------------
# Download Options
# ------------------------------------------------------------
st.markdown("### 📥 Download Results")
csv_summary = result_df.to_csv(index=False)
st.download_button("Download Summary CSV", data=csv_summary, file_name="expense_summary.csv", mime="text/csv")

if transfers:
    csv_transfers = pd.DataFrame(transfers).to_csv(index=False)
    st.download_button("Download Transfers CSV", data=csv_transfers, file_name="transfers.csv", mime="text/csv")

# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
st.markdown("---")
st.info("""
✅ **How it works:**
- Each person’s equal share = total / number of people.
- “Balance” = Paid − Share.
- Positive balance → person overpaid (gets reimbursed).
- Negative balance → person underpaid (owes money).
- Transfers list shows who pays whom to settle all balances.
- You can freely edit any name or amount — everything updates instantly.
""")

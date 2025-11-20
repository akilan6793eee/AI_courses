"""
R+T+C+F+T (Role + Task + Context + Few-shot + Tone/Style)
{
  "role": "You are a skilled Python developer specializing in Streamlit apps.",
  "task": "Build a Gym Workout Logger that collects exercise entries (exercise name, sets, reps, weight), stores them in a table, and shows a weekly progress graph.",
  "context": "Users will log workouts during a session; data should persist during the Streamlit session (use session_state) and be exportable as CSV. The UI must be simple, mobile-friendly and allow quick entry and quick overview of weekly progress.",
  "fewshot": [
    {"input": {"exercise": "Squat", "sets": 3, "reps": 5, "weight": 100}, "output": "Row added with timestamp; weekly chart shows max weight by day"},
    {"input": {"exercise": "Bench Press", "sets": 4, "reps": 8, "weight": 70}, "output": "Row added; export CSV contains new row"}
  ],
  "tone_style": "Clear, helpful, friendly, and concise with light emoji use"
}
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Gym Workout Logger", page_icon="🏋️‍♂️", layout="wide")

st.title("🏋️‍♂️ Gym Workout Logger")
st.write("Quickly log your exercises and visualize weekly progress. Data persists during the session and can be exported.")

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=['timestamp', 'exercise', 'sets', 'reps', 'weight'])

# Sidebar: filters and export
with st.sidebar:
    st.header("Filters & Export")
    days = st.selectbox("Show last how many days?", options=[7, 14, 30, 90], index=0)
    exercise_filter = st.text_input("Filter exercise (leave blank for all)")

    st.markdown("---")
    if not st.session_state.logs.empty:
        csv = st.session_state.logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", data=csv, file_name=f"workout_logs_{datetime.now().date()}.csv", mime='text/csv')
    else:
        st.info("No logs to export yet — add an entry!")

# Main layout: entry form and table/visuals
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Add a workout entry")
    with st.form("entry_form", clear_on_submit=True):
        exercise = st.text_input("Exercise name", max_chars=100)
        sets = st.number_input("Sets", min_value=1, max_value=50, value=3, step=1)
        reps = st.number_input("Reps", min_value=1, max_value=100, value=8, step=1)
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=1000.0, value=20.0, step=0.5)
        submitted = st.form_submit_button("➕ Add Entry")

    if submitted:
        if not exercise.strip():
            st.error("Please enter an exercise name.")
        else:
            row = {
                'timestamp': datetime.now(),
                'exercise': exercise.strip(),
                'sets': int(sets),
                'reps': int(reps),
                'weight': float(weight)
            }
            st.session_state.logs = pd.concat([st.session_state.logs, pd.DataFrame([row])], ignore_index=True)
            st.success(f"Added: {row['exercise']} — {row['sets']}x{row['reps']} @ {row['weight']} kg")

    st.markdown("---")
    st.subheader("Quick actions")
    if st.button("Clear all logs"):
        st.session_state.logs = pd.DataFrame(columns=['timestamp', 'exercise', 'sets', 'reps', 'weight'])
        st.info("All logs cleared.")

with col2:
    st.subheader("Workout logs")
    if st.session_state.logs.empty:
        st.info("No logs yet. Add your first workout entry from the left panel.")
    else:
        df = st.session_state.logs.copy()

        # Apply exercise filter
        if exercise_filter.strip():
            df = df[df['exercise'].str.contains(exercise_filter.strip(), case=False, na=False)]

        # Filter by days
        cutoff = datetime.now() - timedelta(days=int(days))
        df = df[df['timestamp'] >= cutoff]

        # Show table
        df_display = df.copy()
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(df_display.sort_values('timestamp', ascending=False).reset_index(drop=True))

        st.markdown("---")
        st.subheader("Weekly progress chart")

        # Prepare data for chart: group by date and exercise, choose max weight per day
        chart_df = df.copy()
        chart_df['date'] = chart_df['timestamp'].dt.date
        chart_df['date'] = pd.to_datetime(chart_df['date'])

        if chart_df.empty:
            st.info("No data to chart for the selected filter/time range.")
        else:
            # Aggregate: for each date & exercise show max weight * reps or just weight depending on preference
            agg = chart_df.groupby(['date', 'exercise']).agg(
                max_weight=('weight', 'max'),
                total_volume=('weight', lambda w: 0),
            ).reset_index()

            # Build an interactive Altair chart showing max weight per day for selected exercises
            selection = alt.selection_multi(fields=['exercise'], bind='legend')

            base = alt.Chart(agg).mark_line(point=True).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('max_weight:Q', title='Max weight (kg)'),
                color='exercise:N',
                tooltip=['date:T', 'exercise:N', 'max_weight:Q']
            ).add_selection(selection).interactive()

            st.altair_chart(base.properties(height=300), use_container_width=True)

        st.markdown("---")
        st.subheader("Summary stats")
        summary = df.groupby('exercise').agg(
            sessions=('timestamp', 'count'),
            avg_weight=('weight', 'mean'),
            max_weight=('weight', 'max')
        ).reset_index()
        summary['avg_weight'] = summary['avg_weight'].round(2)
        st.table(summary.sort_values('sessions', ascending=False))

# Footer: tips
st.markdown("---")
st.write("Tips: Use the exercise filter to focus charts. Use the CSV export to save your data or import into other apps.")

# Keep session_state.logs as a pandas DataFrame with proper dtypes on reload
if isinstance(st.session_state.logs, pd.DataFrame):
    # Ensure timestamp column is datetime
    if not st.session_state.logs.empty and not pd.api.types.is_datetime64_any_dtype(st.session_state.logs['timestamp']):
        try:
            st.session_state.logs['timestamp'] = pd.to_datetime(st.session_state.logs['timestamp'])
        except Exception:
            pass

# EOF

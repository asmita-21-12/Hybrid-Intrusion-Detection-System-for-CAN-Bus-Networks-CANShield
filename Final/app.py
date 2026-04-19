import streamlit as st
import pandas as pd
import time
import threading
import queue
import plotly.graph_objects as go
from collections import deque
from preprocessing import load_and_preprocess_data, ATTACK_LABELS
from feature_engineering import create_features
from model import train_models
from realtime_simulation import simulate_realtime, FEATURE_COLUMNS
import explainability as expl
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = 'data'
DEFAULT_SAMPLE = 'sample_data.csv'

data_queue = queue.Queue()

def load_data():
    df = load_and_preprocess_data(DATA_DIR, sample_path=DEFAULT_SAMPLE)
    df = create_features(df)
    return df


def train_system(df):
    feature_columns = FEATURE_COLUMNS
    X = df[feature_columns].fillna(0).values
    y = df['label'].values
    models = train_models(X, y)
    return models, feature_columns

st.title('Advanced Hybrid IDS for CAN Bus Networks')
st.sidebar.header('Simulation Controls')
speed = st.sidebar.selectbox('Simulation speed', ['Slow', 'Normal', 'Fast'], index=1)
replay_now = st.sidebar.button('Replay attack scenario')
alert_severities = st.sidebar.multiselect('Alert severity filter', ['HIGH', 'MEDIUM', 'LOW'], default=['HIGH', 'MEDIUM', 'LOW'])

speed_map = {'Slow': 0.8, 'Normal': 0.3, 'Fast': 0.05}
delay = speed_map[speed]

if 'packets' not in st.session_state:
    st.session_state.packets = deque(maxlen=200)
    st.session_state.alerts = []

if replay_now:
    data_queue = queue.Queue()
    st.session_state.packets = deque(maxlen=200)
    st.session_state.alerts = []

with st.spinner('Loading data and training models...'):
    df = load_data()
    models, feature_columns = train_system(df)
    rf_model = models['rf']
    xgb_model = models['xgb']
    iso_model = models['iso']

st.markdown(f'**Dataset rows:** {len(df)}')
st.markdown(f'**Attack classes:** {len(ATTACK_LABELS)-1} + Normal')
if xgb_model is not None:
    st.markdown('- XGBoost enabled for advanced detection.')
else:
    st.markdown('- XGBoost unavailable; using Random Forest only.')

known_ids = set(df[df['label'] == 0]['can_id'].unique())

packet_placeholder = st.empty()
alert_placeholder = st.empty()
source_placeholder = st.empty()
severity_placeholder = st.empty()

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

timeline_placeholder = col1.empty()
gauge_placeholder = col2.empty()
importance_placeholder = col3.empty()
suspicious_placeholder = col4.empty()

if 'simulation_thread' not in st.session_state or not st.session_state.simulation_thread.is_alive():
    def simulation_worker():
        for packet in simulate_realtime(df, known_ids, rf_model, xgb_model, iso_model, delay=delay, batch_size=32):
            data_queue.put(packet)

    st.session_state.simulation_thread = threading.Thread(target=simulation_worker, daemon=True)
    st.session_state.simulation_thread.start()

chart_iteration = 0
while True:
    try:
        packet = data_queue.get(timeout=1)
        st.session_state.packets.append(packet)
        if packet['attack']:
            st.session_state.alerts.append(packet)
            pd.DataFrame(st.session_state.alerts).to_csv('alerts.csv', index=False)

        packet_df = pd.DataFrame(list(st.session_state.packets))
        packet_placeholder.dataframe(packet_df.tail(50), width='stretch')

        if st.session_state.alerts:
            filtered_alerts = [a for a in st.session_state.alerts if a['severity'] in alert_severities]
            alert_text = '🚨 Latest Alerts:\n' + '\n'.join([
                f"{a['timestamp']}: {a['attack_type']} ({a['severity']}) - {a['reason']}" for a in filtered_alerts[-5:]
            ])
            alert_placeholder.text(alert_text)
        else:
            alert_placeholder.text('No attacks detected yet.')

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=packet_df['timestamp'],
            y=packet_df['confidence'],
            mode='lines+markers',
            name='Confidence'
        ))
        fig_timeline.update_layout(title='Attack Timeline', xaxis_title='Time', yaxis_title='Confidence')
        timeline_placeholder.plotly_chart(fig_timeline, width='stretch', key=f'timeline_chart_{chart_iteration}')

        latest_confidence = packet['confidence'] * 100
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=latest_confidence,
            title={'text': 'Latest Confidence (%)'},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'darkred'}}
        ))
        gauge_placeholder.plotly_chart(fig_gauge, width='stretch', key=f'gauge_chart_{chart_iteration}')

        importance = expl.get_feature_importance(rf_model, feature_columns, top_n=8)
        importance_df = pd.DataFrame(importance, columns=['feature', 'importance'])
        fig_imp = go.Figure(go.Bar(x=importance_df['importance'], y=importance_df['feature'], orientation='h'))
        fig_imp.update_layout(title='Feature Importance', yaxis_title='Feature')
        importance_placeholder.plotly_chart(fig_imp, width='stretch', key=f'importance_chart_{chart_iteration}')

        suspicious_counts = packet_df[packet_df['attack']].groupby('attack_type').size().sort_values(ascending=False)
        suspicious_placeholder.table(suspicious_counts.reset_index(name='count'))

        source_counts = packet_df.groupby('source').size().to_dict()
        source_placeholder.metric('Detection source', ', '.join([f'{k}:{v}' for k, v in source_counts.items()]))
        severity_counts = packet_df.groupby('severity').size().to_dict()
        severity_placeholder.markdown('**Severity:** ' + ', '.join([f'{k}:{v}' for k, v in severity_counts.items()]))

        chart_iteration += 1
        time.sleep(0.1)
    except queue.Empty:
        break

if len(df) > 0:
    try:
        y_true = df['label'].values
        y_pred = rf_model.predict(df[feature_columns].fillna(0).values)
        unique_labels = sorted(set(y_true) | set(y_pred))
        target_names = [ATTACK_LABELS[i] for i in unique_labels]
        report = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        st.subheader('Evaluation Metrics')
        st.text(report)
        st.write('Confusion Matrix')
        st.write(pd.DataFrame(cm, index=target_names, columns=target_names))
    except Exception as exc:
        st.write('Evaluation metrics unavailable.')
        st.write(f'Error: {exc}')

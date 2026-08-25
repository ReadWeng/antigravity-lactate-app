import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import glob
from datetime import datetime, timedelta
import re
import base64
import streamlit.components.v1 as components

try:
    import fitparse
except ImportError:
    fitparse = None

try:
    import fitdecode
except ImportError:
    fitdecode = None

# 設定頁面標題與佈局
st.set_page_config(
    page_title="FIT & CSV (Final LA/10) 協同分析工具",
    page_icon="🩸",
    layout="wide"
)

# 自訂 CSS 樣式
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stApp { max-width: 1400px; margin: 0 auto; }
.title-container {
    background: linear-gradient(90deg, #1e2638 0%, #0d1b2a 100%);
    padding: 24px;
    border-radius: 12px;
    border-left: 6px solid #00e676;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.subtitle-text { color: #a0aec0; font-size: 1.1rem; margin-top: 6px; }
.metric-card {
    background-color: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #00e676; }
.metric-label { font-size: 0.9rem; color: #a0aec0; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ----------------- FIT 與 CSV 解析邏輯 -----------------

@st.cache_data(show_spinner=False)
def parse_fit_file_data(uploaded_file_bytes):
    """
    解析 FIT 檔案的 records 數據，轉為 UTC+8 時區。
    """
    records = []
    laps_list = []
    use_fallback = False
    
    if fitparse is not None:
        try:
            fit_file = fitparse.FitFile(io.BytesIO(uploaded_file_bytes))
            for record in fit_file.get_messages('record'):
                vals = {field.name: field.value for field in record.fields}
                records.append(vals)
            for i, lap in enumerate(fit_file.get_messages('lap')):
                vals = {field.name: field.value for field in lap.fields}
                laps_list.append(vals)
        except Exception:
            use_fallback = True
    else:
        use_fallback = True
        
    if use_fallback and fitdecode is not None:
        records = []
        laps_list = []
        try:
            with fitdecode.FitReader(io.BytesIO(uploaded_file_bytes)) as fit:
                for frame in fit:
                    if frame.frame_type == fitdecode.FIT_FRAME_DATA:
                        if frame.name == "record":
                            row = {field.name: field.value for field in frame.fields}
                            records.append(row)
                        elif frame.name == "lap":
                            row = {field.name: field.value for field in frame.fields}
                            laps_list.append(row)
        except Exception:
            return pd.DataFrame(), None
            
    df = pd.DataFrame(records)
    if df.empty or 'timestamp' not in df.columns:
        return pd.DataFrame(), None
        
    # 排序並取得開始時間，加上 8 小時轉換為 UTC+8
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=8)
        
    start_time = df['timestamp'].iloc[0]
    df['elapsed_minutes'] = (df['timestamp'] - start_time).dt.total_seconds() / 60.0
    
    for col in ['heart_rate', 'power', 'temperature', 'skin_temperature', 'core_temperature']:
        if col not in df.columns:
            df[col] = np.nan
            
    # 替代心率欄位處理
    if df['heart_rate'].isna().all():
        for alt_hr in ['heartrate', 'hr', 'HeartRate', 'Heart_Rate']:
            if alt_hr in df.columns and df[alt_hr].notna().any():
                df['heart_rate'] = df[alt_hr]
                break
                
    # 核心溫度處理
    if 'core_temperature' in df.columns and df['core_temperature'].notna().any():
        df['core_temp'] = df['core_temperature'].apply(lambda x: x / 100.0 if (pd.notna(x) and x > 1000) else x)
    elif 'unknown_139' in df.columns:
        df['core_temp'] = df['unknown_139'].apply(lambda x: x / 100.0 if (pd.notna(x) and x > 1000) else x)
    else:
        df['core_temp'] = np.nan
        
    df_clean = df[['timestamp', 'elapsed_minutes', 'heart_rate', 'power', 'core_temp', 'skin_temperature', 'temperature']].copy()
    return df_clean, start_time

@st.cache_data(show_spinner=False)
def parse_device_csv_lactate(csv_source):
    """
    解析生化儀器產出的 CSV 檔 (含 Year, Month, Day, Hour, Minute, Final LA)。
    自動提取 Final LA / 10.0 作為乳酸值 (mmol/L)。
    """
    encodings_to_try = ['cp950', 'big5', 'utf-8-sig', 'latin1', 'gbk']
    df_csv = None
    
    for enc in encodings_to_try:
        try:
            if isinstance(csv_source, bytes):
                df_csv = pd.read_csv(io.BytesIO(csv_source), encoding=enc, index_col=False)
            else:
                df_csv = pd.read_csv(csv_source, encoding=enc, index_col=False)
            if 'Year' in df_csv.columns and 'Final LA' in df_csv.columns:
                break
        except Exception:
            continue
            
    if df_csv is None or df_csv.empty:
        return pd.DataFrame()
        
    records = []
    for idx, r in df_csv.iterrows():
        try:
            if pd.isna(r.get('Year')) or pd.isna(r.get('Final LA')):
                continue
            yr = int(r['Year'])
            yr = 2000 + yr if yr < 100 else yr
            mo = int(r['Month'])
            dy = int(r['Day'])
            hr = int(r['Hour'])
            mn = int(r['Minute'])
            
            la_raw = float(r['Final LA'])
            la_val = round(la_raw / 10.0, 2)
            
            dt = datetime(yr, mo, dy, hr, mn)
            records.append({
                'csv_datetime': dt,
                'year': yr, 'month': mo, 'day': dy, 'hour': hr, 'minute': mn,
                'final_la_raw': la_raw,
                'lactate': la_val,
                'source': f"CSV (Record {r.get('Record', idx+1)})"
            })
        except Exception:
            pass
            
    return pd.DataFrame(records)

def match_csv_records_to_fit(df_fit, start_time, df_csv):
    """
    將 CSV 中的測量時間 (Year/Month/Day/Hour/Minute) 與 FIT 檔的時間進行對齊，
    計算相對時間 elapsed_minutes，並填入乳酸數據表格。
    """
    if df_csv.empty or start_time is None or df_fit.empty:
        return pd.DataFrame(columns=['相對時間 (分鐘)', '乳酸值 (mmol/L)', '血糖值 (mg/dL)', '來源'])
        
    end_time = df_fit['timestamp'].iloc[-1]
    
    # 允許在活動前 60 分鐘至活動後 60 分鐘內的測量點對齊
    window_start = start_time - timedelta(minutes=60)
    window_end = end_time + timedelta(minutes=60)
    
    matched = df_csv[(df_csv['csv_datetime'] >= window_start) & (df_csv['csv_datetime'] <= window_end)].copy()
    
    if matched.empty:
        # 如果時間範圍無重疊，則放寬按「同月同日」過濾
        fit_date = start_time.date()
        matched = df_csv[df_csv['csv_datetime'].dt.date == fit_date].copy()
        
    if matched.empty:
        return pd.DataFrame(columns=['相對時間 (分鐘)', '乳酸值 (mmol/L)', '血糖值 (mg/dL)', '來源'])
        
    matched['elapsed_minutes'] = (matched['csv_datetime'] - start_time).dt.total_seconds() / 60.0
    matched['elapsed_minutes'] = matched['elapsed_minutes'].round(1)
    
    matched_df = pd.DataFrame({
        '相對時間 (分鐘)': matched['elapsed_minutes'],
        '乳酸值 (mmol/L)': matched['lactate'],
        '血糖值 (mg/dL)': [None] * len(matched),
        '來源': matched['source']
    })
    
    return matched_df.sort_values(by='相對時間 (分鐘)').reset_index(drop=True)

# ----------------- HTML 報告生成器 -----------------

def generate_html_report(summary_df, fig, start_time, fit_filename, stats_dict):
    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    table_rows_html = ""
    for idx, row in summary_df.iterrows():
        table_rows_html += f"""
        <tr>
            <td>{row['量測時間 (分)']}</td>
            <td><strong>{row['乳酸值 (mmol/L)']}</strong></td>
            <td><strong>{row.get('血糖值 (mg/dL)', '-')}</strong></td>
            <td>{row['對應功率 (W)']}</td>
            <td>{row['對應心率 (BPM)']}</td>
            <td>{row['對應核心溫度 (°C)']}</td>
            <td>{row['量測點來源']}</td>
            <td>{row['實際時間 (Time)']}</td>
        </tr>
        """

    html_template = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>乳酸與生理指標單期分析報告 - {start_time.strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #0e1117;
            color: #e0e6ed;
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1280px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1e2638 0%, #0d1b2a 100%);
            padding: 24px;
            border-radius: 12px;
            border-left: 6px solid #00e676;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 1.8rem; color: #ffffff; }}
        .header .meta {{ color: #a0aec0; font-size: 0.95rem; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background-color: #1a202c;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .kpi-val {{ font-size: 1.6rem; font-weight: bold; color: #00e676; margin-top: 4px; }}
        .kpi-lbl {{ font-size: 0.85rem; color: #a0aec0; }}
        .chart-container {{
            background-color: #1a202c;
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #1a202c;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 16px;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px 16px;
            text-align: center;
            border-bottom: 1px solid #2d3748;
        }}
        .summary-table th {{ background-color: #2d3748; color: #e2e8f0; font-weight: 600; }}
        .summary-table tr:hover {{ background-color: #283141; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🩸 乳酸與生理指標單期分析報告 (fitANDcsvLA)</h1>
            <div class="meta">
                <span>📅 活動日期：{start_time.strftime('%Y-%m-%d')}</span> | 
                <span>⏰ 開始時間：{start_time.strftime('%H:%M:%S')} (UTC+8)</span> | 
                <span>📁 FIT 檔案：{fit_filename}</span>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-lbl">總運動時間</div>
                <div class="kpi-val">{stats_dict.get('duration_str', '-')}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">平均 / 最大功率</div>
                <div class="kpi-val">{stats_dict.get('avg_power', '-')} / {stats_dict.get('max_power', '-')} W</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">平均 / 最大心率</div>
                <div class="kpi-val">{stats_dict.get('avg_hr', '-')} / {stats_dict.get('max_hr', '-')} BPM</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">最高核心溫度</div>
                <div class="kpi-val">{stats_dict.get('max_core', '-')}</div>
            </div>
        </div>

        <div class="chart-container">
            {chart_html}
        </div>

        <h3>📋 乳酸 / 血糖數據與生理指標對照表</h3>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>量測時間 (分)</th>
                    <th>乳酸值 (mmol/L)</th>
                    <th>血糖值 (mg/dL)</th>
                    <th>對應功率 (W)</th>
                    <th>對應心率 (BPM)</th>
                    <th>對應核心溫度 (°C)</th>
                    <th>量測點來源</th>
                    <th>實際時間 (Time)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    return html_template

# ----------------- Streamlit UI 主程式 -----------------

st.markdown('<div class="title-container">🩸 FIT 檔與生化檢測 CSV (fitANDcsvLA) 協同分析工具</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">上傳 FIT 運動檔案與儀器 CSV 檔，系統將自動依據 Year/Month/Day/Hour/Minute 自動抓取並計算 Final LA / 10 乳酸對照數據。</div>', unsafe_allow_html=True)

# 側邊欄設定
st.sidebar.header("📁 檔案載入與設定")
uploaded_fit = st.sidebar.file_uploader("1. 上傳運動 .fit 檔案", type=["fit"])

default_csv_path = r"c:\Users\ReadWeng\Documents\antigravity\RunDataDayu\record_20260813 155812-14.csv"
use_default_csv = False

st.sidebar.markdown("---")
st.sidebar.markdown("**2. 生化檢測 CSV 檔案**")
uploaded_csv = st.sidebar.file_uploader("上傳儀器 CSV 檔案 (.csv)", type=["csv"])

if not uploaded_csv and os.path.exists(default_csv_path):
    use_default_csv = st.sidebar.checkbox("使用預設 CSV 檔案 (RunDataDayu)", value=True)

smooth_power = st.sidebar.checkbox("顯示 30 秒平滑功率", value=True)

if uploaded_fit is None:
    st.info("👈 請於左側邊欄上傳 `.fit` 運動檔案以開始分析。")
else:
    # 1. 解析 FIT 檔
    fit_bytes = uploaded_fit.read()
    file_name = uploaded_fit.name
    df, start_time = parse_fit_file_data(fit_bytes)
    
    if df.empty or start_time is None:
        st.error("FIT 檔解析失敗，請確認檔案格式是否正確。")
    else:
        duration_min = df['elapsed_minutes'].iloc[-1]
        duration_str = f"{int(duration_min // 60)}h {int(duration_min % 60)}m {int((duration_min*60)%60)}s" if duration_min >= 60 else f"{int(duration_min)}m {int((duration_min*60)%60)}s"
        
        avg_power = int(df['power'].mean()) if df['power'].notna().any() else "-"
        max_power = int(df['power'].max()) if df['power'].notna().any() else "-"
        avg_hr = int(df['heart_rate'].mean()) if df['heart_rate'].notna().any() else "-"
        max_hr = int(df['heart_rate'].max()) if df['heart_rate'].notna().any() else "-"
        
        has_core_temp = 'core_temp' in df.columns and df['core_temp'].notna().any()
        max_core = f"{df['core_temp'].max():.2f} °C" if has_core_temp else "無"
        
        # 2. 顯示頂部 KPI 卡片
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{duration_str}</div><div class="metric-label">總運動時間</div></div>', unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_power} / {max_power} W</div><div class="metric-label">平均 / 最大功率</div></div>', unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_hr} / {max_hr} BPM</div><div class="metric-label">平均 / 最大心率</div></div>', unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{max_core}</div><div class="metric-label">最高核心溫度</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**活動開始時間**：`{start_time.strftime('%Y-%m-%d %H:%M:%S')}` (UTC+8)")
        
        # 3. 自動解析與對齊 CSV 數據
        auto_matched_df = pd.DataFrame()
        csv_source_name = ""
        
        if uploaded_csv is not None:
            csv_bytes = uploaded_csv.read()
            df_csv = parse_device_csv_lactate(csv_bytes)
            csv_source_name = uploaded_csv.name
            auto_matched_df = match_csv_records_to_fit(df, start_time, df_csv)
        elif use_default_csv and os.path.exists(default_csv_path):
            df_csv = parse_device_csv_lactate(default_csv_path)
            csv_source_name = os.path.basename(default_csv_path)
            auto_matched_df = match_csv_records_to_fit(df, start_time, df_csv)
            
        st.subheader("🧪 乳酸與血糖測試數據 (支援自動抓取與手動微調)")
        if not auto_matched_df.empty:
            st.success(f"✓ 已自動從 `{csv_source_name}` 抓取與 FIT 檔同時間點的 `Final LA / 10` 乳酸數據 (共 {len(auto_matched_df)} 筆)！")
        else:
            st.info("💡 未偵測到同時間點的 CSV 乳酸數據，您可以在下表中手動新增或微調數據點。")

        # 初始資料表
        initial_editor_df = auto_matched_df if not auto_matched_df.empty else pd.DataFrame({
            '相對時間 (分鐘)': [0.0, 10.0, 20.0],
            '乳酸值 (mmol/L)': [1.5, 2.0, 4.0],
            '血糖值 (mg/dL)': [None, None, None],
            '來源': ['手動登錄'] * 3
        })

        edited_df = st.data_editor(
            initial_editor_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                '相對時間 (分鐘)': st.column_config.NumberColumn("相對時間 (分鐘)", help="相對於運動開始的時間（分鐘），支援負數", step=0.5, format="%.1f"),
                '乳酸值 (mmol/L)': st.column_config.NumberColumn("乳酸值 (mmol/L)", help="乳酸濃度，如 Final LA / 10.0", step=0.1, format="%.2f"),
                '血糖值 (mg/dL)': st.column_config.NumberColumn("血糖值 (mg/dL)", help="血糖濃度", step=1, format="%d"),
                '來源': st.column_config.TextColumn("來源", help="數據來源說明")
            },
            key="lac_editor_table"
        )
        
        # 清理並過濾有效點
        valid_lac = edited_df.dropna(subset=['相對時間 (分鐘)']).copy()
        valid_lac = valid_lac[(valid_lac['乳酸值 (mmol/L)'].notna()) | (valid_lac['血糖值 (mg/dL)'].notna())]
        valid_lac = valid_lac.sort_values(by='相對時間 (分鐘)').reset_index(drop=True)
        
        # 4. 建立多層 Subplot 圖表
        show_temp_panel = has_core_temp
        row_heights = [0.55, 0.45] if not show_temp_panel else [0.45, 0.25, 0.30]
        specs = [[{"secondary_y": True}], [{"secondary_y": True}]] if not show_temp_panel else [[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]]
        subplot_titles = ("功率 (Power) & 心率 (HR)", "乳酸 (Lactate) & 血糖 (Glucose)") if not show_temp_panel else ("功率 (Power) & 心率 (HR)", "核心體溫 (Core Temp)", "乳酸 (Lactate) & 血糖 (Glucose)")
        
        fig = make_subplots(
            rows=len(row_heights), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=row_heights,
            specs=specs,
            subplot_titles=subplot_titles
        )
        
        # --- 第一層：功率與心率 ---
        fig.add_trace(
            go.Scatter(
                x=df['elapsed_minutes'],
                y=df['power'],
                name="功率 (W)",
                line=dict(color="rgba(0, 176, 255, 0.25)", width=1),
                hoverinfo="skip" if smooth_power else "all"
            ),
            row=1, col=1, secondary_y=False
        )
        
        if smooth_power and df['power'].notna().any():
            df['power_smoothed'] = df['power'].rolling(window=30, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['elapsed_minutes'],
                    y=df['power_smoothed'],
                    name="功率 (30s 平均)",
                    line=dict(color="#00b0ff", width=2),
                ),
                row=1, col=1, secondary_y=False
            )
            
        fig.add_trace(
            go.Scatter(
                x=df['elapsed_minutes'],
                y=df['heart_rate'],
                name="心率 (BPM)",
                line=dict(color="#ff2a5f", width=1.5),
            ),
            row=1, col=1, secondary_y=True
        )
        
        # --- 第二層：核心溫度 (如有) ---
        if show_temp_panel:
            fig.add_trace(
                go.Scatter(
                    x=df['elapsed_minutes'],
                    y=df['core_temp'],
                    name="核心溫度 (°C)",
                    line=dict(color="#ff9100", width=2.5),
                    connectgaps=True
                ),
                row=2, col=1
            )
            
        # --- 乳酸與血糖對照圖 ---
        lactate_row = 3 if show_temp_panel else 2
        if not valid_lac.empty:
            if '乳酸值 (mmol/L)' in valid_lac.columns and valid_lac['乳酸值 (mmol/L)'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=valid_lac['相對時間 (分鐘)'],
                        y=valid_lac['乳酸值 (mmol/L)'],
                        name="乳酸 (mmol/L)",
                        mode="lines+markers",
                        marker=dict(size=10, color="#00e676", symbol="diamond", line=dict(color="white", width=1.5)),
                        line=dict(color="#00e676", width=2.5, dash="dash"),
                        text=valid_lac['來源'],
                        hovertemplate="時間: %{x:.1f} 分<br>乳酸: %{y:.2f} mmol/L<br>來源: %{text}<extra></extra>"
                    ),
                    row=lactate_row, col=1, secondary_y=False
                )
                
            if '血糖值 (mg/dL)' in valid_lac.columns and valid_lac['血糖值 (mg/dL)'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=valid_lac['相對時間 (分鐘)'],
                        y=valid_lac['血糖值 (mg/dL)'],
                        name="血糖 (mg/dL)",
                        mode="lines+markers",
                        marker=dict(size=10, color="#d500f9", symbol="circle", line=dict(color="white", width=1.5)),
                        line=dict(color="#d500f9", width=2.5, dash="dot"),
                        text=valid_lac['來源'],
                        hovertemplate="時間: %{x:.1f} 分<br>血糖: %{y:.1f} mg/dL<br>來源: %{text}<extra></extra>"
                    ),
                    row=lactate_row, col=1, secondary_y=True
                )
                
            for idx, r in valid_lac.iterrows():
                fig.add_vline(
                    x=r['相對時間 (分鐘)'],
                    line_width=1,
                    line_dash="dash",
                    line_color="rgba(255, 255, 255, 0.35)",
                    row="all", col=1
                )
                
        fig.update_layout(
            height=750,
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=60, t=40, b=40),
            plot_bgcolor='rgba(30, 30, 38, 0.4)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_text="相對時間 (分鐘)")
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. 生理指標與乳酸摘要對照表
        st.markdown("### 📋 運動軌跡與乳酸數據精確對照表")
        summary_rows = []
        for idx, r in valid_lac.iterrows():
            t = r['相對時間 (分鐘)']
            lac = r.get('乳酸值 (mmol/L)')
            glc = r.get('血糖值 (mg/dL)')
            source = r['來源']
            
            diffs = (df['elapsed_minutes'] - t).abs()
            nearest_idx = diffs.idxmin()
            nearest = df.iloc[nearest_idx]
            
            calc_time = start_time + timedelta(minutes=float(t))
            is_before_start = (t < 0)
            
            summary_rows.append({
                '量測時間 (分)': round(t, 1),
                '乳酸值 (mmol/L)': round(lac, 2) if pd.notna(lac) else "-",
                '血糖值 (mg/dL)': int(glc) if pd.notna(glc) else "-",
                '量測點來源': source,
                '對應功率 (W)': "-" if is_before_start else (str(int(nearest['power'])) if pd.notna(nearest['power']) else "-"),
                '對應心率 (BPM)': "-" if is_before_start else (str(int(nearest['heart_rate'])) if pd.notna(nearest['heart_rate']) else "-"),
                '對應核心溫度 (°C)': "-" if is_before_start else (f"{nearest['core_temp']:.2f}" if pd.notna(nearest['core_temp']) else "-"),
                '實際時間 (Time)': calc_time.strftime('%H:%M:%S')
            })
            
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)
        
        # 6. 匯出報告
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("💾 儲存與產出分析報告")
        
        html_report_data = generate_html_report(
            summary_df, fig, start_time, file_name,
            {
                'duration_str': duration_str,
                'avg_power': avg_power, 'max_power': max_power,
                'avg_hr': avg_hr, 'max_hr': max_hr,
                'max_core': max_core
            }
        )
        
        save_cols = st.columns(3)
        with save_cols[0]:
            if st.button("💾 儲存報告至本機 saved_reports/", use_container_width=True):
                try:
                    os.makedirs("saved_reports", exist_ok=True)
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    html_path = f"saved_reports/lactate_report_{timestamp_str}.html"
                    csv_path = f"saved_reports/lactate_report_{timestamp_str}.csv"
                    
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_report_data)
                    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    st.success(f"✨ 報告已儲存至 `{html_path}`！")
                except Exception as e:
                    st.error(f"儲存失敗: {e}")
                    
        with save_cols[1]:
            st.download_button(
                label="📥 下載 HTML 網頁報告",
                data=html_report_data,
                file_name=f"lactate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
            
        with save_cols[2]:
            csv_buffer = io.StringIO()
            summary_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下載生理彙整 CSV 檔",
                data=csv_buffer.getvalue(),
                file_name=f"lactate_physiological_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

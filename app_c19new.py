# =============================================================================
# 台灣 COVID-19 儀表板｜學生閱讀版：app_c19n.py
# =============================================================================
# 本檔由 app_c19.py 另存，只增加繁體中文教學註解；原始檔與運算邏輯均保留。
#
# 【啟動方式】在本檔所在資料夾的終端機輸入：
#   streamlit run app_c19n.py
# 資料來源已改為直接讀取 Google 雲端硬碟；程式會自動下載並以快取載入。
# 套件：streamlit、pandas、numpy、plotly、requests。
# 本版使用 Streamlit 1.63.0 支援的 persist_state 等功能。
#
# 【先理解 Streamlit】
# 1. 首次開啟頁面會從上到下執行。def 先定義函式，呼叫時才執行函式內容。
# 2. 左側選單先局部執行，篩選集合真的改變才整頁更新；分頁內控制項仍通常整頁執行。
# 3. cache_data 重用資料處理結果；session_state 保存此使用者工作階段的選擇。
# 4. with tab_x 決定顯示位置，不表示只有點選該分頁才執行裡面的 Python。
# 5. 圖內動畫由 Plotly 在瀏覽器切換影格，不是 Python 用 while 迴圈播放。
#
# 【資料流程】可先閱讀主程式，再回頭研究函式；搜尋「【章節」可快速定位。
# CSV → load_data 清理 → df
#                         └→ common_mask（年份／年齡／來源／性別）
#                              └→ comparison_base
#                                   ├→ 左側縣市篩選 → filtered
#                                   │    └→ 指標、縣市、年齡、動畫、趨勢、下載
#                                   └→ 區域分頁縣市篩選 → comparison_filtered
#                                        └→ 區域堆疊圖、雙層圓餅圖
#
# 【閱讀小字典】
# DataFrame：二維表格；Series：單一欄位。
# mask：每列一個 True／False 條件；df.loc[mask]：保留 True 的列。
# groupby(...).sum()：先分組再加總；transform('sum')：把組內總數對齊回每一列。
# CSV 每列已是彙整紀錄，「人數」要加總確定病例數，不能用 len(df) 的列數代替。
# 型別註記（如 list[str]）是閱讀及工具檢查提示，不是自動輸入驗證。
# =============================================================================

# 【章節 1】匯入套件與設定常數
# 延後解析型別註記；定義資料欄名、順序及分類配色供後面共用。
from __future__ import annotations

# Path 處理路徑；pd 處理表格，px 快速畫圖，go 精細設定動畫及雙層圖。
# sample_colorscale 從漸層指定位置取色，st 則建立網頁與互動控制項。
import io
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import requests
import streamlit as st


# __file__ 是本程式路徑；resolve().parent 取得所在資料夾。
# 因此從不同終端機目錄啟動，仍能找到程式旁的 CSV。
APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "covid-19all.csv"

# 雲端資料來源（Google 雲端硬碟分享連結與直接下載端點）
GDRIVE_FILE_URL = "https://drive.google.com/file/d/1Oss2N3zqn_fVk9PGAPB_GsfA8p0owDa4/view?usp=drive_link"
GDRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1Oss2N3zqn_fVk9PGAPB_GsfA8p0owDa4"

# 集中管理 CSV 欄名，避免後面重複輸入而拼錯；換資料來源時先檢查這些名稱。
CASE_COL = "確定病例數"
YEAR_COL = "發病年份"
WEEK_COL = "發病週別"
CITY_COL = "縣市"
AGE_COL = "年齡層"
SEX_COL = "性別"
IMPORT_COL = "是否為境外移入"

# 字典把選單名稱對應到 Plotly 名稱，例如 Blue 實際使用 Blues 色帶。
CITY_COLOR_SCALES = {
    "Blue": "Blues",
    "Green": "Greens",
    "Oranges": "Oranges",
    "RdPu": "RdPu",
    "YlOrRd": "YlOrRd",
    "YlGnBu": "YlGnBu",
    "PuBuGn": "PuBuGn",
}

# 年齡層是分類而非連續數字，不能照字串排序（10~14 會跑到 2 前面）。
# 此清單明列 0、1、2、3、4、5~9、10~14…，保留 CSV 原本的分組定義。
AGE_ORDER = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5~9",
    "10~14",
    "15~19",
    "20~24",
    "25~29",
    "30~34",
    "35~39",
    "40~44",
    "45~49",
    "50~54",
    "55~59",
    "60~64",
    "65~69",
    "70+",
]

# 這是選單優先順序，不是病例排名；病例排名在篩選後另外計算。
CITY_ORDER = [
    "台北市",
    "新北市",
    "桃園市",
    "台中市",
    "台南市",
    "高雄市",
    "基隆市",
    "新竹市",
    "嘉義市",
    "新竹縣",
    "苗栗縣",
    "彰化縣",
    "南投縣",
    "雲林縣",
    "嘉義縣",
    "屏東縣",
    "宜蘭縣",
    "花蓮縣",
    "台東縣",
    "澎湖縣",
    "金門縣",
    "連江縣",
]

# 本儀表板分區：北中南東參考國發會，澎湖另與金門、連江列為離島。
# 五大都市與西部是快速篩選組合，不加入互斥分區加總。
REGION_COL = "區域"
REGION_CITIES = {
    "北部縣市": ["台北市", "新北市", "桃園市", "基隆市", "新竹市", "新竹縣", "宜蘭縣"],
    "中部縣市": ["台中市", "苗栗縣", "彰化縣", "南投縣", "雲林縣"],
    "南部縣市": ["台南市", "高雄市", "嘉義市", "嘉義縣", "屏東縣"],
    "東部縣市": ["花蓮縣", "台東縣"],
    "離島縣市": ["澎湖縣", "金門縣", "連江縣"],
}
# 反查字典：把「區域 → 縣市清單」轉成「縣市 → 區域」。
# 雙層推導式先走訪每個區域，再走訪其中每個縣市。
CITY_TO_REGION = {
    city: region for region, cities in REGION_CITIES.items() for city in cities
}
# 快速選單可重疊（五大都市／西部），但正式統計分區必須互斥，避免重複加總。
# ** 展開另一個字典；西部用清單推導式組合各縣市，再排除宜蘭。
REGION_PRESETS = {
    "五大都市": ["台北市", "桃園市", "台中市", "台南市", "高雄市"],
    **{region: cities for region, cities in REGION_CITIES.items() if region != "離島縣市"},
    "西部縣市": [
        city
        for region in ["北部縣市", "中部縣市", "南部縣市"]
        for city in REGION_CITIES[region]
        if city != "宜蘭縣"
    ],
    "離島縣市": REGION_CITIES["離島縣市"],
}
REGION_COLORS = {
    "北部縣市": "#167d91",
    "中部縣市": "#68b984",
    "南部縣市": "#e5b24b",
    "東部縣市": "#7f58a5",
    "離島縣市": "#4095d1",
    "其他／未分類": "#6e7c86",
}
# 預設色帶取色區間，單位是百分比，不是病例比例；北部偏深、離島偏淺。
# 原始區域配色不套用這些範圍，只有其他漸層配色使用。
REGION_COLOR_RANGES = {
    "北部縣市": (70, 100),
    "中部縣市": (55, 85),
    "南部縣市": (40, 70),
    "東部縣市": (25, 55),
    "離島縣市": (10, 40),
    "其他／未分類": (35, 65),
}

# 顯示文字與資料代碼分開：選單可顯示女性，實際篩選值仍為 F。
SEX_LABELS = {"M": "男性", "F": "女性", "X": "其他", "未註明": "未註明"}
IMPORT_LABELS = {0: "本土", 1: "境外移入"}
# 動畫分類色清單；後面用 %（餘數）循環取色，避免超出清單索引。
COLORS = [
    "#123955",
    "#167d91",
    "#2aa4a1",
    "#68b984",
    "#9ac46a",
    "#e5b24b",
    "#e6814f",
    "#d95d6b",
    "#b14d8d",
    "#7f58a5",
    "#5b6fc0",
    "#4095d1",
    "#855c47",
    "#a27662",
    "#6e7c86",
    "#8d955b",
    "#b88b2b",
    "#c46b3c",
    "#8f3e57",
    "#70477c",
    "#3f5688",
    "#397a86",
    "#638f72",
    "#927f58",
]


# 【章節 2】頁面設定與讀檔函式
# layout='wide' 讓圖表使用寬版面；這裡尚未讀取資料。
st.set_page_config(
    page_title="台灣 COVID-19 確診資料儀表板",
    page_icon="🦠",
    layout="wide",
)


# # 裝飾器快取相同參數的結果，選單改變時不用一直重讀資料。
@st.cache_data(show_spinner="正在從雲端載入確診資料…")
def load_data(url: str = GDRIVE_DOWNLOAD_URL) -> pd.DataFrame:
    """從雲端 Google Drive 下載並載入必要欄位；若雲端讀取失敗則自動嘗試本地快取。"""
    columns = [YEAR_COL, WEEK_COL, CITY_COL, SEX_COL, IMPORT_COL, AGE_COL, CASE_COL]
    encodings = ["cp950", "utf-8", "utf-8-sig", "big5"]

    # 若為 Google Drive 檢視分享連結，自動提取 file_id 並轉為直接下載端點
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    download_url = f"https://drive.google.com/uc?export=download&id={match.group(1)}" if match else url

    data = None
    download_err = None

    # 1. 優先嘗試從雲端下載並解碼
    try:
        resp = requests.get(
            download_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.content

        for enc in encodings:
            try:
                data = pd.read_csv(io.BytesIO(content), usecols=columns, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        # 下載成功後寫入本地備用，斷網時可無縫切換
        if data is not None:
            try:
                CSV_PATH.write_bytes(content)
            except Exception:
                pass
    except Exception as exc:
        download_err = exc

    # 2. 若雲端連線失敗，自動回退嘗試載入本地備份檔案
    if data is None and CSV_PATH.exists():
        for enc in encodings:
            try:
                data = pd.read_csv(CSV_PATH, usecols=columns, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

    if data is None:
        raise RuntimeError(f"無法載入確診資料（雲端下載錯誤：{download_err}），且無可用之本地資料。")

    # errors='coerce' 將無法轉換的值變成缺值；Int16／Int8（大寫 I）容許缺值。
    # 病例數的無效／缺失值依原版視為 0；strip() 清除文字前後空白。
    data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce").astype("Int16")
    data[WEEK_COL] = pd.to_numeric(data[WEEK_COL], errors="coerce").astype("Int8")
    data[CASE_COL] = pd.to_numeric(data[CASE_COL], errors="coerce").fillna(0).astype("int32")
    data[CITY_COL] = data[CITY_COL].astype("string").str.strip()
    data[AGE_COL] = data[AGE_COL].astype("string").str.strip()
    data[SEX_COL] = data[SEX_COL].fillna("未註明").astype("string").str.strip()
    data[IMPORT_COL] = pd.to_numeric(data[IMPORT_COL], errors="coerce").astype("Int8")

    # & 是逐列的「且」：必要欄位不缺、病例數非負才保留。
    # 此處未額外檢查週別是否在 1～53；copy() 回傳獨立資料，避免後續修改影響來源。
    valid = (
        data[YEAR_COL].notna()
        & data[WEEK_COL].notna()
        & data[CITY_COL].notna()
        & data[AGE_COL].notna()
        & (data[CASE_COL] >= 0)
    )
    return data.loc[valid].copy()


# 【章節 3】整理選單順序
# 輸入欄位及優先順序，回傳資料中存在的已知項目＋排序後的未知項目。
# set 去除重複，difference 找出未在優先清單內的新值，避免漏掉新分類。
def ordered_values(values: pd.Series, preferred_order: list[str]) -> list[str]:
    present = set(values.dropna().astype(str).unique())
    known = [value for value in preferred_order if value in present]
    unknown = sorted(present.difference(known))
    return known + unknown


# 【章節 4】整理動畫資料（回傳表格，還不是圖）
# dimension 是縣市或年齡層，categories 是所選分類。
# time_grain 決定每週／年度，value_mode 決定當期／累積。
def build_race_data(
    filtered: pd.DataFrame,
    dimension: str,
    categories: list[str],
    time_grain: str,
    value_mode: str,
    selected_years: list[int],
) -> pd.DataFrame:
    # 年度使用所有選取年份；每週使用篩選資料實際出現的年度、週別。
    # 後面只補「已有時間點 × 所選分類」，不會新增完全沒有出現過的週別。
    if time_grain == "年度":
        period_cols = [YEAR_COL]
        periods = pd.DataFrame({YEAR_COL: sorted(selected_years)})
    else:
        period_cols = [YEAR_COL, WEEK_COL]
        periods = (
            filtered[[YEAR_COL, WEEK_COL]]
            .drop_duplicates()
            .sort_values([YEAR_COL, WEEK_COL])
            .reset_index(drop=True)
        )

    # 依時間＋分類加總病例；as_index=False 讓分組鍵保持為一般欄位。
    # observed=True 對分類型欄位只聚合實際觀察到的組合。
    aggregated = (
        filtered.groupby(period_cols + [dimension], observed=True, as_index=False)[CASE_COL]
        .sum()
    )
    # cross 交叉合併建立所有時間點與分類組合，再 left merge 接上實際數值。
    # 沒有紀錄的組合補 0，避免某分類突然從影格消失；排序後才能正確累積。
    grid = periods.merge(pd.DataFrame({dimension: categories}), how="cross")
    race = grid.merge(aggregated, on=period_cols + [dimension], how="left")
    race[CASE_COL] = race[CASE_COL].fillna(0).astype("int64")
    race = race.sort_values(period_cols + [dimension]).reset_index(drop=True)

    # cumsum 對每個分類各自按時間累加，跨年不歸零。
    # 只累積本次篩選內的資料，不能直接當作疫情開始以來的完整累積。
    if value_mode == "累積病例數":
        race[CASE_COL] = race.groupby(dimension, sort=False)[CASE_COL].cumsum()

    # 影格名稱如 2023-W02；zfill(2) 將週別補成兩位，維持閱讀格式一致。
    if time_grain == "年度":
        race["影格"] = race[YEAR_COL].astype(str)
    else:
        race["影格"] = (
            race[YEAR_COL].astype(str)
            + "-W"
            + race[WEEK_COL].astype(str).str.zfill(2)
        )
    return race


# 【章節 5】建立動畫圖（回傳 go.Figure）
# 內部三個小函式分工：選單幀資料、建立長條 trace、設定 Y 軸。
# 呼叫者提供資料、維度、前 N 名、固定色彩表及時間／數值模式。
def create_bar_race(
    race: pd.DataFrame,
    dimension: str,
    top_n: int,
    color_map: dict[str, str],
    time_grain: str,
    value_mode: str,
) -> go.Figure:
    frame_labels = race["影格"].drop_duplicates().tolist()
    # 所有影格固定同一個 X 軸上限，才不會因縮放誤判長條大小。
    # 乘 1.16 留文字空間；max(1, ...) 避免全 0 時產生零寬座標軸。
    global_max = max(1, int(race[CASE_COL].max() * 1.16))
    frame_duration = 850 if time_grain == "年度" else 210
    transition_duration = 420 if time_grain == "年度" else 120
    # 年齡模式不依病例數排名，而是顯示全部所選年齡並固定順序。
    fixed_age_order = ordered_values(race[dimension], AGE_ORDER) if dimension == AGE_COL else None
    if fixed_age_order is not None:
        top_n = len(fixed_age_order)

    # 依影格名稱取資料：年齡用 reindex 照清單排；縣市用 nlargest 取前 N 名。
    # 縣市再由小到大排列，水平圖的最大值才會在上方。
    def values_for_frame(label: str) -> pd.DataFrame:
        current = race.loc[race["影格"] == label, [dimension, CASE_COL]]
        if fixed_age_order is not None:
            # 顯示全部所選年齡層，每幀使用相同年齡順序，不依病例數取前 N 名。
            return current.set_index(dimension).reindex(fixed_age_order).reset_index()
        return current.nlargest(top_n, CASE_COL).sort_values(CASE_COL, ascending=True)

    # trace 是圖中的一組長條；x 是病例、y 是分類，orientation='h' 表示水平。
    # text 是數字標籤；hovertemplate 設定滑鼠提示，%{...} 由 Plotly 代入。
    # f 字串中的 {{、}} 用來保留大括號給 Plotly，而非讓 Python 先代入。
    def trace_for_frame(label: str) -> go.Bar:
        current = values_for_frame(label)
        return go.Bar(
            x=current[CASE_COL],
            y=current[dimension].astype(str).tolist(),
            orientation="h",
            marker=dict(
                color=[color_map[str(value)] for value in current[dimension]],
                line=dict(color="rgba(255,255,255,0.9)", width=1),
            ),
            text=current[CASE_COL],
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"{dimension}：%{{y}}<br>{value_mode}：%{{x:,.0f}}<extra></extra>",
        )

    # categoryarray 決定分類順序，tickvals/ticktext 明列刻度位置和文字。
    # 年齡的 range 反向，將最小年齡放在頂端；±0.5 留出長條半格邊界。
    def axis_for_frame(current: pd.DataFrame) -> dict:
        # 年齡層的「0、1、2…」是分類標籤，不可讓 Plotly 推斷成數值軸。
        # 每幀更新刻度與範圍，避免留下上一幀的類別或省略年齡標籤。
        labels = current[dimension].astype(str).tolist()
        return dict(
            title=dimension,
            type="category",
            categoryorder="array",
            categoryarray=labels,
            tickmode="array",
            tickvals=labels,
            ticktext=labels,
            range=(
                [len(labels) - 0.5, -0.5]  # 年齡由上到下：0、1、2、3、4、5~9…
                if fixed_age_order is not None else [-0.5, len(labels) - 0.5]
            ),
            autorange=False,
            automargin=True,
            showgrid=False,
        )

    # 先取第一個時間點作初始畫面，再將每個時間點封裝成 go.Frame。
    # 每幀也更新 Y 軸和時間文字，避免殘留上一幀的分類。
    first_label = frame_labels[0]
    first_values = values_for_frame(first_label)
    frames = []
    for label in frame_labels:
        current = values_for_frame(label)
        frames.append(
            go.Frame(
                name=label,
                data=[trace_for_frame(label)],
                layout=go.Layout(
                    yaxis=axis_for_frame(current),
                    annotations=[
                        dict(
                            text=label,
                            x=0.99,
                            y=0.04,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=36, color="rgba(18,57,85,0.28)"),
                            xanchor="right",
                        )
                    ],
                ),
            )
        )

    # data 是首次顯示的圖，frames 是後續可切換的所有影格。
    # xref/yref='paper' 表示相對繪圖區的 0～1 位置，不是病例數座標。
    figure = go.Figure(data=[trace_for_frame(first_label)], frames=frames)
    figure.update_layout(
        height=max(520, 42 * top_n + 210),
        margin=dict(l=20, r=72, t=35, b=115),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans TC, Microsoft JhengHei, sans-serif", color="#243447"),
        showlegend=False,
        xaxis=dict(
            title=value_mode,
            range=[0, global_max],
            tickformat=",.0f",
            gridcolor="rgba(36,52,71,0.10)",
            zeroline=False,
        ),
        yaxis=axis_for_frame(first_values),
        annotations=[
            dict(
                text=first_label,
                x=0.99,
                y=0.04,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=36, color="rgba(18,57,85,0.28)"),
                xanchor="right",
            )
        ],
        # Plotly 播放／暫停按鈕：duration 單位是毫秒，fromcurrent=True 從目前影格接續。
        # redraw=True 讓分類位置更新；這些按鈕不是觸發 Python 重新執行的 st.button。
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0,
                y=-0.16,
                xanchor="left",
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ 播放",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                transition=dict(duration=transition_duration),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Ⅱ 暫停",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        # 圖內時間軸不是 st.slider；每個 step 用影格名稱跳到指定時間。
        sliders=[
            dict(
                active=0,
                x=0.24,
                y=-0.13,
                len=0.76,
                currentvalue=dict(prefix="時間：", font=dict(size=15, color="#123955")),
                pad=dict(t=24, b=6),
                steps=[
                    dict(
                        label=label,
                        method="animate",
                        args=[
                            [label],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=transition_duration),
                                mode="immediate",
                            ),
                        ],
                    )
                    for label in frame_labels
                ],
            )
        ],
    )
    return figure


# 【章節 6】共用樣式與數字格式
# 直接修改傳入的 Figure，再回傳同一物件；只設定外觀，不進行篩選或統計。
def apply_chart_style(fig: go.Figure, *, height: int = 520) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans TC, Microsoft JhengHei, sans-serif", color="#243447"),
        hoverlabel=dict(bgcolor="white", font_size=14),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(36,52,71,0.10)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# 格式 :,.0f 表示加千分位並顯示零位小數，只改文字顯示，不修改數值。
def fmt_number(value: int | float) -> str:
    return f"{value:,.0f}"


# 【交叉熱力圖】先加總與補零，再換算色階；customdata 保留實際病例數。
# fragment 讓本分頁的控制項只更新熱力圖；0–4 歲合併不改動其他圖表。
def build_heatmap_matrix(
    data: pd.DataFrame, cities: list[str], ages: list[str],
    *, merge_infants: bool = True, sort_by_total: bool = True,
) -> pd.DataFrame:
    """沿用已篩選資料；只合併已選幼兒歲數，並補齊沒有紀錄的組合。"""
    source = data[[CITY_COL, AGE_COL, CASE_COL]].copy()
    age_order = ordered_values(pd.Series(ages, dtype="string"), AGE_ORDER)
    if merge_infants:
        infant_map = {str(age): "0~4" for age in range(5)}
        source[AGE_COL] = source[AGE_COL].replace(infant_map)
        age_order = list(dict.fromkeys(infant_map.get(age, age) for age in age_order))
    matrix = source.pivot_table(
        index=CITY_COL, columns=AGE_COL, values=CASE_COL,
        aggfunc="sum", fill_value=0, observed=True,
    ).reindex(index=cities, columns=age_order, fill_value=0).fillna(0)
    if sort_by_total:
        order = matrix.sum(axis=1).sort_values(ascending=False, kind="stable").index
        matrix = matrix.loc[order]
    return matrix


def create_heatmap(
    matrix: pd.DataFrame, *, mode: str = "確診人數", palette: str = "Reds",
    show_values: bool = False,
) -> go.Figure:
    counts = matrix.to_numpy(dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts * 100, totals, out=np.zeros_like(counts), where=totals != 0)
    colorbar = dict(title="確診人數", tickformat=",.0f")
    z = counts
    zmax = max(1.0, float(counts.max()))
    if mode == "縣市內占比（%）":
        z, zmax = shares, 100
        colorbar = dict(title="縣市內占比", ticksuffix="%")
    elif mode == "確診人數（對數色階）":
        z = np.log1p(counts)
        zmax = float(np.log1p(zmax))
        maximum = int(counts.max())
        ticks = sorted({0, maximum, *(10 ** p for p in range(len(str(maximum))) if 10 ** p < maximum)})
        colorbar = dict(
            title="確診人數<br>對數色階", tickvals=np.log1p(ticks),
            ticktext=[f"{value:,}" for value in ticks],
        )
    fig = go.Figure(go.Heatmap(
        x=matrix.columns.tolist(), y=matrix.index.tolist(), z=z,
        zmin=0, zmax=zmax, colorscale=palette, colorbar=colorbar,
        customdata=np.stack([counts, shares], axis=-1),
        xgap=1, ygap=1,
        texttemplate=(
            "%{customdata[1]:.1f}%" if mode == "縣市內占比（%）"
            else "%{customdata[0]:,.0f}"
        ) if show_values else None,
        textfont=dict(size=11),
        hovertemplate=(
            "%{y}<br>年齡層：%{x}<br>確診人數：%{customdata[0]:,.0f}"
            "<br>占該縣市已選年齡層：%{customdata[1]:.2f}%<extra></extra>"
        ),
    ))
    apply_chart_style(fig, height=max(440, 30 * len(matrix) + 160))
    fig.update_layout(title=f"縣市 × 年齡層｜{mode}", margin=dict(l=90, r=30, t=65, b=70))
    fig.update_xaxes(
        title="年齡層", type="category", categoryorder="array",
        categoryarray=matrix.columns.tolist(), side="bottom", tickangle=-35,
        showgrid=False, automargin=True,
    )
    fig.update_yaxes(
        title="縣市", type="category", categoryorder="array",
        categoryarray=matrix.index.tolist(), autorange="reversed", automargin=True,
    )
    return fig


@st.fragment
def render_heatmap(data: pd.DataFrame, cities: list[str], ages: list[str]):
    st.subheader("縣市 × 年齡層交叉熱力圖")
    st.caption("沿用左側所有篩選。顏色呈現病例數或病例組成；未納入人口分母，不代表感染率。")
    mode_col, palette_col, order_col = st.columns(3)
    with mode_col:
        mode = st.selectbox(
            "熱力圖數值", ["確診人數", "確診人數（對數色階）", "縣市內占比（%）"],
            key="heatmap_mode",
        )
    with palette_col:
        palette = st.selectbox("熱力圖配色", ["Reds", *CITY_COLOR_SCALES, "Viridis"], key="heatmap_palette")
    with order_col:
        order = st.selectbox("熱力圖縣市排序", ["病例數由高到低", "預設縣市順序"], key="heatmap_order")
    merge = st.checkbox("合併已選的 0–4 歲（僅熱力圖）", value=True, key="heatmap_merge_infants")
    show_values = st.checkbox("顯示格內數值", value=False, key="heatmap_show_values")
    matrix = build_heatmap_matrix(
        data, cities, ages, merge_infants=merge, sort_by_total=order == "病例數由高到低",
    )
    if merge:
        chosen_infants = [age for age in AGE_ORDER[:5] if age in ages]
        if chosen_infants:
            st.caption(f"「0~4」欄僅加總目前已選的 {'、'.join(chosen_infants)} 歲。")
    if mode == "確診人數（對數色階）":
        st.caption("對數色階加強小病例數的差異；格內數字、滑鼠提示與色條刻度仍顯示實際病例數。")
    elif mode == "縣市內占比（%）":
        st.caption("每列分母為該縣市在目前篩選條件下、已選年齡層的病例合計；合計為 0 時占比以 0 顯示。")
    st.plotly_chart(
        create_heatmap(matrix, mode=mode, palette=CITY_COLOR_SCALES.get(palette, palette), show_values=show_values),
        width="stretch", key="heatmap_chart", config={"displaylogo": False},
    )
    st.caption("無紀錄的縣市 × 年齡層組合補 0；可拖曳放大或使用圖表工具列下載圖片。")
    st.download_button(
        "下載熱力圖病例數矩陣（CSV）", matrix.to_csv().encode("utf-8-sig"),
        file_name="covid19_熱力圖_病例數矩陣.csv", mime="text/csv", key="heatmap_download",
    )


# 【章節 7】主畫面、檔案檢查與實際載入
# 這段 CSS 是原版既有外觀，本教學版照留，不參與資料運算。
# unsafe_allow_html=True 允許固定 HTML/CSS；勿任意拼入不可信的外部輸入。
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f5f9fc);
        border: 1px solid #dce7ef;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 5px 18px rgba(35, 75, 105, 0.06);
    }
    [data-testid="stMetricLabel"] {color: #567086;}
    [data-testid="stMetricValue"] {
        color: #123955;
        font-size: clamp(1.55rem, 2.1vw, 2.15rem);
    }
    .dashboard-note {
        color: #52687a;
        background: #f2f7fa;
        border-left: 4px solid #1f8a9e;
        border-radius: 4px 10px 10px 4px;
        padding: 10px 14px;
        margin: 0.4rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("台灣 COVID-19 確診資料儀表板")
st.caption("依區域、縣市、年齡層、年份、病例來源與性別篩選；區域比較可另選跨區域縣市。")

# 從雲端 Google Drive 載入資料（若雲端下載失敗將自動載入本地備份）
try:
    df = load_data(GDRIVE_FILE_URL)
except Exception as exc:
    st.error(f"資料載入失敗：{exc}。請檢查網路連線或雲端檔案存取權限。")
    st.stop()

# 由乾淨資料建立可選清單，不把年份寫死；all_* 是可選值，selected_* 才是選中值。
all_years = sorted(int(value) for value in df[YEAR_COL].dropna().unique())
all_cities = ordered_values(df[CITY_COL], CITY_ORDER)
all_ages = ordered_values(df[AGE_COL], AGE_ORDER)
all_sexes = [value for value in ["F", "M", "X", "未註明"] if value in set(df[SEX_COL])]
all_import_types = [value for value in [0, 1] if value in set(df[IMPORT_COL].dropna().astype(int))]

# 【章節 8】左側選單與區域 → 縣市連動
# with 決定元件的顯示位置；multiselect 回傳清單，selectbox 回傳單一值。
_rendering_sidebar_in_full_run = True


@st.fragment
def render_sidebar_filters():
    # 選單先局部更新；只有篩選內容改變，才要求整頁重新計算圖表。
    st.header("篩選條件")
    selected_years = st.multiselect("發病年份", all_years, default=all_years)
    selected_region = st.selectbox(
        "區域選單",
        ["全部縣市", *REGION_PRESETS],
        key="region_filter",
        help="切換區域會更新縣市選項，並自動全選該區域縣市；仍可自行縮小縣市範圍。",
    )
    city_options = (
        all_cities
        if selected_region == "全部縣市"
        else [city for city in all_cities if city in REGION_PRESETS[selected_region]]
    )
    # 狀態快照 (區域, 可選縣市) 用來判斷範圍是否真的改變。
    # 只有切換區域範圍才全選；改年份不應把手動移除的縣市重新加回。
    # 在建立 city_filter 元件之前設定它的 session_state，避免修改已建立元件的狀態。
    city_scope = (selected_region, tuple(city_options))
    if st.session_state.get("_city_scope") != city_scope:
        st.session_state["city_filter"] = city_options.copy()
        st.session_state["_city_scope"] = city_scope
    else:
        # 其他篩選改動時保留手動選取（包括清空），只移除已不在選項內的縣市。
        st.session_state["city_filter"] = [
            city for city in st.session_state.get("city_filter", city_options)
            if city in city_options
        ]
    selected_cities = st.multiselect("縣市", city_options, key="city_filter")
    selected_ages = st.multiselect("年齡層", all_ages, default=all_ages)
    # format_func 只改選單文字，回傳值仍為原始代碼，後面才能比對 CSV。
    selected_imports = st.multiselect(
        "病例來源",
        all_import_types,
        default=all_import_types,
        format_func=lambda value: IMPORT_LABELS.get(value, str(value)),
    )
    selected_sexes = st.multiselect(
        "性別",
        all_sexes,
        default=all_sexes,
        format_func=lambda value: SEX_LABELS.get(value, str(value)),
    )
    st.divider()
    st.caption("資料來源：Google 雲端硬碟 (covid-19all.csv)")
    st.caption(f"原始彙整紀錄：{len(df):,} 筆")
    with st.expander("區域分組說明"):
        st.caption("五大都市：台北、桃園、台中、台南、高雄（不含新北）。")
        for region, cities in REGION_CITIES.items():
            st.caption(f"{region}：{'、'.join(cities)}。")
        st.caption("西部縣市：北、中、南部的本島縣市，但不含宜蘭；這是本程式的快速篩選定義。")
        st.caption("切換區域會重新全選該區域；調整其他條件則保留手動縣市選取。")

    filters = (selected_years, selected_cities, selected_ages, selected_imports, selected_sexes)
    # 多選的排列順序不影響篩選結果，用集合比較，避免同內容不同順序造成重繪。
    signature = (selected_region, *(frozenset(values) for values in filters))
    previous = st.session_state.get("_filter_signature")
    if previous != signature:
        # 先保存再 rerun；下一次整頁執行讀到相同快照，不會無限重新執行。
        st.session_state["_active_filters"] = filters
        st.session_state["_filter_signature"] = signature
        if previous is not None and not _rendering_sidebar_in_full_run:
            st.rerun()


with st.sidebar:
    try:
        render_sidebar_filters()
    finally:
        # 全頁執行會接著計算圖表，不必再觸發一次；之後的局部事件才需要 rerun。
        _rendering_sidebar_in_full_run = False

# fragment 局部執行時不依賴回傳值；主程式從工作階段狀態取得已採用的條件。
selected_years, selected_cities, selected_ages, selected_imports, selected_sexes = (
    st.session_state["_active_filters"]
)

# 【章節 9】共用條件、資料分流與摘要指標
# 空清單視為 False，all(...) 確認四種共用條件都有選項。
# 刻意不在此排除空的縣市清單，因為區域分頁仍可獨立選擇跨區域縣市。
if not all([selected_years, selected_ages, selected_imports, selected_sexes]):
    st.warning("年份、年齡層、病例來源與性別至少需保留一個選項。")
    st.stop()

# isin(清單) 為每列產生布林值，& 將四個條件取交集。
# comparison_base 尚未限制地理範圍，filtered 才追加左側縣市。
# 先共用、再分流，才能讓區域比較與其他分頁各用不同地理範圍。
common_mask = (
    df[YEAR_COL].isin(selected_years)
    & df[AGE_COL].isin(selected_ages)
    & df[IMPORT_COL].isin(selected_imports)
    & df[SEX_COL].isin(selected_sexes)
)
comparison_base = df.loc[common_mask]
filtered = comparison_base.loc[comparison_base[CITY_COL].isin(selected_cities)]

if comparison_base.empty:
    st.warning("目前年份、年齡層、病例來源與性別組合沒有資料，請調整左側條件。")
    st.stop()

# city_total 每縣市一列，age_total 每年齡層一列，皆加總病例並降冪排序。
# 後面的 iloc[0] 取第一列，只有確認 filtered 不為空之後才可安全使用。
city_total = (
    filtered.groupby(CITY_COL, observed=True, as_index=False)[CASE_COL]
    .sum()
    .sort_values(CASE_COL, ascending=False)
)
age_total = (
    filtered.groupby(AGE_COL, observed=True, as_index=False)[CASE_COL]
    .sum()
    .sort_values(CASE_COL, ascending=False)
)
total_cases = int(filtered[CASE_COL].sum())
if filtered.empty:
    st.warning("左側縣市篩選沒有資料；仍可在「區域比較」選擇「自行選擇跨區域」。")
else:
    top_city = city_total.iloc[0]
    top_age = age_total.iloc[0]

    # KPI 是摘要指標，這四項只代表左側 filtered，不隨區域分頁獨立篩選改變。
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("篩選後確診人數", fmt_number(total_cases))
    kpi2.metric("涵蓋縣市", f"{filtered[CITY_COL].nunique():,} 個")
    kpi3.metric("病例最多縣市", str(top_city[CITY_COL]))
    kpi4.metric("病例最多年齡層", str(top_age[AGE_COL]))

st.markdown(
    '<div class="dashboard-note">長條圖比較確診人數；圓餅圖呈現區域或年齡組成；交叉熱力圖對照各縣市與年齡層的病例分布。</div>',
    unsafe_allow_html=True,
)

# 【章節 10】分頁容器
# 顯示順序由 st.tabs 清單決定；Python 先處理區域分頁，是為了讓左側無資料時
# 仍能完成獨立區域比較。顯示順序與程式執行順序是兩件事。
tab_city, tab_region, tab_age, tab_heat, tab_race, tab_trend, tab_data = st.tabs(
    ["縣市比較", "區域比較", "年齡結構", "交叉熱力圖", "動態排名", "時間趨勢", "彙整資料"]
)

# 【章節 11】區域比較：選範圍 → 配色 → 彙整 → 左右兩圖
with tab_region:
    st.caption("本分頁可獨立選區域與縣市；年份、年齡層、病例來源及性別仍沿用左側。此處操作不改動其他分頁。")
    comparison_scope = st.selectbox(
        "比較範圍", ["沿用左側縣市", "自行選擇跨區域"], key="comparison_scope",
        help="自行選擇跨區域時，不受左側區域／縣市限制。上方指標仍代表左側篩選。",
    )
    # 沿用左側就限制可選縣市，自行跨區域則開放 all_cities。
    # 底層仍是 comparison_base，故年份、年齡、來源與性別繼續生效。
    comparison_pool = selected_cities if comparison_scope == "沿用左側縣市" else all_cities
    region_options = [
        region for region in REGION_COLORS
        if any(CITY_TO_REGION.get(city, "其他／未分類") == region for city in comparison_pool)
    ]
    # 範圍快照變了才重新選取全部可選區域，否則保留使用者選擇。
    region_scope = (comparison_scope, tuple(region_options))
    if st.session_state.get("_comparison_region_scope") != region_scope:
        st.session_state["comparison_regions"] = region_options.copy()
        st.session_state["_comparison_region_scope"] = region_scope
    chosen_regions = st.multiselect("比較區域", region_options, key="comparison_regions")
    comparison_city_options = [
        city for city in comparison_pool
        if CITY_TO_REGION.get(city, "其他／未分類") in chosen_regions
    ]
    # 同理：區域改變後全選所屬縣市，但範圍不變時保留手動勾選。
    comparison_city_scope = (comparison_scope, tuple(comparison_city_options))
    if st.session_state.get("_comparison_city_scope") != comparison_city_scope:
        st.session_state["comparison_cities"] = comparison_city_options.copy()
        st.session_state["_comparison_city_scope"] = comparison_city_scope
    comparison_cities = st.multiselect(
        "納入比較的縣市", comparison_city_options, key="comparison_cities",
        format_func=lambda city: f"{city}（{CITY_TO_REGION.get(city, '其他／未分類')}）",
        help="選區域後會全選其縣市；可移除部分縣市，或跨不同區域選取。",
    )
    comparison_unit = st.selectbox(
        "圖表比較單位", ["區域合計", "縣市明細"], key="comparison_unit",
        help="區域合計：各區域所選縣市加總。縣市明細：逐一比較所選縣市。",
    )
    # 配色控制集中在展開區；key 是元件唯一身分，與其他分頁的配色選單分開。
    with st.expander("色彩設定", expanded=True):
        comparison_palette = st.selectbox(
            "區域比較配色", ["原始區域配色", *CITY_COLOR_SCALES],
            key="comparison_palette",
            help="同步套用本分頁的長條圖與圓餅圖，不影響其他分頁配色。",
        )
        st.caption("區域合計模式下，同一縣市在堆疊長條與外圈使用相同顏色；顏色表示分類，不代表病例數大小。")
        comparison_color_ranges = {}
        if comparison_palette != "原始區域配色":
            st.caption("各區域可各自選取色帶中的一段：0% 為色帶起點、100% 為終點。拉開兩端增加區域內縣市的色差；移動整段可改變該區域色調。")
            for region, default_range in REGION_COLOR_RANGES.items():
                # key 同時包含配色與區域，讓不同色帶各自記住調整值。
                # persist_state='session' 保留暫時隱藏元件的值，但不是永久存檔。
                range_key = f"comparison_color_range_{comparison_palette}_{region}"
                if region in chosen_regions:
                    comparison_color_ranges[region] = st.slider(
                        f"{region}色帶範圍（%）", min_value=0, max_value=100,
                        value=default_range, step=5, key=range_key, persist_state="session",
                        help="各區域獨立設定，可使用不同或重疊範圍；兩端相同時該區域縣市皆為同色。切換配色或暫時移除區域後，本次工作階段仍保留設定。",
                    )
                    if comparison_color_ranges[region][0] == comparison_color_ranges[region][1]:
                        st.caption(f"{region}目前使用單一顏色，拉開兩端可恢復縣市深淺變化。")
                else:
                    comparison_color_ranges[region] = st.session_state.get(range_key, default_range)
    # 區域內圈取區間中點：(start+end)/2/100。sample_colorscale 接受 0～1 的位置。
    comparison_region_colors = REGION_COLORS.copy()
    if comparison_palette != "原始區域配色":
        comparison_region_colors = {
            region: sample_colorscale(CITY_COLOR_SCALES[comparison_palette], [(start + end) / 200])[0]
            for region, (start, end) in comparison_color_ranges.items()
        }
    # 依完整縣市清單固定色彩，不因移除縣市、病例排序或年份改變而重新配色。
    # 依完整 all_cities 固定縣市取色位置，再套用篩選。
    # 否則移除一個縣市可能讓其他縣市都換色，失去兩次畫面的可比性。
    comparison_city_colors = {}
    for region, base_color in comparison_region_colors.items():
        palette_cities = [city for city in all_cities if CITY_TO_REGION.get(city, "其他／未分類") == region]
        if palette_cities:
            if comparison_palette == "原始區域配色":
                palette_colors = sample_colorscale(
                    [base_color, "#ffffff"],
                    [0.12 + 0.43 * i / max(1, len(palette_cities) - 1) for i in range(len(palette_cities))],
                )
            else:
                # 等距取色：起點＋區間寬度×i/(n-1)；只有一項就取中點，避免除零。
                # 依分類位置而非病例數取色；兩端相同時整區同色。
                start, end = comparison_color_ranges[region]
                palette_colors = sample_colorscale(
                    CITY_COLOR_SCALES[comparison_palette],
                    [start / 100 + (end - start) / 100 * (i / (len(palette_cities) - 1) if len(palette_cities) > 1 else 0.5)
                     for i in range(len(palette_cities))],
                )
            comparison_city_colors.update(zip(
                palette_cities, palette_colors,
            ))
    # 只套用本分頁的比較縣市，不覆寫其他分頁使用的 filtered。
    comparison_filtered = comparison_base.loc[comparison_base[CITY_COL].isin(comparison_cities)]
    if not chosen_regions:
        st.info("請至少選擇一個比較區域；其他分頁不受影響。")
    elif not comparison_cities:
        st.info("請至少選擇一個納入比較的縣市；其他分頁不受影響。")
    elif comparison_filtered.empty:
        st.info("所選比較縣市在目前年份、年齡層、病例來源與性別條件下沒有資料。")
    else:
        # 先把資料加總成每縣市一列，再 reindex 補回所選但沒紀錄的縣市（填 0）。
        # rename_axis/reset_index 把縣市索引還原成一般欄位，方便 Plotly 使用。
        comparison_source = (
            comparison_filtered.groupby(CITY_COL, observed=True)[CASE_COL].sum()
            .reindex(comparison_cities, fill_value=0).rename_axis(CITY_COL).reset_index()
        )
        comparison_source[REGION_COL] = comparison_source[CITY_COL].map(CITY_TO_REGION).fillna("其他／未分類")
        # 區域合計再做一次 groupby；縣市明細直接保留各縣市，兩者共用畫圖流程。
        comparison_dimension = REGION_COL if comparison_unit == "區域合計" else CITY_COL
        comparison_label = "區域" if comparison_dimension == REGION_COL else "縣市"
        comparison_plot = (
            comparison_source.groupby(REGION_COL, observed=True, as_index=False)[CASE_COL].sum()
            if comparison_dimension == REGION_COL else comparison_source
        ).sort_values(CASE_COL, ascending=False)
        comparison_cases = int(comparison_plot[CASE_COL].sum())
        # transform('sum') 把區域總數對齊回每個縣市列，方便相除。
        # where 將 0 分母換成缺值，再 fillna(0)，避免提示比例出現無限大。
        region_denominators = comparison_source.groupby(REGION_COL, observed=True)[CASE_COL].transform("sum")
        comparison_source["區域內占比"] = comparison_source[CASE_COL].div(region_denominators.where(region_denominators != 0)).fillna(0)
        st.caption(
            f"本次比較：{len(comparison_cities)} 個縣市，合計 {comparison_cases:,.0f} 人。"
            "圓餅圖以本次比較合計為分母；部分縣市的區域合計不代表整個區域。"
        )
        # 兩欄共用篩選資料：左邊比較人數長度，右邊比較組成。
        region_bar_col, region_pie_col = st.columns(2)
        with region_bar_col:
            # 區域合計：y=區域、color=縣市，一個區域長條由多個縣市色塊組成。
            # 縣市明細：y=縣市、color=區域，維持一縣市一條。
            is_region_stack = comparison_dimension == REGION_COL
            region_bar = px.bar(
                comparison_source.sort_values([REGION_COL, CASE_COL, CITY_COL], ascending=[True, False, True])
                if is_region_stack else comparison_plot.sort_values(CASE_COL, ascending=True),
                x=CASE_COL,
                y=comparison_dimension,
                color=CITY_COL if is_region_stack else REGION_COL,
                color_discrete_map=comparison_city_colors if is_region_stack else comparison_region_colors,
                orientation="h",
                text=CITY_COL if is_region_stack else CASE_COL,
                custom_data=[CITY_COL, REGION_COL, "區域內占比"] if is_region_stack else None,
                title=f"各{comparison_label}確診人數比較",
                labels={CASE_COL: "確診人數", comparison_dimension: ""},
            )
            # customdata 帶入縣市、區域和區域內占比，供滑鼠提示讀取。
            # stack 堆疊的是人數，不是把每個區域都正規化成等長的百分比圖。
            if is_region_stack:
                region_bar.update_traces(
                    texttemplate="%{text}", textposition="inside", insidetextanchor="middle",
                    marker_line_color="white", marker_line_width=1,
                    hovertemplate=("%{customdata[0]}（%{y}）<br>確診人數：%{x:,.0f}"
                                   "<br>占所屬區域：%{customdata[2]:.1%}<extra></extra>"),
                )
                region_bar.update_layout(barmode="stack", uniformtext=dict(minsize=10, mode="hide"))
                # 用 annotation 標示長條右端的區域合計，不新增假資料，以免重複計算。
                for region, cases in comparison_plot[[REGION_COL, CASE_COL]].itertuples(index=False, name=None):
                    region_bar.add_annotation(
                        x=int(cases), y=region, text=f"{cases:,.0f}",
                        xanchor="left", xshift=6, showarrow=False,
                    )
                st.caption("每段色塊為一個縣市，長條右端標示區域合計；小色塊可移入滑鼠查看縣市與人數。")
            else:
                region_bar.update_traces(
                    texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}<br>確診人數：%{x:,.0f}<extra></extra>",
                )
            region_bar.update_layout(showlegend=False)
            region_bar.update_yaxes(
                type="category",
                categoryorder="array",
                categoryarray=comparison_plot.sort_values(CASE_COL, ascending=True)[comparison_dimension].tolist(),
            )
            region_bar.update_xaxes(tickformat=",.0f", range=[0, max(1, comparison_plot[CASE_COL].max() * 1.25)])
            apply_chart_style(region_bar, height=max(480, 30 * len(comparison_plot) + 160))
            st.plotly_chart(region_bar, width="stretch", key="region_bar", config={"displaylogo": False})
        # 全 0 仍可呈現長條，但無法合理計算圓餅占比，所以圓餅先檢查總數大於 0。
        with region_pie_col:
            if comparison_cases > 0:
                region_pie = px.pie(
                    comparison_plot,
                    names=comparison_dimension,
                    values=CASE_COL,
                    color=REGION_COL,
                    color_discrete_map=comparison_region_colors,
                    hole=0.45,
                    title=f"各{comparison_label}確診人數占比",
                )
                region_pie.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    hovertemplate="%{label}<br>確診人數：%{value:,.0f}<br>占比：%{percent}<extra></extra>",
                )
                # 雙層圖以兩個 Pie trace 共用中心：內圈區域、外圈縣市。
                # 內圈 domain 寬 0.79-0.21=0.58，對應外圈 hole=0.58，使兩圈相接。
                # 兩圈各自加總都等於本次比較總數，不可把兩個 trace 的 values 再相加。
                if comparison_dimension == REGION_COL:
                    # 兩圈使用相同起點、方向及區域順序，縣市弧段才會對齊所屬區域。
                    region_pie.update_traces(
                        domain=dict(x=[0.21, 0.79], y=[0.21, 0.79]),
                        sort=False,
                        direction="clockwise",
                        rotation=0,
                        textposition="inside",
                        name="區域",
                        insidetextorientation="auto",
                    )
                    # 外圈先按內圈的區域順序，再按該區域的縣市病例排序，讓同區域的弧段相鄰。
                    # sort=False 防止 Plotly 再次打散順序；兩圈方向、旋轉角相同才能對齊。
                    city_labels, city_values, city_colors, city_details = [], [], [], []
                    for region, region_cases in comparison_plot[[REGION_COL, CASE_COL]].itertuples(index=False, name=None):
                        region_cities = comparison_source.loc[
                            comparison_source[REGION_COL] == region
                        ].sort_values([CASE_COL, CITY_COL], ascending=[False, True])
                        for city, cases in region_cities[[CITY_COL, CASE_COL]].itertuples(index=False, name=None):
                            city_labels.append(city)
                            city_values.append(int(cases))
                            city_colors.append(comparison_city_colors[city])
                            city_details.append([region, cases / region_cases if region_cases else 0])
                    # 扇形大小仍用實際病例數；customdata 另存所屬區域與區域內占比。
                    # uniformtext 隱藏放不下的文字，但色塊資料未刪除，仍可移入滑鼠查看。
                    region_pie.add_trace(go.Pie(
                        labels=city_labels,
                        values=city_values,
                        customdata=city_details,
                        marker=dict(colors=city_colors, line=dict(color="white", width=1)),
                        hole=0.58,
                        sort=False,
                        direction="clockwise",
                        rotation=0,
                        textinfo="label",
                        textposition="inside",
                        insidetextorientation="radial",
                        name="縣市",
                        hovertemplate=(
                            "%{label}（%{customdata[0]}）<br>確診人數：%{value:,.0f}"
                            "<br>占所屬區域：%{customdata[1]:.1%}"
                            "<br>占本次比較：%{percent}<extra></extra>"
                        ),
                    ))
                    region_pie.update_layout(uniformtext=dict(minsize=10, mode="hide"))
                    st.caption("內圈為區域，外圈為所屬縣市；圖上百分比皆占本次比較合計，小區塊可移入滑鼠查看。")
                region_pie.update_layout(showlegend=False)
                region_pie.add_annotation(
                    text=f"合計<br><b>{comparison_cases:,.0f}</b>",
                    x=0.5, y=0.5, showarrow=False,
                )
                apply_chart_style(region_pie, height=480)
                st.plotly_chart(region_pie, width="stretch", key="region_pie", config={"displaylogo": False})
            else:
                st.info("目前比較病例合計為 0，無法計算占比；長條圖仍顯示所選項目的 0 值。")
    st.caption("區域採北、中、南、東、離島互斥分組，每個縣市只計算一次。")

# 【章節 12】一般分頁：縣市比較
# 區域比較完成後才停止其他空資料分頁，避免左側清空阻擋跨區域比較。
if filtered.empty:
    for empty_tab in [tab_city, tab_age, tab_heat, tab_race, tab_trend, tab_data]:
        with empty_tab:
            st.info("左側縣市篩選沒有資料，請調整左側縣市；區域比較可另外選取縣市。")
    st.stop()

# 單一縣市直接顯示，因為 st.slider 不允許 min_value=max_value=1。
# 只有多個縣市才提供顯示數量滑桿。
with tab_city:
    if len(city_total) == 1:
        view_limit = 1
        st.caption("目前僅有 1 個縣市，直接顯示該縣市。")
    else:
        view_limit = st.slider(
            "顯示縣市數",
            min_value=1,
            max_value=len(city_total),
            value=len(city_total),
            help="依篩選後病例數由高至低選取。",
        )
    with st.expander("色彩對比設定", expanded=True):
        color_palette = st.selectbox(
            "色帶配色", list(CITY_COLOR_SCALES),
            index=list(CITY_COLOR_SCALES).index("YlOrRd"), key="city_color_palette",
            help="僅改變縣市比較配色；Blue、Green 分別使用 Blues、Greens 漸層色帶。",
        )
        color_strength = st.slider(
            "小值色彩加強", min_value=1.0, max_value=4.0, value=2.5, step=0.5,
            key="city_color_strength",
            help="1 為原始等比例；數值越大，越強調小病例數的色差，但大值之間的色差會縮小。只影響顏色，不改變長條長度。",
        )
        color_start, color_end = st.slider(
            "色帶使用範圍（%）", min_value=0, max_value=100, value=(20, 100), step=5,
            key="city_color_range",
            help="0 是所選色帶的最淺端，100 是最深端。提高左端可避開接近白色的區段；兩端拉開可增加整體深淺差異。",
        )
        if color_start == color_end:
            st.info("色帶兩端相同，所有長條會使用同色；請拉開兩端以區分數值。")
    # 先 head 取病例最多的前幾名，再按病例升冪排，讓水平圖的大值位於上方。
    city_view = city_total.head(view_limit).sort_values(CASE_COL, ascending=True).copy()
    # 使用所有篩選後縣市的最大值，避免改變顯示數量時同一縣市變色。
    color_max = max(1, int(city_total[CASE_COL].max()))
    # 設 r=病例數/最大值，顏色用 r^(1/加強倍數)。r 在 0～1 間，次方小於 1
    # 會拉高小值的色彩位置；只有 color 改變，x、text、hover 仍是真實病例數。
    city_view["色彩強度"] = (city_view[CASE_COL] / color_max) ** (1.0 / color_strength)
    # 在自訂色帶範圍取 17 個位置，讓 Plotly 在中間插值成連續漸層。
    city_colors = sample_colorscale(
        CITY_COLOR_SCALES[color_palette],
        [color_start / 100 + (color_end - color_start) / 100 * index / 16 for index in range(17)],
    )
    st.caption(
        f"{color_palette} 色彩加強 {color_strength:g} 倍，使用色帶 {color_start}–{color_end}%；"
        "顏色僅輔助區分大小，長條長度與標示數字仍為實際病例數。"
    )
    city_fig = px.bar(
        city_view,
        x=CASE_COL,
        y=CITY_COL,
        orientation="h",
        text=CASE_COL,
        title="各縣市確診人數排名",
        color="色彩強度",
        color_continuous_scale=city_colors,
        range_color=(0, 1),
        labels={CASE_COL: "確診人數", CITY_COL: ""},
    )
    city_fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>確診人數：%{x:,.0f}<extra></extra>",
    )
    city_fig.update_layout(coloraxis_showscale=False)
    city_fig.update_xaxes(tickformat=",.0f")
    apply_chart_style(city_fig, height=max(470, 31 * len(city_view) + 130))
    st.plotly_chart(city_fig, width="stretch", config={"displaylogo": False})


# 【章節 13】年齡結構
# age_city 是每縣市×每年齡層一列的加總表，最後下載分頁也會重用。
with tab_age:
    age_chart_type = st.radio(
        "呈現方式",
        ["跨縣市比較（推薦）", "整體年齡占比"],
        horizontal=True,
    )

    age_city = (
        filtered.groupby([CITY_COL, AGE_COL], observed=True, as_index=False)[CASE_COL]
        .sum()
    )
    selected_age_order = [age for age in all_ages if age in selected_ages]
    city_rank_order = city_total[CITY_COL].tolist()

    if age_chart_type == "跨縣市比較（推薦）":
        scale_mode = st.radio("數值尺度", ["確診人數", "縣市內占比"], horizontal=True)
        age_fig = px.bar(
            age_city,
            x=CASE_COL,
            y=CITY_COL,
            color=AGE_COL,
            orientation="h",
            title="縣市與年齡層交叉比較",
            category_orders={CITY_COL: list(reversed(city_rank_order)), AGE_COL: selected_age_order},
            labels={CASE_COL: "確診人數", CITY_COL: "", AGE_COL: "年齡層"},
            color_discrete_sequence=px.colors.qualitative.Light24,
        )
        # barnorm='percent' 把每縣市堆疊正規化為 100%，比較組成而非人數。
        # 分母為該縣市在目前篩選內的病例，並非人口，因此不是感染率。
        if scale_mode == "縣市內占比":
            age_fig.update_layout(barnorm="percent")
            age_fig.update_xaxes(title="縣市內占比", ticksuffix="%")
            age_fig.update_traces(
                hovertemplate="%{y}<br>年齡層：%{fullData.name}<br>病例數：%{x:,.0f}<extra></extra>"
            )
        else:
            age_fig.update_xaxes(tickformat=",.0f")
            age_fig.update_traces(
                hovertemplate="%{y}<br>年齡層：%{fullData.name}<br>確診人數：%{x:,.0f}<extra></extra>"
            )
        apply_chart_style(age_fig, height=max(520, 31 * len(city_rank_order) + 150))
        st.plotly_chart(age_fig, width="stretch", config={"displaylogo": False})
    else:
        # 整體年齡占比合併所有所選縣市，分母是全部篩選病例，不是各縣市各自的分母。
        age_plot = age_total.copy()
        age_plot["占比"] = age_plot[CASE_COL] / age_plot[CASE_COL].sum()
        age_fig = px.pie(
            age_plot,
            names=AGE_COL,
            values=CASE_COL,
            hole=0.48,
            title="篩選範圍內的年齡層占比",
            category_orders={AGE_COL: selected_age_order},
            color_discrete_sequence=px.colors.qualitative.Light24,
        )
        age_fig.update_traces(
            textposition="inside" if len(age_plot) <= 10 else "none",
            textinfo="percent+label",
            hovertemplate="年齡層：%{label}<br>確診人數：%{value:,.0f}<br>占比：%{percent}<extra></extra>",
        )
        age_fig.add_annotation(
            text=f"合計<br><b>{total_cases:,.0f}</b>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=17, color="#123955"),
        )
        apply_chart_style(age_fig, height=590)
        st.plotly_chart(age_fig, width="stretch", config={"displaylogo": False})

# 【交叉熱力圖分頁】重用年齡結構已彙整的資料，避免再次讀取原始 CSV。
with tab_heat:
    render_heatmap(age_city, [city for city in all_cities if city in selected_cities], selected_age_order)

# 【章節 14】動畫控制區
# 先呼叫 build_race_data 算數值，再呼叫 create_bar_race 畫圖，分開資料與呈現邏輯。
with tab_race:
    st.caption("按下「▶ 播放」觀看病例變化，也可拖曳時間軸直接跳到指定時間。縣市依病例數排名，年齡層依年齡固定排列。")

    control1, control2, control3 = st.columns(3)
    with control1:
        ranking_target = st.radio("排名對象", ["縣市", "年齡層"], horizontal=True)
    with control2:
        time_grain = st.radio("動畫時間", ["每週", "年度"], horizontal=True)
    with control3:
        value_mode = st.radio("數值方式", ["當期病例數", "累積病例數"], horizontal=True)

    # A if 條件 else B 是條件運算式，讓同一套函式可切換處理縣市／年齡。
    dimension = CITY_COL if ranking_target == "縣市" else AGE_COL
    categories = selected_cities if dimension == CITY_COL else selected_ages
    if dimension == AGE_COL:
        top_n = len(categories)
        st.caption("年齡層由上到下固定按年齡由小到大排列，顯示左側勾選的全部年齡層，不取病例前 N 名。")
    elif len(categories) == 1:
        top_n = 1
        st.caption(f"目前僅有 1 個{ranking_target}，動畫直接顯示該項目。")
    else:
        top_n = st.slider(
            "動畫顯示前 N 名",
            min_value=1,
            max_value=len(categories),
            value=min(12, len(categories)),
            help="每個時間點會重新排序並顯示前 N 名。",
        )

    race_data = build_race_data(
        filtered=filtered,
        dimension=dimension,
        categories=categories,
        time_grain=time_grain,
        value_mode=value_mode,
        selected_years=selected_years,
    )
    # enumerate 同時提供位置與分類；% len(COLORS) 讓索引在色表範圍內循環。
    # 整段動畫共用此表，所以排名交換不會讓同一分類換色。
    color_map = {
        str(category): COLORS[index % len(COLORS)]
        for index, category in enumerate(categories)
    }

    period_title = "每週消長" if time_grain == "每週" else ("年度變化" if dimension == AGE_COL else "年度排名")
    st.subheader(f"{ranking_target}{period_title}")
    race_fig = create_bar_race(
        race=race_data,
        dimension=dimension,
        top_n=top_n,
        color_map=color_map,
        time_grain=time_grain,
        value_mode=value_mode,
    )
    st.plotly_chart(race_fig, width="stretch", key=f"race_{dimension}", config={"displaylogo": False})

# 【章節 15】時間趨勢
# 合計模式每週一個總數；依縣市模式每週每縣市一個總數，皆先排序再畫折線。
# 年度週別是字串分類軸，未補齊完全未出現的週別，間距不能解讀為實際天數距離。
with tab_trend:
    trend_mode = st.radio("趨勢比較", ["篩選條件合計", "依縣市"], horizontal=True)
    if trend_mode == "篩選條件合計":
        trend = (
            filtered.groupby([YEAR_COL, WEEK_COL], observed=True, as_index=False)[CASE_COL]
            .sum()
            .sort_values([YEAR_COL, WEEK_COL])
        )
        trend["年度週別"] = trend[YEAR_COL].astype(str) + "-W" + trend[WEEK_COL].astype(str).str.zfill(2)
        trend_fig = px.line(
            trend,
            x="年度週別",
            y=CASE_COL,
            markers=False,
            title="每週確診人數趨勢",
            labels={CASE_COL: "確診人數", "年度週別": "發病年度／週別"},
        )
        trend_fig.update_traces(
            line=dict(color="#167d91", width=2.5),
            hovertemplate="%{x}<br>確診人數：%{y:,.0f}<extra></extra>",
        )
    else:
        trend = (
            filtered.groupby([YEAR_COL, WEEK_COL, CITY_COL], observed=True, as_index=False)[CASE_COL]
            .sum()
            .sort_values([YEAR_COL, WEEK_COL])
        )
        trend["年度週別"] = trend[YEAR_COL].astype(str) + "-W" + trend[WEEK_COL].astype(str).str.zfill(2)
        trend_fig = px.line(
            trend,
            x="年度週別",
            y=CASE_COL,
            color=CITY_COL,
            title="各縣市每週確診人數趨勢",
            labels={CASE_COL: "確診人數", "年度週別": "發病年度／週別", CITY_COL: "縣市"},
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        trend_fig.update_traces(
            hovertemplate="%{fullData.name}<br>%{x}<br>確診人數：%{y:,.0f}<extra></extra>"
        )
        if filtered[CITY_COL].nunique() > 8:
            st.info("目前縣市較多，折線會較密集；可從左側縮小縣市範圍以便比較。")

    trend_fig.update_xaxes(nticks=16, tickangle=-35)
    trend_fig.update_yaxes(tickformat=",.0f")
    apply_chart_style(trend_fig, height=570)
    st.plotly_chart(trend_fig, width="stretch", config={"displaylogo": False})

# 【章節 16】表格與 CSV 下載
# copy() 讓排序和新增比例欄位不會影響原本 age_city 圖表資料。
with tab_data:
    table = age_city.copy()
    # transform 使每列都取得同縣市總數。此處保留原版未加除零處理的寫法：
    # 若某縣市合計為 0，比例會成為缺值 NaN；不要把這個病例比例當成感染率。
    table["占縣市病例比例"] = table[CASE_COL] / table.groupby(CITY_COL)[CASE_COL].transform("sum")
    # Categorical 明列排序規則：縣市照病例排名、年齡照年齡順序，不照文字排序。
    table[CITY_COL] = pd.Categorical(table[CITY_COL], categories=city_total[CITY_COL], ordered=True)
    table[AGE_COL] = pd.Categorical(table[AGE_COL], categories=selected_age_order, ordered=True)
    table = table.sort_values([CITY_COL, AGE_COL]).reset_index(drop=True)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            CITY_COL: st.column_config.TextColumn("縣市"),
            AGE_COL: st.column_config.TextColumn("年齡層"),
            CASE_COL: st.column_config.NumberColumn("確診人數", format="localized"),
            "占縣市病例比例": st.column_config.ProgressColumn(
                "占縣市病例比例", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )
    # 匯出前把分類型別轉回文字；** 將欄名→欄值字典展開成 assign 的參數。
    # 最終 encode('utf-8-sig') 產生含 BOM 的位元組，方便 Excel 辨識中文。
    # index=False 不另外匯出 pandas 列索引。
    csv_bytes = table.assign(
        **{
            CITY_COL: table[CITY_COL].astype("string"),
            AGE_COL: table[AGE_COL].astype("string"),
        }
    ).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下載目前篩選後的縣市 × 年齡層資料",
        data=csv_bytes,
        file_name="covid19_縣市_年齡層_篩選結果.csv",
        mime="text/csv",
    )

# 【章節 17】資料解讀說明
# 圖表必須交代統計範圍、分母、時間與地理定義，才不會把病例數誤讀為風險。
with st.expander("資料定義與注意事項"):
    st.markdown(
        """
        - **確診人數**：CSV 中「確定病例數」欄位加總；原始每列是依週別、鄉鎮、性別、來源與年齡層等維度彙整的紀錄。
        - **發病時間**：時間趨勢使用資料中的發病年份與發病週別，不代表通報日或公布日。
        - **動態排名**：可依每週或年度播放，並切換當期與累積病例數；縣市依病例數取前 N 名，年齡層顯示全部所選年齡、由上到下固定由小到大排序（0、1、2、3、4、5~9、10~14…）。X 軸範圍在所有影格間固定。
        - **區域比較**：以北、中、南、東、離島互斥分組，每個縣市只加總一次；未知縣市歸為「其他／未分類」。可選區域合計或縣市明細；占比分母為本分頁實際納入比較的病例合計，並非全國人口或感染率。
        - **區域定義**：北中南東參考[國發會分區](https://www.ndc.gov.tw/Content_List.aspx?n=5A2D326B69DD0112)，本圖表另將澎湖與金門、連江統一歸入離島；完整縣市清單見左側「區域分組說明」。
        - **性別未註明**：原始性別缺值保留為「未註明」，可從左側篩選。
        - **圖表選擇**：長條圖適合精確比較類別；圓餅圖只用於單一篩選母體的組成。直方圖不適合此彙整資料，因此未採用。
        """
    )

# =============================================================================
# 【課堂練習】只供閱讀、討論，不會執行
# 1. 選 2023 年與單一縣市，追蹤 selected_years → common_mask → filtered → city_total。
# 2. 在區域分頁另選跨區域縣市：為何上方 KPI 不變，但區域兩圖會改變？
# 3. 比較 groupby().sum() 與 groupby().transform('sum') 的結果列數及用途。
# 4. 色彩加強設為 1 與 4：哪些值改變？長條長度有沒有改變？
# 5. 找到動畫補零、排序與 cumsum，說明當期和累積的差別。
# 6. 找出避免單一選項滑桿出錯與除以零的寫法，也找出仍保留原版限制的地方。
# 7. 檢查：區域各縣市病例和＝堆疊總長度＝內圈該區域扇形病例數。
# 8. 篩選掉一些年份後的累積病例，能否代表疫情全部歷史的累積？為什麼？
# 建議除錯順序：控制項回傳值 → 篩選表 → 加總表 → 圖表設定。
# 修改練習請另外存檔，保留本教學版與 app_c19.py 作為比較基準。
# =============================================================================

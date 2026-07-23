import argparse
import os
import sys
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

import akshare as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import money_thermo
import tech_thermo

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def configure_console_encoding():
    """
    配置控制台输出编码，避免 Windows 下中文输出报错。
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')


def get_project_root():
    """
    获取项目根目录。
    """
    return SCRIPT_DIR.parent


def build_output_path(filename):
    """
    构造输出文件路径。
    """
    output_dir = get_project_root() / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def load_env_value(key, env_path=None):
    """
    从 .env 文件中读取指定配置项。
    """
    if env_path is None:
        env_path = get_project_root() / '.env'

    if not env_path.exists():
        raise FileNotFoundError(f'未找到 .env 文件: {env_path}')

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue

        current_key, value = line.split('=', 1)
        if current_key.strip() == key:
            return value.strip().strip('"').strip("'")

    raise KeyError(f'.env 中缺少配置项: {key}')


def merge_config(base, override):
    """
    递归合并默认配置与外部配置。
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG = {
    'data': {
        'value_history_start': '20100101',
        'emotion_min_history_start': '20150101',
        'money_min_history_start': '20150101',
        'tech_min_history_start': '20100101',
        'history_buffer_years': 3,
    },
    'weights': {
        'value': 0.40,
        'emotion': 0.25,
        'money': 0.20,
        'tech': 0.15,
    },
    'anchor': {
        'value': 0.70,
        'money': 0.15,
        'emotion': 0.10,
        'tech': 0.05,
    },
    'value': {
        'spread_weight': 0.70,
        'pe_weight': 0.30,
    },
    'plot_thresholds': {
        'timeseries_main': {
            'cold': 20,
            'hot': 70,
        },
        'timeseries_dimensions': {
            'cold': 20,
            'hot': 80,
        },
        'dashboard': {
            'cold': 20,
            'hot': 80,
        },
    },
}


def load_parameter_config(config_path=None):
    """
    加载综合温度计参数配置。
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'market_thermo_params.toml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = get_project_root() / config_path

    config = DEFAULT_CONFIG
    if config_path.exists():
        with config_path.open('rb') as file:
            loaded = tomllib.load(file)
        config = merge_config(DEFAULT_CONFIG, loaded)
        print(f'已加载综合参数文件: {config_path}')
    else:
        print(f'未找到综合参数文件，使用默认参数: {config_path}')

    return config, config_path


def normalize_date_str(date_str):
    """
    标准化日期字符串为 YYYYMMDD。
    """
    return pd.to_datetime(date_str).strftime('%Y%m%d')


def shift_start_date(start_date, years, min_start):
    """
    将起始日期向前回溯若干年，并限定最早边界。
    """
    start_ts = pd.to_datetime(start_date)
    min_ts = pd.to_datetime(min_start)
    shifted = start_ts - pd.DateOffset(years=years)
    return max(shifted, min_ts).strftime('%Y%m%d')


def daterange_chunks(start_date, end_date, chunk_days=365):
    """
    将长区间拆分为多个子区间，规避单次接口返回行数上限。
    """
    current_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)

    while current_start <= final_end:
        current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), final_end)
        yield current_start.strftime('%Y%m%d'), current_end.strftime('%Y%m%d')
        current_start = current_end + pd.Timedelta(days=1)


def classify_temperature(score):
    """
    将 0-100 分映射为市场温度状态。
    """
    if score < 20:
        return '极度冰点'
    if score < 40:
        return '低温磨底'
    if score < 60:
        return '常温震荡'
    if score < 80:
        return '高温过热'
    return '极度沸腾'


def get_hs300_data(start_date='20100101', end_date=None):
    """
    获取沪深300指数的 PE_TTM 和收盘价。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    pe_frames = []
    price_frames = []
    for chunk_start, chunk_end in daterange_chunks(start_date, end_date, chunk_days=365):
        df_pe_chunk = pro.index_dailybasic(
            ts_code='000300.SH',
            start_date=chunk_start,
            end_date=chunk_end,
            fields='trade_date,pe_ttm'
        )
        df_price_chunk = pro.index_daily(
            ts_code='000300.SH',
            start_date=chunk_start,
            end_date=chunk_end,
            fields='trade_date,close'
        )
        if df_pe_chunk is not None and len(df_pe_chunk) > 0:
            pe_frames.append(df_pe_chunk)
        if df_price_chunk is not None and len(df_price_chunk) > 0:
            price_frames.append(df_price_chunk)

    if not pe_frames or not price_frames:
        return None

    df_pe = pd.concat(pe_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    df_price = pd.concat(price_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    df = pd.merge(df_pe, df_price, on='trade_date', how='inner')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['pe_ttm'] = pd.to_numeric(df['pe_ttm'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['pe_ttm', 'close']).sort_values('trade_date').reset_index(drop=True)
    return df


def get_index_close(ts_code, start_date='20100101', end_date=None):
    """
    分段获取指定指数的日度收盘价。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    frames = []
    for chunk_start, chunk_end in daterange_chunks(start_date, end_date, chunk_days=365):
        df_chunk = pro.index_daily(
            ts_code=ts_code,
            start_date=chunk_start,
            end_date=chunk_end,
            fields='trade_date,close'
        )
        if df_chunk is not None and len(df_chunk) > 0:
            frames.append(df_chunk)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('trade_date').reset_index(drop=True)
    return df


def get_bond_yield(start_date='20100101', end_date=None):
    """
    获取中国 10 年期国债收益率。
    为避免综合脚本被外部接口长时间阻塞，这里直接使用项目内置锚点插值口径。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    china_yield_data = {
        'trade_date': pd.to_datetime([
            '2010-01-04', '2010-07-01', '2011-01-04', '2011-07-01',
            '2012-01-04', '2012-07-02', '2013-01-04', '2013-07-01',
            '2014-01-02', '2014-07-01', '2015-01-05', '2015-07-01',
            '2016-01-04', '2016-07-01', '2017-01-03', '2017-07-03',
            '2018-01-02', '2018-07-02', '2019-01-02', '2019-07-01',
            '2020-01-02', '2020-07-01', '2021-01-04', '2021-07-01',
            '2022-01-04', '2022-07-01', '2023-01-03', '2023-07-03',
            '2024-01-02', '2024-07-01', '2025-01-02', '2025-07-01',
            '2026-01-05', '2026-07-01'
        ]),
        'yield_10y': [
            3.64, 3.28, 3.88, 3.87,
            3.55, 3.34, 4.55, 4.03,
            4.02, 4.10, 3.37, 3.62,
            2.86, 2.92, 3.58, 3.50,
            3.62, 3.48, 3.18, 3.20,
            2.97, 3.10, 3.03, 2.85,
            2.84, 2.82, 2.43, 2.65,
            1.68, 1.75, 1.66, 1.60,
            1.66, 1.60
        ]
    }
    df_china = pd.DataFrame(china_yield_data).sort_values('trade_date').reset_index(drop=True)
    df_china = df_china.set_index('trade_date').resample('D').interpolate(method='linear').reset_index()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    return df_china[(df_china['trade_date'] >= start_dt) & (df_china['trade_date'] <= end_dt)].copy()


def compute_value_score(df_value, current_date, config):
    """
    计算价值维度得分。
    """
    history = df_value[df_value['trade_date'] <= current_date].copy()
    latest = history.iloc[-1]

    current_spread = latest['spread']
    current_pe = latest['pe_ttm']
    current_yield = latest['yield_10y']
    current_earnings_yield = latest['earnings_yield']
    spread_percentile = (history['spread'] <= current_spread).mean() * 100
    pe_percentile = (history['pe_ttm'] <= current_pe).mean() * 100

    spread_weight = config['value']['spread_weight']
    pe_weight = config['value']['pe_weight']
    # 温度语义统一为：高温=偏贵/偏热，低温=偏便宜/偏冷。
    # 因此股债收益差越高（越便宜）温度应越低，PE 分位越高（越贵）温度应越高。
    score_spread = 100 - spread_percentile
    score_pe = pe_percentile
    total_score = score_spread * spread_weight + score_pe * pe_weight

    return {
        'dimension': '价值',
        'total_score': total_score,
        'score_spread': score_spread,
        'score_pe': score_pe,
        'spread': current_spread,
        'spread_percentile': spread_percentile,
        'pe_ttm': current_pe,
        'pe_percentile': pe_percentile,
        'yield_10y': current_yield,
        'earnings_yield': current_earnings_yield,
        'current_date': latest['trade_date'],
        'state': classify_temperature(total_score),
    }


def prepare_value_dimension_data(end_date, config):
    """
    准备价值维度原始数据。
    """
    value_start = config['data']['value_history_start']
    df_hs300 = get_hs300_data(value_start, end_date)
    df_bond = get_bond_yield(value_start, end_date)
    if df_hs300 is None or df_bond is None or len(df_hs300) == 0 or len(df_bond) == 0:
        raise ValueError('价值维度数据获取失败')

    df_value = pd.merge(df_hs300, df_bond, on='trade_date', how='inner').sort_values('trade_date').reset_index(drop=True)
    df_value['earnings_yield'] = (1 / df_value['pe_ttm']) * 100
    df_value['spread'] = df_value['earnings_yield'] - df_value['yield_10y']
    return df_value


def build_value_dimension_series(df_value, start_date, end_date, config):
    """
    构建价值维度的日度得分序列。
    """
    df = df_value.copy().sort_values('trade_date').reset_index(drop=True)
    df['spread_percentile'] = expanding_percentile(df['spread'])
    df['pe_percentile'] = expanding_percentile(df['pe_ttm'])
    df['score_spread'] = 100 - df['spread_percentile']
    df['score_pe'] = df['pe_percentile']
    df['value_score'] = (
        df['score_spread'] * config['value']['spread_weight'] +
        df['score_pe'] * config['value']['pe_weight']
    )
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return df[(df['trade_date'] >= start_ts) & (df['trade_date'] <= end_ts)].copy()


def get_market_turnover(start_date='20100101', end_date=None):
    """
    获取两市日度成交额。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    df_sh = pro.index_daily(
        ts_code='000001.SH',
        start_date=start_date,
        end_date=end_date,
        fields='trade_date,amount'
    )
    df_sz = pro.index_daily(
        ts_code='399001.SZ',
        start_date=start_date,
        end_date=end_date,
        fields='trade_date,amount'
    )

    if df_sh is None or len(df_sh) == 0 or df_sz is None or len(df_sz) == 0:
        return None

    df_sh['trade_date'] = pd.to_datetime(df_sh['trade_date'])
    df_sz['trade_date'] = pd.to_datetime(df_sz['trade_date'])
    df = pd.merge(
        df_sh.rename(columns={'amount': 'amount_sh'}),
        df_sz.rename(columns={'amount': 'amount_sz'}),
        on='trade_date',
        how='inner'
    )
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['turnover_thousand_yuan'] = pd.to_numeric(df['amount_sh'], errors='coerce') + pd.to_numeric(df['amount_sz'], errors='coerce')
    df['turnover_billion'] = df['turnover_thousand_yuan'] / 100000
    return df[['trade_date', 'turnover_thousand_yuan', 'turnover_billion']]


def get_market_turnover_rate_monthly():
    """
    获取月度全市场换手率。
    """
    df = ak.macro_china_stock_market_cap()
    if df is None or len(df) == 0:
        return None

    date_col = df.columns[0]
    sh_mkt_col = df.columns[3]
    sz_mkt_col = df.columns[4]
    sh_amt_col = df.columns[5]
    sz_amt_col = df.columns[6]

    df = df.rename(
        columns={
            date_col: 'trade_date',
            sh_mkt_col: 'market_cap_sh',
            sz_mkt_col: 'market_cap_sz',
            sh_amt_col: 'turnover_sh',
            sz_amt_col: 'turnover_sz',
        }
    )
    df['trade_date'] = pd.to_datetime(
        df['trade_date'].astype(str).str.replace('年', '-', regex=False).str.replace('月份', '-01', regex=False),
        errors='coerce'
    )
    for col in ['market_cap_sh', 'market_cap_sz', 'turnover_sh', 'turnover_sz']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['trade_date', 'market_cap_sh', 'market_cap_sz', 'turnover_sh', 'turnover_sz']).sort_values('trade_date').reset_index(drop=True)
    df['market_cap_total'] = df['market_cap_sh'] + df['market_cap_sz']
    df['turnover_total'] = df['turnover_sh'] + df['turnover_sz']
    df['market_turnover_rate'] = df['turnover_total'] / df['market_cap_total'] * 100
    return df[['trade_date', 'market_turnover_rate', 'turnover_total', 'market_cap_total']]


def get_margin_buy_ratio(df_turnover, start_date='20100101', end_date=None):
    """
    获取融资买入额占成交额比例。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    df_sh = ak.macro_china_market_margin_sh()
    df_sz = ak.macro_china_market_margin_sz()
    df_sh = df_sh.rename(columns={df_sh.columns[0]: 'trade_date', df_sh.columns[1]: 'margin_buy_sh'})
    df_sz = df_sz.rename(columns={df_sz.columns[0]: 'trade_date', df_sz.columns[1]: 'margin_buy_sz'})
    df_sh['trade_date'] = pd.to_datetime(df_sh['trade_date'])
    df_sz['trade_date'] = pd.to_datetime(df_sz['trade_date'])
    df_sh['margin_buy_sh'] = pd.to_numeric(df_sh['margin_buy_sh'], errors='coerce')
    df_sz['margin_buy_sz'] = pd.to_numeric(df_sz['margin_buy_sz'], errors='coerce')

    df_margin = pd.merge(df_sh[['trade_date', 'margin_buy_sh']], df_sz[['trade_date', 'margin_buy_sz']], on='trade_date', how='outer')
    df_margin = df_margin[(df_margin['trade_date'] >= pd.to_datetime(start_date)) & (df_margin['trade_date'] <= pd.to_datetime(end_date))]
    df_margin = df_margin.sort_values('trade_date').reset_index(drop=True)
    df_margin[['margin_buy_sh', 'margin_buy_sz']] = df_margin[['margin_buy_sh', 'margin_buy_sz']].fillna(0)
    df_margin['margin_buy_yuan'] = df_margin['margin_buy_sh'] + df_margin['margin_buy_sz']

    df = pd.merge(df_turnover[['trade_date', 'turnover_thousand_yuan']], df_margin[['trade_date', 'margin_buy_yuan']], on='trade_date', how='inner')
    df['turnover_yuan'] = df['turnover_thousand_yuan'] * 1000
    df['margin_buy_ratio'] = df['margin_buy_yuan'] / df['turnover_yuan'] * 100
    df['margin_buy_ratio_ma20'] = df['margin_buy_ratio'].rolling(window=20, min_periods=5).mean()
    return df[['trade_date', 'margin_buy_yuan', 'margin_buy_ratio', 'margin_buy_ratio_ma20']]


def rolling_percentile(series, window, min_periods):
    """
    计算滚动窗口内当前位置的百分位。
    """
    def _last_percentile(values):
        s = pd.Series(values)
        return s.rank(pct=True).iloc[-1] * 100

    return series.rolling(window=window, min_periods=min_periods).apply(_last_percentile, raw=False)


def expanding_percentile(series):
    """
    计算逐步扩展样本下当前位置的历史分位。
    """
    def _last_percentile(values):
        s = pd.Series(values)
        return s.rank(pct=True).iloc[-1] * 100

    return series.expanding(min_periods=1).apply(_last_percentile, raw=False)


def calculate_emotion_indicators(df_turnover, df_turnover_rate, df_margin):
    """
    将原始情绪数据转换为可打分指标。
    """
    df_t = df_turnover.copy()
    df_t['high_1y'] = df_t['turnover_billion'].rolling(window=252, min_periods=100).max()
    df_t['turnover_decline_rate'] = (df_t['turnover_billion'] / df_t['high_1y'] - 1) * 100
    df_t['turnover_decline_rate'] = df_t['turnover_decline_rate'].clip(-100, 0)
    df_t['high_3y'] = df_t['turnover_billion'].rolling(window=252 * 3, min_periods=252).max()
    df_t['turnover_decline_rate_3y'] = (df_t['turnover_billion'] / df_t['high_3y'] - 1) * 100
    df_t['turnover_decline_rate_3y'] = df_t['turnover_decline_rate_3y'].clip(-100, 0)

    df_r = None
    if df_turnover_rate is not None:
        df_r = df_turnover_rate.copy()
        df_r['market_turnover_rate_ma3'] = df_r['market_turnover_rate'].rolling(window=3, min_periods=2).mean()
        df_r['market_turnover_rate_percentile'] = rolling_percentile(df_r['market_turnover_rate'], window=36, min_periods=12)

    df_m = df_margin.copy()
    df_m['margin_buy_ratio_percentile'] = rolling_percentile(df_m['margin_buy_ratio'], window=252 * 3, min_periods=252)
    df_m['margin_buy_ratio_percentile'] = df_m['margin_buy_ratio_percentile'].fillna(expanding_percentile(df_m['margin_buy_ratio']))
    return df_t, df_r, df_m


def compute_emotion_score(df_t, df_r, df_m, current_date=None):
    """
    计算情绪维度得分。
    """
    if current_date is None:
        current_date = df_t['trade_date'].iloc[-1]

    latest_t = df_t[df_t['trade_date'] == current_date]
    if len(latest_t) == 0:
        latest_t = df_t[df_t['trade_date'] <= current_date].iloc[-1:]

    latest_m = df_m[df_m['trade_date'] == current_date]
    if len(latest_m) == 0:
        latest_m = df_m[df_m['trade_date'] <= current_date].iloc[-1:]

    latest_r = None
    if df_r is not None:
        latest_r = df_r[df_r['trade_date'] <= current_date]
        if len(latest_r) > 0:
            latest_r = latest_r.iloc[-1:]

    decline_rate = latest_t['turnover_decline_rate'].iloc[-1]
    decline_rate_3y = latest_t['turnover_decline_rate_3y'].iloc[-1]
    score_turnover = max(0, min(100, decline_rate + 100))
    margin_buy_ratio = latest_m['margin_buy_ratio'].iloc[-1]
    margin_percentile = latest_m['margin_buy_ratio_percentile'].iloc[-1]
    score_margin = max(0, min(100, margin_percentile))
    market_turnover_rate = latest_r['market_turnover_rate'].iloc[-1] if latest_r is not None and len(latest_r) > 0 else np.nan
    total_score = score_turnover * 0.6 + score_margin * 0.4

    return {
        'dimension': '情绪',
        'total_score': total_score,
        'score_turnover': score_turnover,
        'score_margin': score_margin,
        'decline_rate': decline_rate,
        'decline_rate_3y': decline_rate_3y,
        'market_turnover_rate': market_turnover_rate,
        'margin_buy_ratio': margin_buy_ratio,
        'margin_percentile': margin_percentile,
        'current_date': current_date,
        'state': classify_temperature(total_score),
    }


def prepare_emotion_dimension_data(start_date, end_date, config):
    """
    准备情绪维度原始数据。
    """
    history_start = shift_start_date(
        start_date,
        config['data']['history_buffer_years'],
        config['data']['emotion_min_history_start']
    )
    df_turnover = get_market_turnover(history_start, end_date)
    df_turnover_rate = get_market_turnover_rate_monthly()
    if df_turnover is None or df_turnover_rate is None:
        raise ValueError('情绪维度基础数据获取失败')

    df_turnover_rate = df_turnover_rate[df_turnover_rate['trade_date'] <= pd.to_datetime(end_date)].copy()
    df_margin = get_margin_buy_ratio(df_turnover, history_start, end_date)
    if df_margin is None or len(df_margin) == 0:
        raise ValueError('情绪维度融资买入数据获取失败')

    return calculate_emotion_indicators(df_turnover, df_turnover_rate, df_margin)


def build_emotion_dimension_series(df_t, df_m, start_date, end_date):
    """
    构建情绪维度的日度得分序列。
    """
    df = pd.merge(
        df_t[['trade_date', 'turnover_decline_rate', 'turnover_decline_rate_3y']],
        df_m[['trade_date', 'margin_buy_ratio', 'margin_buy_ratio_percentile']],
        on='trade_date',
        how='outer'
    ).sort_values('trade_date').reset_index(drop=True)
    df['score_turnover'] = (df['turnover_decline_rate'] + 100).clip(lower=0, upper=100)
    df['score_margin'] = df['margin_buy_ratio_percentile'].clip(lower=0, upper=100)
    df['emotion_score'] = df['score_turnover'] * 0.6 + df['score_margin'] * 0.4
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return df[(df['trade_date'] >= start_ts) & (df['trade_date'] <= end_ts)].copy()


def prepare_money_dimension_data(start_date, end_date, config):
    """
    准备资金维度原始数据与得分结果。
    """
    history_start = shift_start_date(
        start_date,
        config['data']['history_buffer_years'],
        config['data']['money_min_history_start']
    )
    money_config, _ = money_thermo.load_parameter_config()
    df_raw = money_thermo.get_holdertrade_data(
        start_date=history_start,
        end_date=end_date,
        chunk_days=money_config['data']['chunk_days'],
        include_holder_types=money_config['data'].get('include_holder_types')
    )
    if df_raw is None or len(df_raw) == 0:
        raise ValueError('资金维度增减持数据获取失败')

    df_repurchase = None
    if money_config.get('repurchase', {}).get('enabled', False):
        df_repurchase = money_thermo.get_repurchase_data(
            start_date=history_start,
            end_date=end_date,
            chunk_days=money_config['data']['chunk_days'],
            include_proc=money_config['repurchase'].get('include_proc')
        )

    df_weekly, df_monthly = money_thermo.aggregate_holdertrade(df_raw, df_repurchase=df_repurchase)
    fund_result = money_thermo.compute_fund_score(df_weekly, df_monthly, money_config)
    fund_result['money_config'] = money_config
    return fund_result


def build_money_dimension_series(fund_result, target_dates):
    """
    将月度资金得分扩展为可与日度温度曲线对齐的序列。
    """
    df_monthly = fund_result.get('df_monthly')
    money_config = fund_result.get('money_config', {})
    if df_monthly is None or len(df_monthly) == 0:
        result = target_dates.copy()
        result['money_score'] = 50.0
        return result

    score_config = money_config.get('score', {})
    amount_weight = score_config.get('amount_weight', 0.0)
    count_weight = score_config.get('count_weight', 0.0)
    repurchase_weight = score_config.get('repurchase_weight', 0.0) if fund_result.get('repurchase_enabled') else 0.0
    total_weight = amount_weight + count_weight + repurchase_weight

    df = df_monthly.copy().sort_values('trade_date').reset_index(drop=True)
    df['money_score'] = (
        df['amount_score'] * amount_weight +
        df['count_score'] * count_weight +
        df['repurchase_score'] * repurchase_weight
    ) / total_weight if total_weight > 0 else 50.0
    df['effective_date'] = (df['trade_date'] + pd.offsets.MonthEnd(0)).dt.normalize()

    daily = target_dates[['trade_date']].sort_values('trade_date').reset_index(drop=True)
    mapped = pd.merge_asof(
        daily,
        df[['effective_date', 'money_score', 'amount_score', 'count_score', 'repurchase_score']].sort_values('effective_date'),
        left_on='trade_date',
        right_on='effective_date',
        direction='backward'
    )
    mapped = mapped.drop(columns=['effective_date'])
    for column in ['money_score', 'amount_score', 'count_score', 'repurchase_score']:
        mapped[column] = mapped[column].fillna(50.0)
    return mapped


def prepare_tech_dimension_data(start_date, end_date, config):
    """
    准备技术维度原始广度数据。
    """
    history_start = shift_start_date(
        start_date,
        config['data']['history_buffer_years'],
        config['data']['tech_min_history_start']
    )
    tech_config = tech_thermo.load_parameter_config()
    df_daily = tech_thermo.get_all_stock_daily(history_start, end_date)
    if df_daily is None or len(df_daily) == 0:
        raise ValueError('技术维度日线数据获取失败')

    df_breadth = tech_thermo.compute_breadth_indicators(df_daily, tech_config)
    return df_breadth, tech_config


def build_tech_dimension_series(df_breadth, start_date, end_date, tech_config):
    """
    构建技术维度的日度得分序列。
    """
    df = df_breadth.copy().sort_values('trade_date').reset_index(drop=True)
    df['score_cum_diff'] = df['cumulative_diff_norm']
    fallback_cum_diff = expanding_percentile(df['cumulative_diff'])
    df['score_cum_diff'] = df['score_cum_diff'].fillna(fallback_cum_diff)
    df['score_mcclellan'] = expanding_percentile(df['mcclellan'].ffill())
    arms_pct = expanding_percentile(df['arms_ma'].ffill())
    df['score_arms'] = 100 - arms_pct
    df['score_arms'] = df['score_arms'].fillna(50.0)

    weights = tech_config['scoring']['weights']
    df['tech_score'] = (
        df['score_cum_diff'] * weights['cum_diff'] +
        df['score_mcclellan'] * weights['mcclellan'] +
        df['score_arms'] * weights['arms']
    )
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return df[(df['trade_date'] >= start_ts) & (df['trade_date'] <= end_ts)].copy()


def build_temperature_timeseries(
    start_date,
    end_date,
    config,
    df_value,
    emotion_tuple,
    fund_result,
    df_breadth,
    tech_config
):
    """
    构建综合市场温度与价值锚定温度的日度时间序列。
    """
    hs300_df = get_index_close('000300.SH', start_date, end_date)
    cyb_df = get_index_close('399006.SZ', start_date, end_date)
    if hs300_df is None or len(hs300_df) == 0:
        raise ValueError('沪深300指数数据获取失败，无法构建时间序列')

    base = hs300_df[['trade_date', 'close']].rename(columns={'close': 'hs300_close'}).copy()
    if cyb_df is not None and len(cyb_df) > 0:
        base = pd.merge(
            base,
            cyb_df[['trade_date', 'close']].rename(columns={'close': 'cyb_close'}),
            on='trade_date',
            how='left'
        )
        base['cyb_close'] = base['cyb_close'].ffill()
    else:
        base['cyb_close'] = np.nan

    value_series = build_value_dimension_series(df_value, start_date, end_date, config)
    df_t, _, df_m = emotion_tuple
    emotion_series = build_emotion_dimension_series(df_t, df_m, start_date, end_date)
    money_series = build_money_dimension_series(fund_result, base[['trade_date']])
    tech_series = build_tech_dimension_series(df_breadth, start_date, end_date, tech_config)

    series_df = base.copy()
    series_df = pd.merge(
        series_df,
        value_series[['trade_date', 'value_score', 'spread_percentile', 'pe_percentile']],
        on='trade_date',
        how='left'
    )
    series_df = pd.merge(
        series_df,
        emotion_series[['trade_date', 'emotion_score', 'score_turnover', 'score_margin']],
        on='trade_date',
        how='left'
    )
    series_df = pd.merge(
        series_df,
        money_series[['trade_date', 'money_score', 'amount_score', 'count_score', 'repurchase_score']],
        on='trade_date',
        how='left'
    )
    series_df = pd.merge(
        series_df,
        tech_series[['trade_date', 'tech_score', 'score_cum_diff', 'score_mcclellan', 'score_arms']],
        on='trade_date',
        how='left'
    )

    for column in [
        'value_score', 'spread_percentile', 'pe_percentile',
        'emotion_score', 'score_turnover', 'score_margin',
        'money_score', 'amount_score', 'count_score', 'repurchase_score',
        'tech_score', 'score_cum_diff', 'score_mcclellan', 'score_arms'
    ]:
        series_df[column] = series_df[column].ffill()

    weights = config['weights']
    anchor_weights = config['anchor']
    series_df['composite_score'] = (
        series_df['value_score'] * weights['value'] +
        series_df['emotion_score'] * weights['emotion'] +
        series_df['money_score'] * weights['money'] +
        series_df['tech_score'] * weights['tech']
    )
    series_df['anchor_score'] = (
        series_df['value_score'] * anchor_weights['value'] +
        series_df['money_score'] * anchor_weights['money'] +
        series_df['emotion_score'] * anchor_weights['emotion'] +
        series_df['tech_score'] * anchor_weights['tech']
    )
    series_df['hs300_norm'] = series_df['hs300_close'] / series_df['hs300_close'].iloc[0] * 100

    if series_df['cyb_close'].notna().any():
        first_cyb_close = series_df['cyb_close'].dropna().iloc[0]
        series_df['cyb_norm'] = series_df['cyb_close'] / first_cyb_close * 100
    else:
        series_df['cyb_norm'] = np.nan

    return series_df


def export_temperature_series_csv(series_df, output_path='市场温度时间序列.csv'):
    """
    导出市场温度时间序列 CSV。
    """
    export_df = series_df.copy()
    export_df['trade_date'] = export_df['trade_date'].dt.strftime('%Y-%m-%d')
    export_df.to_csv(build_output_path(output_path), index=False, encoding='utf-8-sig')
    print(f'时间序列数据已导出: {output_path}')


def plot_temperature_timeseries(series_df, config, output_path='市场温度时间序列分析图.png'):
    """
    绘制温度时间序列，并叠加沪深300和创业板指数。
    """
    main_thresholds = config['plot_thresholds']['timeseries_main']
    dimension_thresholds = config['plot_thresholds']['timeseries_dimensions']
    main_cold = main_thresholds['cold']
    main_hot = main_thresholds['hot']
    dimension_cold = dimension_thresholds['cold']
    dimension_hot = dimension_thresholds['hot']

    fig, axes = plt.subplots(2, 1, figsize=(17, 12), sharex=True)

    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    line1, = ax1.plot(series_df['trade_date'], series_df['composite_score'], color='teal', linewidth=2.2, label='综合市场温度')
    line2, = ax1.plot(series_df['trade_date'], series_df['anchor_score'], color='firebrick', linewidth=2.2, label='价值锚定温度')
    line3, = ax1_twin.plot(series_df['trade_date'], series_df['hs300_norm'], color='royalblue', linewidth=1.8, alpha=0.9, label='沪深300(起点=100)')

    handles = [line1, line2, line3]
    labels = [handle.get_label() for handle in handles]
    if series_df['cyb_norm'].notna().any():
        line4, = ax1_twin.plot(series_df['trade_date'], series_df['cyb_norm'], color='darkorange', linewidth=1.8, alpha=0.9, label='创业板指(起点=100)')
        handles.append(line4)
        labels.append(line4.get_label())

    ax1.axhspan(0, main_cold, color='green', alpha=0.08)
    ax1.axhspan(main_hot, 100, color='red', alpha=0.08)
    ax1.axhline(main_cold, color='green', linestyle='--', linewidth=1)
    ax1.axhline(main_hot, color='red', linestyle='--', linewidth=1)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('温度得分')
    ax1_twin.set_ylabel('指数归一化点位')
    ax1.set_title('市场温度时间序列与指数走势对照')
    ax1.grid(True, alpha=0.25)
    ax1.legend(handles, labels, loc='upper left', ncol=2)
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.set_xlabel('时间')

    ax2 = axes[1]
    ax2.plot(series_df['trade_date'], series_df['value_score'], color='steelblue', linewidth=2, label='价值')
    ax2.plot(series_df['trade_date'], series_df['emotion_score'], color='seagreen', linewidth=1.8, label='情绪')
    ax2.plot(series_df['trade_date'], series_df['money_score'], color='darkorange', linewidth=1.8, label='资金')
    ax2.plot(series_df['trade_date'], series_df['tech_score'], color='mediumpurple', linewidth=1.8, label='技术')
    ax2.axhspan(0, dimension_cold, color='green', alpha=0.06)
    ax2.axhspan(dimension_hot, 100, color='red', alpha=0.06)
    ax2.axhline(dimension_cold, color='green', linestyle='--', linewidth=1)
    ax2.axhline(dimension_hot, color='red', linestyle='--', linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('维度得分')
    ax2.set_title('四维温度曲线')
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='upper left', ncol=4)
    ax2.set_xlabel('时间')

    year_locator = mdates.YearLocator()
    year_formatter = mdates.DateFormatter('%Y')
    quarter_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])

    for axis in axes:
        axis.xaxis.set_major_locator(year_locator)
        axis.xaxis.set_major_formatter(year_formatter)
        axis.xaxis.set_minor_locator(quarter_locator)
        axis.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.8, alpha=0.35)

    plt.tight_layout()
    plt.savefig(build_output_path(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'时间序列图已保存为: {output_path}')


def get_value_dimension_score(end_date, config):
    """
    获取价值维度结果。
    """
    df_value = prepare_value_dimension_data(end_date, config)
    current_date = df_value['trade_date'].iloc[-1]
    return compute_value_score(df_value, current_date, config)


def get_emotion_dimension_score(start_date, end_date, config):
    """
    获取情绪维度结果。
    """
    df_t, df_r, df_m = prepare_emotion_dimension_data(start_date, end_date, config)
    current_date = df_t[df_t['trade_date'] <= pd.to_datetime(end_date)]['trade_date'].max()
    return compute_emotion_score(df_t, df_r, df_m, current_date=current_date)


def get_money_dimension_score(start_date, end_date, config):
    """
    获取资金维度结果。
    """
    result = prepare_money_dimension_data(start_date, end_date, config)
    result['dimension'] = '资金'
    result['total_score'] = result['fund_score']
    result['state'] = classify_temperature(result['fund_score'])
    result['current_date'] = pd.to_datetime(end_date)
    return result


def get_tech_dimension_score(start_date, end_date, config):
    """
    获取技术维度结果。
    """
    df_breadth, tech_config = prepare_tech_dimension_data(start_date, end_date, config)
    result = tech_thermo.compute_breadth_score(df_breadth, tech_config)
    result['dimension'] = '技术'
    result['state'] = classify_temperature(result['total_score'])
    result['current_date'] = pd.to_datetime(df_breadth['trade_date'].iloc[-1])
    return result


def compute_composite_score(results, config):
    """
    同时计算综合市场温度和价值锚定温度。
    """
    weights = config['weights']
    total_score = (
        results['value']['total_score'] * weights['value'] +
        results['emotion']['total_score'] * weights['emotion'] +
        results['money']['total_score'] * weights['money'] +
        results['tech']['total_score'] * weights['tech']
    )
    anchor_weights = config['anchor']
    anchor_score = (
        results['value']['total_score'] * anchor_weights['value'] +
        results['money']['total_score'] * anchor_weights['money'] +
        results['emotion']['total_score'] * anchor_weights['emotion'] +
        results['tech']['total_score'] * anchor_weights['tech']
    )
    return {
        'total_score': total_score,
        'state': classify_temperature(total_score),
        'anchor_score': anchor_score,
        'anchor_state': classify_temperature(anchor_score),
        'weights': weights,
        'anchor_weights': anchor_weights,
    }


def build_analysis_text(results, composite):
    """
    生成综合分析结论文本。
    """
    ordered = sorted(
        [(name, detail['total_score']) for name, detail in results.items()],
        key=lambda item: item[1],
        reverse=True
    )
    strongest_name, strongest_score = ordered[0]
    weakest_name, weakest_score = ordered[-1]

    support_dims = [name for name, detail in results.items() if detail['total_score'] >= 60]
    drag_dims = [name for name, detail in results.items() if detail['total_score'] <= 40]

    parts = [
        f'综合市场温度为 {composite["total_score"]:.1f} 分，当前处于“{composite["state"]}”阶段。',
        f'价值锚定温度为 {composite["anchor_score"]:.1f} 分，当前处于“{composite["anchor_state"]}”阶段。',
        f'四个维度中，最强支撑来自{strongest_name}维度（{strongest_score:.1f}分），最主要拖累来自{weakest_name}维度（{weakest_score:.1f}分）。',
    ]

    if support_dims:
        parts.append(f'当前偏强维度包括：{"、".join(support_dims)}。')
    if drag_dims:
        parts.append(f'当前偏弱维度包括：{"、".join(drag_dims)}。')

    value = results['value']
    emotion = results['emotion']
    money = results['money']
    tech = results['tech']

    parts.append(
        f'价值维度上，股债收益差分位为 {value["spread_percentile"]:.1f}%，PE 分位为 {value["pe_percentile"]:.1f}%，说明估值相对吸引力与绝对估值处于共同校验状态。'
    )
    parts.append(
        f'情绪维度上，成交额萎缩率得分 {emotion["score_turnover"]:.1f}、融资买入占比得分 {emotion["score_margin"]:.1f}，反映当前情绪修复力度。'
    )
    parts.append(
        f'资金维度上，金额口径得分 {money["amount_score"]:.1f}、笔数口径得分 {money["count_score"]:.1f}、回购口径得分 {money["repurchase_score"]:.1f}。'
    )
    parts.append(
        f'技术维度上，新高新低得分 {tech["score_cum_diff"]:.1f}、麦克莱伦得分 {tech["score_mcclellan"]:.1f}、阿姆氏得分 {tech["score_arms"]:.1f}。'
    )
    parts.append(
        f'其中“价值锚定温度”采用价值主导口径：价值占 70%，资金占 15%，情绪占 10%，技术占 5%，更偏向中期顶底判断。'
    )
    return '\n'.join(parts)


def export_summary_csv(results, composite, analysis_text, output_path='综合市场温度得分.csv'):
    """
    导出综合市场温度摘要 CSV。
    """
    summary = pd.DataFrame([
        {
            '评估日期': max(detail['current_date'] for detail in results.values()).strftime('%Y-%m-%d'),
            '综合市场温度得分': round(composite['total_score'], 2),
            '综合状态': composite['state'],
            '价值锚定温度得分': round(composite['anchor_score'], 2),
            '价值锚定状态': composite['anchor_state'],
            '价值得分': round(results['value']['total_score'], 2),
            '情绪得分': round(results['emotion']['total_score'], 2),
            '资金得分': round(results['money']['total_score'], 2),
            '技术得分': round(results['tech']['total_score'], 2),
            '价值_股债收益差分位': round(results['value']['spread_percentile'], 2),
            '价值_PE分位': round(results['value']['pe_percentile'], 2),
            '情绪_成交额萎缩率': round(results['emotion']['decline_rate'], 2),
            '情绪_融资买入占比历史分位': round(results['emotion']['margin_percentile'], 2),
            '资金_金额口径得分': round(results['money']['amount_score'], 2),
            '资金_笔数口径得分': round(results['money']['count_score'], 2),
            '资金_回购口径得分': round(results['money']['repurchase_score'], 2),
            '技术_新高新低得分': round(results['tech']['score_cum_diff'], 2),
            '技术_麦克莱伦得分': round(results['tech']['score_mcclellan'], 2),
            '技术_阿姆氏得分': round(results['tech']['score_arms'], 2),
            '综合分析': analysis_text,
        }
    ])
    summary.to_csv(build_output_path(output_path), index=False, encoding='utf-8-sig')
    print(f'综合摘要已导出: {output_path}')


def plot_composite_dashboard(results, composite, config, output_path='综合市场温度分析图.png'):
    """
    绘制综合市场温度图表。
    """
    dashboard_thresholds = config['plot_thresholds']['dashboard']
    dashboard_cold = dashboard_thresholds['cold']
    dashboard_hot = dashboard_thresholds['hot']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    dimension_order = ['value', 'emotion', 'money', 'tech']
    labels = ['价值', '情绪', '资金', '技术']
    scores = [results[key]['total_score'] for key in dimension_order]
    weights = [composite['weights'][key] * 100 for key in dimension_order]

    ax1 = axes[0, 0]
    bars = ax1.bar(labels, scores, color=['steelblue', 'seagreen', 'darkorange', 'mediumpurple'], alpha=0.78)
    ax1.axhline(dashboard_cold, color='green', linestyle='--', linewidth=1, label='极冷阈值')
    ax1.axhline(dashboard_hot, color='red', linestyle='--', linewidth=1, label='极热阈值')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('维度得分')
    ax1.set_title('四维得分对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, score, weight in zip(bars, scores, weights):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f'{score:.1f}\n权重{weight:.0f}%', ha='center', va='bottom', fontsize=10)

    ax2 = axes[0, 1]
    dual_labels = ['综合温度', '价值锚定温度']
    dual_scores = [composite['total_score'], composite['anchor_score']]
    bars2 = ax2.bar(dual_labels, dual_scores, color=['teal', 'firebrick'], alpha=0.78)
    ax2.axhline(dashboard_cold, color='green', linestyle='--', linewidth=1)
    ax2.axhline(dashboard_hot, color='red', linestyle='--', linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('结果得分')
    ax2.set_title('双结果对比')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, value, state in zip(bars2, dual_scores, [composite['state'], composite['anchor_state']]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{value:.1f}\n{state}', ha='center', va='bottom', fontsize=10)

    ax3 = axes[1, 0]
    ax3.axis('off')
    analysis_text = build_analysis_text(results, composite)
    ax3.text(0.02, 0.98, '综合分析\n\n' + analysis_text, va='top', ha='left', fontsize=11)

    ax4 = axes[1, 1]
    total_score = composite['total_score']
    theta = np.linspace(np.pi, 0, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax4.plot(x, y, color='lightgray', linewidth=15, solid_capstyle='round')
    ax4.plot(x, y, color=plt.cm.RdYlGn_r(total_score / 100), linewidth=12, solid_capstyle='round')
    angle = np.pi * (1 - total_score / 100)
    ax4.arrow(0, 0, 0.8 * np.cos(angle), 0.8 * np.sin(angle), head_width=0.08, head_length=0.08, fc='black', ec='black')
    ax4.text(0, -0.16, f'{total_score:.1f}', ha='center', va='center', fontsize=22)
    ax4.text(0, -0.34, '综合市场温度', ha='center', va='center', fontsize=12)
    ax4.text(0, -0.48, f'价值锚定 {composite["anchor_score"]:.1f}', ha='center', va='center', fontsize=11, color='firebrick')
    ax4.text(-1.15, -0.08, '冰点', fontsize=10, ha='center')
    ax4.text(1.15, -0.08, '沸腾', fontsize=10, ha='center')
    ax4.text(0, 1.05, composite['state'], fontsize=10, ha='center')
    ax4.set_xlim(-1.3, 1.3)
    ax4.set_ylim(-0.45, 1.2)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('综合市场温度得分')

    plt.tight_layout()
    plt.savefig(build_output_path(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'综合图表已保存为: {output_path}')
    return analysis_text


def print_summary(results, composite, analysis_text, start_date, end_date):
    """
    在控制台输出综合分析摘要。
    """
    print('\n' + '=' * 70)
    print('综合市场温度分析结果')
    print('=' * 70)
    print(f'分析区间: {start_date} ~ {end_date}')
    print(f'评估日期: {max(detail["current_date"] for detail in results.values()).strftime("%Y-%m-%d")}')
    print('-' * 70)
    print(f'价值维度: {results["value"]["total_score"]:.1f} / 100')
    print(f'情绪维度: {results["emotion"]["total_score"]:.1f} / 100')
    print(f'资金维度: {results["money"]["total_score"]:.1f} / 100')
    print(f'技术维度: {results["tech"]["total_score"]:.1f} / 100')
    print('-' * 70)
    print(f'综合得分: {composite["total_score"]:.1f} / 100')
    print(f'综合状态: {composite["state"]}')
    print(f'价值锚定得分: {composite["anchor_score"]:.1f} / 100')
    print(f'价值锚定状态: {composite["anchor_state"]}')
    print('-' * 70)
    print(analysis_text)
    print('=' * 70)


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='综合市场温度计分析脚本')
    parser.add_argument('--start', type=str, default='20240101', help='分析起始日期，格式 YYYYMMDD')
    parser.add_argument('--end', type=str, default=None, help='分析结束日期，格式 YYYYMMDD')
    parser.add_argument('--config', type=str, default=None, help='综合参数配置文件路径')
    return parser.parse_args()


def main():
    """
    执行综合市场温度分析主流程。
    """
    args = parse_args()
    config, config_path = load_parameter_config(args.config)
    start_date = normalize_date_str(args.start)
    end_date = normalize_date_str(args.end or datetime.now().strftime('%Y%m%d'))

    print('\n' + '=' * 70)
    print('综合市场温度计')
    print(f'参数文件: {config_path}')
    print(f'分析区间: {start_date} ~ {end_date}')
    print('=' * 70)

    df_value = prepare_value_dimension_data(end_date, config)
    emotion_tuple = prepare_emotion_dimension_data(start_date, end_date, config)
    fund_result = prepare_money_dimension_data(start_date, end_date, config)
    df_breadth, tech_config = prepare_tech_dimension_data(start_date, end_date, config)

    value_result = compute_value_score(df_value, df_value['trade_date'].iloc[-1], config)
    df_t, df_r, df_m = emotion_tuple
    emotion_current_date = df_t[df_t['trade_date'] <= pd.to_datetime(end_date)]['trade_date'].max()
    emotion_result = compute_emotion_score(df_t, df_r, df_m, current_date=emotion_current_date)
    fund_result['dimension'] = '资金'
    fund_result['total_score'] = fund_result['fund_score']
    fund_result['state'] = classify_temperature(fund_result['fund_score'])
    fund_result['current_date'] = pd.to_datetime(end_date)
    tech_result = tech_thermo.compute_breadth_score(df_breadth, tech_config)
    tech_result['dimension'] = '技术'
    tech_result['state'] = classify_temperature(tech_result['total_score'])
    tech_result['current_date'] = pd.to_datetime(df_breadth['trade_date'].iloc[-1])

    results = {
        'value': value_result,
        'emotion': emotion_result,
        'money': fund_result,
        'tech': tech_result,
    }
    composite = compute_composite_score(results, config)
    analysis_text = plot_composite_dashboard(results, composite, config)
    export_summary_csv(results, composite, analysis_text)
    print_summary(results, composite, analysis_text, start_date, end_date)

    series_df = build_temperature_timeseries(
        start_date,
        end_date,
        config,
        df_value,
        emotion_tuple,
        fund_result,
        df_breadth,
        tech_config
    )
    export_temperature_series_csv(series_df)
    plot_temperature_timeseries(series_df, config)


configure_console_encoding()
TOKEN = os.getenv('TUSHARE_API_KEY') or load_env_value('TUSHARE_API_KEY')
pro = ts.pro_api(TOKEN)


if __name__ == '__main__':
    main()

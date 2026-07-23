import argparse
import os
import sys
import tomllib
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def configure_console_encoding():
    """
    配置控制台输出编码，避免 Windows 下打印中文时报错。
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')


def get_project_root():
    """
    获取项目根目录，用于统一定位 .env、配置文件和输出文件。
    """
    return Path(__file__).resolve().parent.parent


def build_output_path(filename):
    """
    构造输出文件路径，确保图表和 CSV 统一写到 outputs 目录。
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


def init_tushare_client():
    """
    初始化 Tushare Pro 客户端。
    """
    token = os.getenv('TUSHARE_API_KEY') or load_env_value('TUSHARE_API_KEY')
    return ts.pro_api(token)


DEFAULT_CONFIG = {
    'data': {
        'start_date': '20150101',
        'chunk_days': 180,
        'include_holder_types': ['C', 'P', 'G'],
    },
    'score': {
        'amount_weight': 0.4,
        'count_weight': 0.3,
        'repurchase_weight': 0.3,
    },
    'weekly': {
        'ma_window': 4,
        'ma_min_periods': 1,
    },
    'threshold': {
        'extreme_positive': 2.0,
        'positive': 1.5,
        'neutral': 1.0,
        'weak_negative': 0.7,
        'negative': 0.3,
    },
    'signal': {
        'strong_positive_score': 70,
        'strong_negative_score': 30,
    },
    'repurchase': {
        'enabled': True,
        'include_proc': ['实施', '完成'],
    },
}


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


def load_parameter_config(config_path=None):
    """
    加载参数配置文件，不存在时退回默认配置。
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'money_thermo_params.toml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = get_project_root() / config_path

    config = DEFAULT_CONFIG
    if config_path.exists():
        with config_path.open('rb') as file:
            loaded = tomllib.load(file)
        config = merge_config(DEFAULT_CONFIG, loaded)
        print(f'已加载参数文件: {config_path}')
    else:
        print(f'未找到参数文件，使用默认参数: {config_path}')

    return config, config_path


def daterange_chunks(start_date, end_date, chunk_days):
    """
    将日期区间拆分为多个子区间，规避单次接口返回行数上限。
    """
    current_start = datetime.strptime(start_date, '%Y%m%d')
    final_end = datetime.strptime(end_date, '%Y%m%d')

    while current_start <= final_end:
        current_end = min(current_start + timedelta(days=chunk_days - 1), final_end)
        yield current_start.strftime('%Y%m%d'), current_end.strftime('%Y%m%d')
        current_start = current_end + timedelta(days=1)


def get_holdertrade_data(start_date='20100101', end_date=None, chunk_days=180, include_holder_types=None):
    """
    获取全市场股东增减持明细数据，并做字段标准化。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if include_holder_types is None:
        include_holder_types = ['C', 'P', 'G']

    print(f'\n获取产业资本增减持数据: {start_date} ~ {end_date}')

    try:
        chunks = []
        for chunk_start, chunk_end in daterange_chunks(start_date, end_date, chunk_days):
            df_chunk = pro.stk_holdertrade(start_date=chunk_start, end_date=chunk_end)
            if df_chunk is not None and len(df_chunk) > 0:
                chunks.append(df_chunk)

        if not chunks:
            print('未获取到增减持记录')
            return None

        df = pd.concat(chunks, ignore_index=True).drop_duplicates().reset_index(drop=True)

        if 'ann_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['ann_date'], errors='coerce')
        elif 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        else:
            raise KeyError('增减持数据缺少公告日期字段')

        if 'in_de' in df.columns and 'change_type' not in df.columns:
            df['change_type'] = df['in_de'].map({'IN': '增持', 'DE': '减持'}).fillna(df['in_de'])
        else:
            df['change_type'] = df['change_type'].fillna('').astype(str)

        if 'holder_type' in df.columns and include_holder_types:
            df = df[df['holder_type'].isin(include_holder_types)].copy()

        if 'change_vol' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['change_vol']

        for column in ['volume', 'avg_price', 'amount', 'after_share', 'total_share']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

        if 'amount' not in df.columns:
            df['amount'] = np.nan

        fallback_amount = df['volume'] * df['avg_price'] / 10000
        df['amount'] = df['amount'].fillna(fallback_amount)

        missing_amount_count = df['amount'].isna().sum()
        if missing_amount_count > 0:
            print(f'提示: 有 {missing_amount_count} 条记录缺少均价信息，金额按 0 处理')
            df['amount'] = df['amount'].fillna(0)

        df = df.dropna(subset=['trade_date']).copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        print(f'增减持记录数: {len(df)}')
        if 'holder_type' in df.columns:
            print(f'股东类型过滤: {include_holder_types}')
        print(f'日期范围: {df["trade_date"].min().strftime("%Y-%m-%d")} ~ {df["trade_date"].max().strftime("%Y-%m-%d")}')
        print(f'增持记录数: {len(df[df["change_type"].str.contains("增", na=False)])}')
        print(f'减持记录数: {len(df[df["change_type"].str.contains("减", na=False)])}')
        print(f'金额覆盖率: {df["amount"].notna().mean() * 100:.1f}%')
        return df

    except Exception as exc:
        print(f'Tushare 增减持数据获取失败: {exc}')
        print('尝试使用 AkShare 作为参考来源...')

    try:
        df_ak = ak.stock_dzjy_mrmx()
        if df_ak is not None and len(df_ak) > 0:
            print(f'AkShare 大宗交易参考记录数: {len(df_ak)}')
            print('注意: 大宗交易不等同于增减持，当前脚本不会直接混入该数据。')
    except Exception as exc:
        print(f'AkShare 参考数据获取失败: {exc}')

    return None


def get_repurchase_data(start_date='20100101', end_date=None, chunk_days=180, include_proc=None):
    """
    获取公司回购数据，默认只纳入“实施”和“完成”阶段。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if include_proc is None:
        include_proc = ['实施', '完成']

    print(f'\n获取公司回购数据: {start_date} ~ {end_date}')

    try:
        chunks = []
        for chunk_start, chunk_end in daterange_chunks(start_date, end_date, chunk_days):
            df_chunk = pro.repurchase(start_date=chunk_start, end_date=chunk_end)
            if df_chunk is not None and len(df_chunk) > 0:
                chunks.append(df_chunk)

        if not chunks:
            print('未获取到回购记录')
            return None

        df = pd.concat(chunks, ignore_index=True).drop_duplicates().reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['ann_date'], errors='coerce')
        df['proc'] = df['proc'].fillna('').astype(str)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') / 10000
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
        df = df.dropna(subset=['trade_date']).copy()
        df = df[df['proc'].isin(include_proc)].copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        if len(df) == 0:
            print(f'回购记录存在，但在进度过滤 {include_proc} 后为空')
            return None

        print(f'回购记录数: {len(df)}')
        print(f'回购进度过滤: {include_proc}')
        print(f'日期范围: {df["trade_date"].min().strftime("%Y-%m-%d")} ~ {df["trade_date"].max().strftime("%Y-%m-%d")}')
        print(f'回购金额合计: {df["amount"].fillna(0).sum():.2f} 万元')
        return df

    except Exception as exc:
        print(f'回购数据获取失败: {exc}')
        return None


def normalize_holdertrade_data(df):
    """
    规范化增减持数据，补齐金额字段并过滤无效记录。
    """
    if df is None or len(df) == 0:
        return None

    normalized = df.copy()
    normalized['trade_date'] = pd.to_datetime(normalized['trade_date'], errors='coerce')
    normalized = normalized.dropna(subset=['trade_date']).copy()
    normalized['change_type'] = normalized['change_type'].fillna('').astype(str)

    if 'change_vol' in normalized.columns and 'volume' not in normalized.columns:
        normalized['volume'] = normalized['change_vol']

    for column in ['volume', 'avg_price', 'amount']:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors='coerce')

    if 'amount' not in normalized.columns:
        normalized['amount'] = np.nan

    fallback_amount = normalized['volume'] * normalized['avg_price'] / 10000
    normalized['amount'] = normalized['amount'].fillna(fallback_amount)
    normalized['amount'] = normalized['amount'].fillna(0)

    buy_mask = normalized['change_type'].str.contains('增', na=False)
    sell_mask = normalized['change_type'].str.contains('减', na=False)
    normalized = normalized[buy_mask | sell_mask].copy()

    return normalized if len(normalized) > 0 else None


def normalize_repurchase_data(df):
    """
    规范化回购数据，统一日期与金额字段。
    """
    if df is None or len(df) == 0:
        return None

    normalized = df.copy()
    normalized['trade_date'] = pd.to_datetime(normalized['trade_date'], errors='coerce')
    normalized['amount'] = pd.to_numeric(normalized['amount'], errors='coerce').fillna(0)
    normalized = normalized.dropna(subset=['trade_date']).copy()
    return normalized if len(normalized) > 0 else None


def merge_period_metrics(base_df, extra_df, period_column):
    """
    按周期字段合并两个聚合结果，并将数值列缺失补为 0。
    """
    if base_df is None or len(base_df) == 0:
        merged = extra_df.copy() if extra_df is not None else None
    elif extra_df is None or len(extra_df) == 0:
        merged = base_df.copy()
    else:
        merged = pd.merge(base_df, extra_df, on=period_column, how='outer')

    if merged is None or len(merged) == 0:
        return merged

    numeric_columns = merged.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        merged[numeric_columns] = merged[numeric_columns].fillna(0)
    return merged


def aggregate_holdertrade(df, df_repurchase=None):
    """
    对增减持和回购数据进行周度、月度双轨聚合。
    """
    if df is None or len(df) == 0:
        print('无数据可聚合')
        return None, None

    print('\n' + '=' * 70)
    print('开始双轨聚合计算...')
    print('=' * 70)

    normalized = normalize_holdertrade_data(df)
    if normalized is None or len(normalized) == 0:
        print('原始数据无法转换为有效的增减持序列')
        return None, None

    repurchase_normalized = normalize_repurchase_data(df_repurchase)

    normalized['year_month'] = normalized['trade_date'].dt.to_period('M')
    normalized['year_week'] = normalized['trade_date'].dt.to_period('W-WED')

    df_buy = normalized[normalized['change_type'].str.contains('增', na=False)].copy()
    df_sell = normalized[normalized['change_type'].str.contains('减', na=False)].copy()

    buy_total = df_buy['amount'].sum()
    sell_total = df_sell['amount'].sum()
    total_ratio = buy_total / sell_total if sell_total > 0 else np.nan

    print(f'总记录数: {len(normalized)}')
    print(f'增持记录: {len(df_buy)}')
    print(f'减持记录: {len(df_sell)}')
    print(f'增持总金额: {buy_total:.2f} 万元')
    print(f'减持总金额: {sell_total:.2f} 万元')
    print(f'净增持金额: {(buy_total - sell_total):.2f} 万元')
    print(f'增减持比: {total_ratio:.2f}' if pd.notna(total_ratio) else '增减持比: 数据不足')

    if repurchase_normalized is not None and len(repurchase_normalized) > 0:
        print(f'回购记录: {len(repurchase_normalized)}')
        print(f'回购总金额: {repurchase_normalized["amount"].sum():.2f} 万元')
    else:
        print('回购记录: 0')

    monthly_buy = df_buy.groupby('year_month')['amount'].sum().reset_index(name='buy_amount')
    monthly_sell = df_sell.groupby('year_month')['amount'].sum().reset_index(name='sell_amount')
    monthly_buy_count = df_buy.groupby('year_month').size().reset_index(name='buy_count')
    monthly_sell_count = df_sell.groupby('year_month').size().reset_index(name='sell_count')

    df_monthly = pd.merge(monthly_buy, monthly_sell, on='year_month', how='outer').fillna(0)
    df_monthly = pd.merge(df_monthly, monthly_buy_count, on='year_month', how='left')
    df_monthly = pd.merge(df_monthly, monthly_sell_count, on='year_month', how='left')
    df_monthly[['buy_count', 'sell_count']] = df_monthly[['buy_count', 'sell_count']].fillna(0)
    df_monthly['net_amount'] = df_monthly['buy_amount'] - df_monthly['sell_amount']
    df_monthly['amount_ratio'] = np.where(
        df_monthly['sell_amount'] > 0,
        df_monthly['buy_amount'] / df_monthly['sell_amount'],
        np.nan
    )
    df_monthly['net_count'] = df_monthly['buy_count'] - df_monthly['sell_count']
    df_monthly['count_ratio'] = np.where(
        df_monthly['sell_count'] > 0,
        df_monthly['buy_count'] / df_monthly['sell_count'],
        np.nan
    )
    df_monthly = pd.merge(
        df_monthly,
        normalized.groupby('year_month').size().reset_index(name='deal_count'),
        on='year_month',
        how='left'
    )

    weekly_buy = df_buy.groupby('year_week')['amount'].sum().reset_index(name='buy_amount')
    weekly_sell = df_sell.groupby('year_week')['amount'].sum().reset_index(name='sell_amount')
    weekly_buy_count = df_buy.groupby('year_week').size().reset_index(name='buy_count')
    weekly_sell_count = df_sell.groupby('year_week').size().reset_index(name='sell_count')

    df_weekly = pd.merge(weekly_buy, weekly_sell, on='year_week', how='outer').fillna(0)
    df_weekly = pd.merge(df_weekly, weekly_buy_count, on='year_week', how='left')
    df_weekly = pd.merge(df_weekly, weekly_sell_count, on='year_week', how='left')
    df_weekly[['buy_count', 'sell_count']] = df_weekly[['buy_count', 'sell_count']].fillna(0)
    df_weekly['net_amount'] = df_weekly['buy_amount'] - df_weekly['sell_amount']
    df_weekly['amount_ratio'] = np.where(
        df_weekly['sell_amount'] > 0,
        df_weekly['buy_amount'] / df_weekly['sell_amount'],
        np.nan
    )
    df_weekly['net_count'] = df_weekly['buy_count'] - df_weekly['sell_count']
    df_weekly['count_ratio'] = np.where(
        df_weekly['sell_count'] > 0,
        df_weekly['buy_count'] / df_weekly['sell_count'],
        np.nan
    )
    df_weekly = pd.merge(
        df_weekly,
        normalized.groupby('year_week').size().reset_index(name='deal_count'),
        on='year_week',
        how='left'
    )

    if repurchase_normalized is not None and len(repurchase_normalized) > 0:
        repurchase_normalized['year_month'] = repurchase_normalized['trade_date'].dt.to_period('M')
        repurchase_normalized['year_week'] = repurchase_normalized['trade_date'].dt.to_period('W-WED')

        monthly_repurchase = repurchase_normalized.groupby('year_month').agg(
            repurchase_amount=('amount', 'sum'),
            repurchase_count=('amount', 'size')
        ).reset_index()
        weekly_repurchase = repurchase_normalized.groupby('year_week').agg(
            repurchase_amount=('amount', 'sum'),
            repurchase_count=('amount', 'size')
        ).reset_index()

        df_monthly = merge_period_metrics(df_monthly, monthly_repurchase, 'year_month')
        df_weekly = merge_period_metrics(df_weekly, weekly_repurchase, 'year_week')
    else:
        df_monthly['repurchase_amount'] = 0.0
        df_monthly['repurchase_count'] = 0
        df_weekly['repurchase_amount'] = 0.0
        df_weekly['repurchase_count'] = 0

    df_monthly['trade_date'] = df_monthly['year_month'].dt.start_time
    df_weekly['trade_date'] = df_weekly['year_week'].dt.start_time

    df_monthly = df_monthly.sort_values('trade_date').reset_index(drop=True)
    df_weekly = df_weekly.sort_values('trade_date').reset_index(drop=True)

    print(f'月度数据: {len(df_monthly)} 个月')
    print(f'周度数据: {len(df_weekly)} 周')
    return df_weekly, df_monthly


def calc_ratio_score(ratio, threshold_config):
    """
    对增减持比值类指标进行 0-100 分打分。
    """
    if pd.isna(ratio):
        return 50
    if ratio >= threshold_config['extreme_positive']:
        return 100
    if ratio >= threshold_config['positive']:
        return 80 + (
            (ratio - threshold_config['positive'])
            / (threshold_config['extreme_positive'] - threshold_config['positive'])
        ) * 20
    if ratio >= threshold_config['neutral']:
        return 50 + (
            (ratio - threshold_config['neutral'])
            / (threshold_config['positive'] - threshold_config['neutral'])
        ) * 30
    if ratio >= threshold_config['weak_negative']:
        return 30 + (
            (ratio - threshold_config['weak_negative'])
            / (threshold_config['neutral'] - threshold_config['weak_negative'])
        ) * 20
    if ratio >= threshold_config['negative']:
        return 10 + (
            (ratio - threshold_config['negative'])
            / (threshold_config['weak_negative'] - threshold_config['negative'])
        ) * 20
    return max(0, ratio / threshold_config['negative'] * 10)


def calc_percentile_series(series):
    """
    将同一口径历史值映射为 0-100 的历史分位分数。
    """
    score_series = pd.Series(50.0, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) == 0:
        return score_series
    score_series.loc[valid.index] = valid.rank(method='average', pct=True) * 100
    return score_series


def compute_fund_score(df_weekly, df_monthly, config):
    """
    根据金额、笔数和回购三条口径计算资金维度综合得分。
    """
    print('\n' + '=' * 70)
    print('计算资金维度得分...')
    print('=' * 70)

    threshold_config = config['threshold']
    weekly_config = config['weekly']
    score_config = config['score']
    repurchase_enabled = config.get('repurchase', {}).get('enabled', False)

    if df_monthly is not None and len(df_monthly) > 0:
        df_monthly = df_monthly.copy()
        df_monthly['amount_score'] = df_monthly['amount_ratio'].apply(lambda x: calc_ratio_score(x, threshold_config))
        df_monthly['count_score'] = df_monthly['count_ratio'].apply(lambda x: calc_ratio_score(x, threshold_config))
        df_monthly['repurchase_score'] = (
            calc_percentile_series(df_monthly['repurchase_amount'])
            if repurchase_enabled else pd.Series(50.0, index=df_monthly.index)
        )

        latest_month_amount_score = df_monthly['amount_score'].iloc[-1]
        latest_month_count_score = df_monthly['count_score'].iloc[-1]
        latest_month_repurchase_score = df_monthly['repurchase_score'].iloc[-1]
        latest_month_amount_ratio = df_monthly['amount_ratio'].iloc[-1]
        latest_month_count_ratio = df_monthly['count_ratio'].iloc[-1]
        latest_month_repurchase_amount = df_monthly['repurchase_amount'].iloc[-1]
        latest_month_repurchase_count = df_monthly['repurchase_count'].iloc[-1]
    else:
        latest_month_amount_score = 50.0
        latest_month_count_score = 50.0
        latest_month_repurchase_score = 50.0
        latest_month_amount_ratio = np.nan
        latest_month_count_ratio = np.nan
        latest_month_repurchase_amount = np.nan
        latest_month_repurchase_count = np.nan

    if df_weekly is not None and len(df_weekly) > 0:
        df_weekly = df_weekly.copy()
        ma_window = weekly_config['ma_window']
        ma_min_periods = weekly_config['ma_min_periods']

        df_weekly['amount_score'] = df_weekly['amount_ratio'].apply(lambda x: calc_ratio_score(x, threshold_config))
        df_weekly['count_score'] = df_weekly['count_ratio'].apply(lambda x: calc_ratio_score(x, threshold_config))
        df_weekly['amount_score_ma'] = df_weekly['amount_score'].rolling(window=ma_window, min_periods=ma_min_periods).mean()
        df_weekly['count_score_ma'] = df_weekly['count_score'].rolling(window=ma_window, min_periods=ma_min_periods).mean()
        df_weekly['amount_ratio_ma'] = df_weekly['amount_ratio'].rolling(window=ma_window, min_periods=ma_min_periods).mean()
        df_weekly['count_ratio_ma'] = df_weekly['count_ratio'].rolling(window=ma_window, min_periods=ma_min_periods).mean()
        df_weekly['repurchase_amount_ma'] = df_weekly['repurchase_amount'].rolling(window=ma_window, min_periods=ma_min_periods).mean()
        df_weekly['repurchase_score'] = (
            calc_percentile_series(df_weekly['repurchase_amount_ma'])
            if repurchase_enabled else pd.Series(50.0, index=df_weekly.index)
        )

        latest_week_amount_score = df_weekly['amount_score_ma'].iloc[-1]
        latest_week_count_score = df_weekly['count_score_ma'].iloc[-1]
        latest_week_repurchase_score = df_weekly['repurchase_score'].iloc[-1]
        latest_week_amount_ratio = df_weekly['amount_ratio'].iloc[-1]
        latest_week_count_ratio = df_weekly['count_ratio'].iloc[-1]
        latest_week_repurchase_amount = df_weekly['repurchase_amount'].iloc[-1]
        latest_week_repurchase_count = df_weekly['repurchase_count'].iloc[-1]
    else:
        latest_week_amount_score = 50.0
        latest_week_count_score = 50.0
        latest_week_repurchase_score = 50.0
        latest_week_amount_ratio = np.nan
        latest_week_count_ratio = np.nan
        latest_week_repurchase_amount = np.nan
        latest_week_repurchase_count = np.nan

    amount_score = latest_month_amount_score if pd.notna(latest_month_amount_ratio) else latest_week_amount_score
    count_score = latest_month_count_score if pd.notna(latest_month_count_ratio) else latest_week_count_score
    repurchase_score = (
        latest_month_repurchase_score
        if pd.notna(latest_month_repurchase_amount)
        else latest_week_repurchase_score
    )

    score_items = [
        (amount_score, score_config['amount_weight']),
        (count_score, score_config['count_weight']),
    ]
    if repurchase_enabled:
        score_items.append((repurchase_score, score_config['repurchase_weight']))

    total_weight = sum(weight for _, weight in score_items)
    fund_score = sum(score * weight for score, weight in score_items) / total_weight if total_weight > 0 else 50.0

    result = {
        'fund_score': fund_score,
        'amount_score': amount_score,
        'count_score': count_score,
        'repurchase_score': repurchase_score,
        'latest_month_amount_score': latest_month_amount_score,
        'latest_month_count_score': latest_month_count_score,
        'latest_month_repurchase_score': latest_month_repurchase_score,
        'latest_month_amount_ratio': latest_month_amount_ratio,
        'latest_month_count_ratio': latest_month_count_ratio,
        'latest_month_repurchase_amount': latest_month_repurchase_amount,
        'latest_month_repurchase_count': latest_month_repurchase_count,
        'latest_week_amount_score': latest_week_amount_score,
        'latest_week_count_score': latest_week_count_score,
        'latest_week_repurchase_score': latest_week_repurchase_score,
        'latest_week_amount_ratio': latest_week_amount_ratio,
        'latest_week_count_ratio': latest_week_count_ratio,
        'latest_week_repurchase_amount': latest_week_repurchase_amount,
        'latest_week_repurchase_count': latest_week_repurchase_count,
        'repurchase_enabled': repurchase_enabled,
        'df_weekly': df_weekly,
        'df_monthly': df_monthly,
    }

    print(f'月度金额比: {latest_month_amount_ratio:.2f}' if pd.notna(latest_month_amount_ratio) else '月度金额比: 不可用')
    print(f'月度笔数比: {latest_month_count_ratio:.2f}' if pd.notna(latest_month_count_ratio) else '月度笔数比: 不可用')
    if repurchase_enabled:
        print(f'月度回购金额: {latest_month_repurchase_amount:.2f} 万元' if pd.notna(latest_month_repurchase_amount) else '月度回购金额: 不可用')
    print(f'金额口径得分: {amount_score:.1f} / 100')
    print(f'笔数口径得分: {count_score:.1f} / 100')
    if repurchase_enabled:
        print(f'回购口径得分: {repurchase_score:.1f} / 100')
    print(f'综合资金维度得分: {fund_score:.1f} / 100')

    return result


def plot_fund_dimension(df_weekly, df_monthly, fund_result):
    """
    生成资金维度图表，展示金额、笔数、回购与综合得分。
    """
    fig, axes = plt.subplots(3, 2, figsize=(17, 16))

    ax1 = axes[0, 0]
    if df_monthly is not None and len(df_monthly) > 0:
        ax1.bar(df_monthly['trade_date'], df_monthly['amount_ratio'], color='steelblue', alpha=0.72, width=20, label='月度金额比')
        ax1.axhline(1.5, color='green', linestyle='--', linewidth=1.2, label='积极阈值(1.5)')
        ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.2, label='消极阈值(0.5)')
        ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='中性(1.0)')
        ax1.set_ylabel('金额比(增持/减持)')
        ax1.set_title('月度金额增减持比')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
    else:
        ax1.text(0.5, 0.5, '月度数据不可用', ha='center', va='center', fontsize=14, transform=ax1.transAxes)

    ax2 = axes[0, 1]
    if df_monthly is not None and len(df_monthly) > 0:
        ax2.bar(df_monthly['trade_date'], df_monthly['count_ratio'], color='mediumpurple', alpha=0.75, width=20, label='月度笔数比')
        ax2.axhline(1.5, color='green', linestyle='--', linewidth=1.2, label='积极阈值(1.5)')
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.2, label='消极阈值(0.5)')
        ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='中性(1.0)')
        ax2.set_ylabel('笔数比(增持/减持)')
        ax2.set_title('月度笔数增减持比')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
    else:
        ax2.text(0.5, 0.5, '月度数据不可用', ha='center', va='center', fontsize=14, transform=ax2.transAxes)

    ax3 = axes[1, 0]
    if df_weekly is not None and len(df_weekly) > 0:
        ax3_twin = ax3.twinx()
        amount_raw, = ax3.plot(df_weekly['trade_date'], df_weekly['amount_ratio'], color='lightblue', alpha=0.35, linewidth=0.9, label='周度金额原始值')
        amount_ma, = ax3.plot(df_weekly['trade_date'], df_weekly['amount_ratio_ma'], color='blue', linewidth=2, label='金额比周均线')
        count_raw, = ax3_twin.plot(df_weekly['trade_date'], df_weekly['count_ratio'], color='plum', alpha=0.35, linewidth=0.9, label='周度笔数原始值')
        count_ma, = ax3_twin.plot(df_weekly['trade_date'], df_weekly['count_ratio_ma'], color='purple', linewidth=2, label='笔数比周均线')
        ax3.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax3_twin.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax3.set_ylabel('金额比', color='blue')
        ax3_twin.set_ylabel('笔数比', color='purple')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3_twin.tick_params(axis='y', labelcolor='purple')
        ax3.set_title('周度金额/笔数双指标')
        ax3.grid(True, alpha=0.3)
        ax3.legend(
            [amount_raw, amount_ma, count_raw, count_ma],
            [amount_raw.get_label(), amount_ma.get_label(), count_raw.get_label(), count_ma.get_label()],
            loc='upper left'
        )
    else:
        ax3.text(0.5, 0.5, '周度数据不可用', ha='center', va='center', fontsize=14, transform=ax3.transAxes)

    ax4 = axes[1, 1]
    if df_monthly is not None and len(df_monthly) > 0:
        ax4_twin = ax4.twinx()
        latest_repurchase_score = df_monthly['repurchase_score'].iloc[-1]
        latest_repurchase_amount = df_monthly['repurchase_amount'].iloc[-1]
        latest_repurchase_date = df_monthly['trade_date'].iloc[-1]
        bar_container = ax4.bar(df_monthly['trade_date'], df_monthly['repurchase_amount'], color='darkorange', alpha=0.68, width=20, label='月度回购金额')
        score_line, = ax4_twin.plot(df_monthly['trade_date'], df_monthly['repurchase_score'], color='firebrick', linewidth=2, label='回购历史分位')
        ax4.set_ylabel('回购金额(万元)', color='darkorange')
        ax4_twin.set_ylabel('回购分位得分', color='firebrick')
        ax4.tick_params(axis='y', labelcolor='darkorange')
        ax4_twin.tick_params(axis='y', labelcolor='firebrick')
        ax4_twin.axhline(50, color='gray', linestyle=':', linewidth=1)
        ax4_twin.scatter([latest_repurchase_date], [latest_repurchase_score], color='firebrick', s=36, zorder=5)
        ax4_twin.annotate(
            f'当前分位 {latest_repurchase_score:.1f}',
            xy=(latest_repurchase_date, latest_repurchase_score),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=10,
            color='firebrick'
        )
        ax4.set_title(f'月度公司回购（当前分位 {latest_repurchase_score:.1f}）')
        ax4.grid(True, alpha=0.3)
        ax4.legend(
            [bar_container, score_line],
            [f'月度回购金额（最近 {latest_repurchase_amount:.0f} 万元）', '回购历史分位'],
            loc='upper left'
        )
    else:
        ax4.text(0.5, 0.5, '回购数据不可用', ha='center', va='center', fontsize=14, transform=ax4.transAxes)

    ax5 = axes[2, 0]
    ax5.axis('off')
    summary_lines = [
        f'金额口径得分: {fund_result["amount_score"]:.1f}',
        f'笔数口径得分: {fund_result["count_score"]:.1f}',
    ]
    if fund_result.get('repurchase_enabled'):
        summary_lines.append(f'回购口径得分: {fund_result["repurchase_score"]:.1f}')
        if pd.notna(fund_result.get('latest_month_repurchase_amount')):
            summary_lines.append(f'最近月度回购金额: {fund_result["latest_month_repurchase_amount"]:.2f} 万元')
            summary_lines.append(f'最近月度回购分位: {fund_result["latest_month_repurchase_score"]:.1f}')
    if pd.notna(fund_result.get('latest_month_amount_ratio')):
        summary_lines.append(f'最近月度金额比: {fund_result["latest_month_amount_ratio"]:.2f}')
    if pd.notna(fund_result.get('latest_month_count_ratio')):
        summary_lines.append(f'最近月度笔数比: {fund_result["latest_month_count_ratio"]:.2f}')
    ax5.text(0.03, 0.95, '资金维度摘要\n\n' + '\n'.join(summary_lines), va='top', ha='left', fontsize=12)

    ax6 = axes[2, 1]
    total_score = fund_result['fund_score']
    theta = np.linspace(np.pi, 0, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax6.plot(x, y, color='lightgray', linewidth=15, solid_capstyle='round')
    ax6.plot(x, y, color=plt.cm.RdYlGn_r(total_score / 100), linewidth=12, solid_capstyle='round')
    angle = np.pi * (1 - total_score / 100)
    ax6.arrow(0, 0, 0.8 * np.cos(angle), 0.8 * np.sin(angle), head_width=0.08, head_length=0.08, fc='black', ec='black')
    ax6.text(0, -0.16, f'{total_score:.1f}', ha='center', va='center', fontsize=22)
    ax6.text(0, -0.34, '资金维度', ha='center', va='center', fontsize=12)
    ax6.text(-1.15, -0.08, '减持偏强', fontsize=10, ha='center')
    ax6.text(1.15, -0.08, '增持偏强', fontsize=10, ha='center')
    ax6.text(0, 1.05, '中性', fontsize=10, ha='center')
    ax6.set_xlim(-1.3, 1.3)
    ax6.set_ylim(-0.45, 1.2)
    ax6.set_aspect('equal')
    ax6.axis('off')
    ax6.set_title('资金维度综合得分')

    plt.tight_layout()
    plt.savefig(build_output_path('资金维度分析图.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('图表已保存为: 资金维度分析图.png')


def export_data(df_weekly, df_monthly):
    """
    导出周度和月度聚合数据到 CSV。
    """
    if df_weekly is not None and len(df_weekly) > 0:
        df_weekly.to_csv(build_output_path('资金维度_周度数据.csv'), index=False, encoding='utf-8-sig')
        print('周度数据已导出: 资金维度_周度数据.csv')

    if df_monthly is not None and len(df_monthly) > 0:
        df_monthly.to_csv(build_output_path('资金维度_月度数据.csv'), index=False, encoding='utf-8-sig')
        print('月度数据已导出: 资金维度_月度数据.csv')


def judge_fund_signal(fund_result, config):
    """
    根据最新月度指标给出资金维度信号判断。
    """
    print('\n' + '=' * 70)
    print('资金维度信号判断')
    print('=' * 70)

    amount_ratio = fund_result['latest_month_amount_ratio']
    count_ratio = fund_result['latest_month_count_ratio']
    amount_score = fund_result['amount_score']
    count_score = fund_result['count_score']
    repurchase_score = fund_result.get('repurchase_score', 50.0)
    repurchase_amount = fund_result.get('latest_month_repurchase_amount', np.nan)
    repurchase_enabled = fund_result.get('repurchase_enabled', False)
    signal_config = config['signal']

    if pd.notna(amount_ratio) or pd.notna(count_ratio):
        strong_pos = signal_config['strong_positive_score']
        strong_neg = signal_config['strong_negative_score']

        if amount_score >= strong_pos and count_score >= strong_pos and (not repurchase_enabled or repurchase_score >= strong_pos):
            signal = '强积极信号'
            detail = '金额、笔数和回购三条口径整体偏强，产业资本与公司层面资金态度较积极。'
        elif amount_score <= strong_neg and count_score <= strong_neg and (not repurchase_enabled or repurchase_score <= strong_neg):
            signal = '强消极信号'
            detail = '金额、笔数和回购三条口径整体偏弱，资金支持不足。'
        elif repurchase_enabled and repurchase_score >= strong_pos:
            signal = '回购托底信号'
            detail = '增减持口径未必全面转强，但公司回购力度位于历史较高位置，存在托底意味。'
        elif amount_score < count_score:
            signal = '分化偏弱信号'
            detail = '增持笔数尚可，但金额口径偏弱，说明大额减持压力更值得关注。'
        elif amount_score > count_score or (repurchase_enabled and repurchase_score >= 60):
            signal = '分化偏强信号'
            detail = '金额口径或回购口径相对更强，说明少数大额增持或公司回购对整体有支撑。'
        else:
            signal = '中性信号'
            detail = '金额与笔数口径接近中性，若有回购也未形成明显偏强或偏弱结论。'

        print(f'当前月度金额比: {amount_ratio:.2f}' if pd.notna(amount_ratio) else '当前月度金额比: 不可用')
        print(f'当前月度笔数比: {count_ratio:.2f}' if pd.notna(count_ratio) else '当前月度笔数比: 不可用')
        if repurchase_enabled:
            print(f'当前月度回购金额: {repurchase_amount:.2f} 万元' if pd.notna(repurchase_amount) else '当前月度回购金额: 不可用')
            print(f'当前回购得分: {repurchase_score:.1f}')
        print(f'信号: {signal}')
        print(f'说明: {detail}')
    else:
        print('月度数据不足，无法给出信号判断')

    print('-' * 70)
    print(f'资金维度综合得分: {fund_result["fund_score"]:.1f} / 100')
    print('=' * 70)


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='资金维度温度计分析脚本')
    parser.add_argument('--config', type=str, default=None, help='参数配置文件路径')
    parser.add_argument('--start', type=str, default=None, help='分析起始日期，格式 YYYYMMDD')
    parser.add_argument('--end', type=str, default=None, help='分析结束日期，格式 YYYYMMDD')
    return parser.parse_args()


def main():
    """
    执行资金维度分析主流程。
    """
    args = parse_args()
    config, config_path = load_parameter_config(args.config)

    print('\n' + '=' * 70)
    print('资金维度指标研究（五维框架 - 权重20%）')
    print('核心指标：产业资本增减持 + 公司回购')
    print('聚合频率：周度 + 月度 双轨')
    print(f'参数文件：{config_path}')
    print('=' * 70)

    start_date = args.start or config['data']['start_date']
    end_date = args.end or datetime.now().strftime('%Y%m%d')

    df_raw = get_holdertrade_data(
        start_date=start_date,
        end_date=end_date,
        chunk_days=config['data']['chunk_days'],
        include_holder_types=config['data'].get('include_holder_types')
    )
    if df_raw is None or len(df_raw) == 0:
        print('\n未获取到任何增减持数据，请检查：')
        print('1. Tushare token 是否正确')
        print('2. `stk_holdertrade` 接口权限是否足够')
        print('3. 网络连接是否正常')
        raise SystemExit('程序终止')

    df_repurchase = None
    if config.get('repurchase', {}).get('enabled', False):
        df_repurchase = get_repurchase_data(
            start_date=start_date,
            end_date=end_date,
            chunk_days=config['data']['chunk_days'],
            include_proc=config['repurchase'].get('include_proc')
        )

    df_weekly, df_monthly = aggregate_holdertrade(df_raw, df_repurchase=df_repurchase)
    if df_weekly is None and df_monthly is None:
        print('\n聚合失败，程序终止')
        raise SystemExit('程序终止')

    fund_result = compute_fund_score(df_weekly, df_monthly, config)
    judge_fund_signal(fund_result, config)
    plot_fund_dimension(fund_result['df_weekly'], fund_result['df_monthly'], fund_result)
    export_data(fund_result['df_weekly'], fund_result['df_monthly'])

    print('\n' + '=' * 70)
    print('资金维度分析完成')
    print('=' * 70)


configure_console_encoding()
pro = init_tushare_client()


if __name__ == '__main__':
    main()

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import tomllib
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tushare as ts
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


DEFAULT_CONFIG = {
    'data': {
        'start_date': '20200101',
        'analyse_date': '20260722',
        'lookback_days': 20,
        'crowded_window': 120,
        'min_history_for_percentile': 20,
        'sw_daily_chunk_trading_days': 100,
        'trade_calendar_exchange': 'SSE',
    },
    'storage': {
        'db_path': 'data/market_structure.db',
        'log_path': 'logs/market_structure.log',
        'output_dir': 'outputs',
        'csv_filename': '市场结构分析结果.csv',
        'json_filename': '市场结构分析结果.json',
    },
    'retry': {
        'max_attempts': 3,
        'sleep_seconds': 1.5,
    },
    'logging': {
        'level': 'INFO',
    },
    'output': {
        'write_csv': True,
        'write_json': True,
        'write_charts': True,
        'rps_chart_filename': '市场结构RPS柱状图.png',
        'quadrant_chart_filename': '市场结构四象限图.png',
        'timeseries_chart_filename': '市场结构时间序列图.png',
    },
    'crowded_model': {
        'amount_ratio_weight': 0.6,
        'turnover_rate_weight': 0.4,
    },
    'state_model': {
        'history_window': 252,
        'min_history': 60,
        'strong_rps_threshold': 80,
        'healthy_strong_quantile': 0.7,
        'healthy_rotation_quantile': 0.4,
        'healthy_crowded_quantile': 0.65,
        'chaos_strong_quantile': 0.3,
        'chaos_rotation_quantile': 0.7,
        'risk_strong_quantile': 0.8,
        'risk_crowded_quantile': 0.8,
    },
    'industry_groups': {
        '农林牧渔': ['农林牧渔'],
        '采掘': ['煤炭', '石油石化'],
        '化工': ['基础化工'],
        '钢铁': ['钢铁'],
        '有色金属': ['有色金属'],
        '电子': ['电子'],
        '家用电器': ['家用电器'],
        '食品饮料': ['食品饮料'],
        '纺织服装': ['纺织服饰'],
        '轻工制造': ['轻工制造'],
        '医药生物': ['医药生物'],
        '公用事业': ['公用事业', '环保'],
        '交通运输': ['交通运输'],
        '房地产': ['房地产'],
        '商业贸易': ['商贸零售'],
        '休闲服务': ['社会服务'],
        '综合': ['综合'],
        '建筑材料': ['建筑材料'],
        '建筑装饰': ['建筑装饰'],
        '电气设备': ['电力设备'],
        '国防军工': ['国防军工'],
        '计算机': ['计算机'],
        '传媒': ['传媒'],
        '通信': ['通信'],
        '银行': ['银行'],
        '非银金融': ['非银金融'],
        '汽车': ['汽车'],
        '机械设备': ['机械设备'],
    },
}


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
    获取项目根目录，统一定位配置、数据库与输出文件。
    """
    script_path = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / 'bin' / 'market_structure.py'
    return script_path.parent.parent


def resolve_project_path(path_text):
    """
    将配置中的相对路径转换为项目根目录下的绝对路径。
    """
    path = Path(path_text)
    if path.is_absolute():
        return path
    return get_project_root() / path


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


def load_parameter_config(config_path=None):
    """
    加载参数文件，不存在时回退到默认配置。
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'market_structure_params.toml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

    config = DEFAULT_CONFIG
    if config_path.exists():
        with config_path.open('rb') as file:
            loaded = tomllib.load(file)
        config = merge_config(DEFAULT_CONFIG, loaded)

    return config, config_path


def parse_args():
    """
    解析命令行参数，支持日期、配置文件与强制补历史。
    """
    parser = argparse.ArgumentParser(description='市场结构分析脚本')
    parser.add_argument('--date', help='分析日期，格式 YYYYMMDD')
    parser.add_argument('--config', help='参数文件路径')
    parser.add_argument('--refresh-history', action='store_true', help='忽略现有缓存，重新检查并补齐历史数据')
    parser.add_argument('--skip-charts', action='store_true', help='预留参数，当前版本未输出图表')
    return parser.parse_args()


def setup_logger(log_path, level='INFO'):
    """
    初始化日志系统，同时输出到控制台和日志文件。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('market_structure')
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    return logger


def call_tushare_with_retry(func, logger, action_name, max_attempts, sleep_seconds, **kwargs):
    """
    对 Tushare 接口调用增加重试，降低网络抖动和限流影响。
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(**kwargs)
        except Exception as error:
            last_error = error
            logger.warning(f'{action_name} 失败，第 {attempt}/{max_attempts} 次重试: {error}')
            if attempt < max_attempts:
                time.sleep(sleep_seconds)
    raise RuntimeError(f'{action_name} 最终失败: {last_error}') from last_error


def init_tushare_client():
    """
    初始化 Tushare Pro 客户端。
    """
    token = os.getenv('TUSHARE_API_KEY') or load_env_value('TUSHARE_API_KEY')
    ts.set_token(token)
    return ts.pro_api(token)


def get_db_connection(db_path):
    """
    获取 SQLite 连接，并打开行工厂便于后续扩展。
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_database(db_path):
    """
    初始化 SQLite 表结构，并补充必要索引。
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS industry_daily (
                trade_date TEXT NOT NULL,
                industry TEXT NOT NULL,
                close REAL,
                amount REAL,
                amount_ratio REAL,
                float_mv REAL,
                turnover_rate REAL,
                source_names TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (trade_date, industry)
            )
            '''
        )
        existing_columns = {
            row['name']
            for row in connection.execute("PRAGMA table_info('industry_daily')").fetchall()
        }
        if 'float_mv' not in existing_columns:
            cursor.execute('ALTER TABLE industry_daily ADD COLUMN float_mv REAL')
        if 'turnover_rate' not in existing_columns:
            cursor.execute('ALTER TABLE industry_daily ADD COLUMN turnover_rate REAL')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_industry_daily_date ON industry_daily (trade_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_industry_daily_industry ON industry_daily (industry)')
        connection.commit()


def get_trade_calendar(pro, start_date, end_date, config, logger):
    """
    获取指定区间内的交易日列表。
    """
    retry_config = config['retry']
    exchange = config['data']['trade_calendar_exchange']
    calendar_df = call_tushare_with_retry(
        pro.trade_cal,
        logger,
        f'获取交易日历 {start_date}-{end_date}',
        retry_config['max_attempts'],
        retry_config['sleep_seconds'],
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        fields='cal_date,is_open',
    )
    if calendar_df.empty:
        return []

    return (
        calendar_df[calendar_df['is_open'] == 1]['cal_date']
        .sort_values()
        .astype(str)
        .tolist()
    )


def get_last_trading_day(pro, date_str, config, logger):
    """
    将输入日期校准到最近交易日，避免周末和节假日导致无数据。
    """
    try:
        input_date = datetime.strptime(date_str, '%Y%m%d')
    except ValueError as error:
        raise ValueError(f'日期格式错误: {date_str}，请使用 YYYYMMDD') from error

    start_date = (input_date - timedelta(days=10)).strftime('%Y%m%d')
    end_date = input_date.strftime('%Y%m%d')
    trade_dates = get_trade_calendar(pro, start_date, end_date, config, logger)
    if not trade_dates:
        raise RuntimeError(f'在 {date_str} 前后未找到可用交易日')
    return trade_dates[-1]


def build_source_target_mapping(industry_groups):
    """
    将配置中的行业聚合映射整理成“新行业名 -> 旧口径行业名”字典。
    """
    mapping = {}
    for target_name, source_names in industry_groups.items():
        for source_name in source_names:
            mapping[source_name] = target_name
    return mapping


def chunk_dates(trade_dates, chunk_size):
    """
    按交易日数量切分日期列表，规避单次接口返回行数上限。
    """
    for offset in range(0, len(trade_dates), chunk_size):
        yield trade_dates[offset: offset + chunk_size]


def get_stored_trade_dates(db_path):
    """
    查询数据库中已经缓存的交易日，供增量更新判断使用。
    """
    with get_db_connection(db_path) as connection:
        rows = connection.execute(
            'SELECT DISTINCT trade_date FROM industry_daily ORDER BY trade_date'
        ).fetchall()
    return [row['trade_date'] for row in rows]


def get_trade_date_counts(db_path):
    """
    统计每个交易日在数据库中的完整度，用于识别需要补刷的日期。
    """
    with get_db_connection(db_path) as connection:
        rows = connection.execute(
            '''
            SELECT
                trade_date,
                COUNT(*) AS row_count,
                SUM(CASE WHEN float_mv IS NOT NULL THEN 1 ELSE 0 END) AS float_mv_count,
                SUM(CASE WHEN turnover_rate IS NOT NULL THEN 1 ELSE 0 END) AS turnover_rate_count
            FROM industry_daily
            GROUP BY trade_date
            '''
        ).fetchall()
    return {
        row['trade_date']: {
            'row_count': row['row_count'],
            'float_mv_count': row['float_mv_count'],
            'turnover_rate_count': row['turnover_rate_count'],
        }
        for row in rows
    }


def get_sw_industry_master(pro, config, logger):
    """
    获取申万一级行业代码表，为按行业代码抓取历史时序做准备。
    """
    retry_config = config['retry']
    industry_df = call_tushare_with_retry(
        pro.index_classify,
        logger,
        '获取申万一级行业列表',
        retry_config['max_attempts'],
        retry_config['sleep_seconds'],
        level='L1',
        src='SW2021',
    )
    if industry_df.empty:
        raise RuntimeError('未获取到申万一级行业列表')
    return industry_df[['index_code', 'industry_name']].drop_duplicates()


def aggregate_sw_daily(raw_df, source_to_target):
    """
    将申万行业日线按旧口径聚合，兼顾 2021 版行业拆分情况。
    """
    if raw_df.empty:
        return pd.DataFrame()

    working = raw_df.copy()
    working['industry'] = working['name'].map(source_to_target)
    working = working.dropna(subset=['industry'])
    if working.empty:
        return pd.DataFrame()

    def aggregate_group(group):
        total_amount = group['amount'].sum()
        total_float_mv = group['float_mv'].sum()
        if total_amount > 0:
            close_value = np.average(group['close'], weights=group['amount'])
        else:
            close_value = group['close'].mean()
        return pd.Series(
            {
                'close': close_value,
                'amount': total_amount,
                'float_mv': total_float_mv,
                'turnover_rate': (total_amount / total_float_mv * 100) if total_float_mv and total_float_mv > 0 else np.nan,
                'source_names': ','.join(sorted(group['name'].astype(str).unique())),
            }
        )

    aggregated = (
        working.groupby(['trade_date', 'industry'], as_index=False)
        .apply(aggregate_group)
        .reset_index(drop=True)
    )

    total_amount_by_date = aggregated.groupby('trade_date')['amount'].transform('sum')
    aggregated['amount_ratio'] = np.where(
        total_amount_by_date > 0,
        aggregated['amount'] / total_amount_by_date * 100,
        0.0,
    )
    return aggregated[['trade_date', 'industry', 'close', 'amount', 'amount_ratio', 'float_mv', 'turnover_rate', 'source_names']]


def save_industry_data(db_path, df):
    """
    批量写入行业日线数据，已有主键记录则自动更新。
    """
    if df.empty:
        return 0

    save_df = df.copy()
    save_df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = [
        (
            row.trade_date,
            row.industry,
            None if pd.isna(row.close) else float(row.close),
            None if pd.isna(row.amount) else float(row.amount),
            None if pd.isna(row.amount_ratio) else float(row.amount_ratio),
            None if pd.isna(row.float_mv) else float(row.float_mv),
            None if pd.isna(row.turnover_rate) else float(row.turnover_rate),
            row.source_names,
            row.updated_at,
        )
        for row in save_df.itertuples(index=False)
    ]

    with get_db_connection(db_path) as connection:
        connection.executemany(
            '''
            INSERT OR REPLACE INTO industry_daily
            (trade_date, industry, close, amount, amount_ratio, float_mv, turnover_rate, source_names, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            payload,
        )
        connection.commit()
    return len(payload)


def update_industry_history(pro, config, db_path, end_date, logger, force_refresh=False):
    """
    增量更新行业历史数据，只抓缺失交易日，避免重复请求。
    """
    start_date = config['data']['start_date']
    trade_dates = get_trade_calendar(pro, start_date, end_date, config, logger)
    if not trade_dates:
        raise RuntimeError('未获取到任何交易日，无法更新行业历史数据')

    expected_count = len(config['industry_groups'])
    trade_date_counts = get_trade_date_counts(db_path)
    if force_refresh:
        missing_dates = trade_dates
    else:
        missing_dates = [
            date for date in trade_dates
            if (
                trade_date_counts.get(date, {}).get('row_count', 0) < expected_count or
                trade_date_counts.get(date, {}).get('float_mv_count', 0) < expected_count or
                trade_date_counts.get(date, {}).get('turnover_rate_count', 0) < expected_count
            )
        ]

    if not missing_dates:
        logger.info('数据库历史已是最新，无需补充行业日线。')
        return 0, len(trade_dates)

    logger.info(f'待补交易日数量: {len(missing_dates)} / {len(trade_dates)}')
    retry_config = config['retry']
    source_to_target = build_source_target_mapping(config['industry_groups'])
    source_names = sorted(source_to_target.keys())
    industry_master = get_sw_industry_master(pro, config, logger)
    industry_code_map = {
        row.industry_name: row.index_code
        for row in industry_master.itertuples(index=False)
    }
    total_saved_rows = 0

    missing_start = missing_dates[0]
    missing_end = missing_dates[-1]
    logger.info(f'按行业代码抓取申万历史: {missing_start} ~ {missing_end}')

    all_frames = []
    for source_name in source_names:
        industry_code = industry_code_map.get(source_name)
        if not industry_code:
            logger.warning(f'未找到行业代码，跳过: {source_name}')
            continue

        logger.info(f'抓取 {source_name}({industry_code})')
        source_df = call_tushare_with_retry(
            pro.sw_daily,
            logger,
            f'获取 {source_name} 行业日线',
            retry_config['max_attempts'],
            retry_config['sleep_seconds'],
            ts_code=industry_code,
            start_date=missing_start,
            end_date=missing_end,
            fields='ts_code,trade_date,name,close,amount,float_mv',
        )
        if source_df.empty:
            logger.warning(f'{source_name} 行业日线为空')
            continue
        all_frames.append(source_df)

    if not all_frames:
        raise RuntimeError('行业日线抓取全部为空，无法更新数据库')

    raw_df = pd.concat(all_frames, ignore_index=True)
    aggregated = aggregate_sw_daily(raw_df, source_to_target)
    if aggregated.empty:
        raise RuntimeError('行业映射后无有效数据，请检查行业名称映射')

    aggregated = aggregated[aggregated['trade_date'].isin(missing_dates)].copy()
    saved_rows = save_industry_data(db_path, aggregated)
    total_saved_rows += saved_rows
    logger.info(f'已写入 {saved_rows} 行行业数据')

    return total_saved_rows, len(trade_dates)


def load_history_dataframe(db_path, end_date):
    """
    从数据库读取截止分析日的全部行业历史，供 RPS 与拥挤度计算复用。
    """
    with get_db_connection(db_path) as connection:
        history_df = pd.read_sql_query(
            '''
            SELECT trade_date, industry, close, amount, amount_ratio, float_mv, turnover_rate, source_names
            FROM industry_daily
            WHERE trade_date <= ?
            ORDER BY trade_date, industry
            ''',
            connection,
            params=(end_date,),
        )
    return history_df


def compute_percentile(history_values, current_value):
    """
    计算当前值在历史样本中的分位位置。
    """
    values = [value for value in history_values if pd.notna(value)]
    values.append(current_value)
    if not values:
        return np.nan

    values_array = np.asarray(values, dtype=float)
    return float((values_array <= current_value).mean() * 100)


def compute_group_trailing_percentiles(df, group_col, value_col, window):
    """
    计算组内时间序列的滚动历史分位，用于拥挤度与状态分位分析。
    """
    if df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(np.nan, index=df.index, dtype=float)
    grouped = df.groupby(group_col, sort=False).groups
    for _, row_indices in grouped.items():
        indices = list(row_indices)
        values = df.loc[indices, value_col].tolist()
        for position, row_index in enumerate(indices):
            current_value = values[position]
            if pd.isna(current_value):
                continue
            start_index = max(0, position - window)
            history_values = [value for value in values[start_index:position] if pd.notna(value)]
            result.at[row_index] = compute_percentile(history_values, current_value) if history_values else 50.0
    return result


def combine_crowded_percentiles(df, config):
    """
    按配置权重合成成交额占比与换手率两类拥挤度分位。
    """
    amount_weight = config['crowded_model']['amount_ratio_weight']
    turnover_weight = config['crowded_model']['turnover_rate_weight']
    total_weight = (
        df['amount_crowded_percentile'].notna().astype(float) * amount_weight +
        df['turnover_crowded_percentile'].notna().astype(float) * turnover_weight
    )
    combined_value = (
        df['amount_crowded_percentile'].fillna(0) * amount_weight +
        df['turnover_crowded_percentile'].fillna(0) * turnover_weight
    )
    return np.where(total_weight > 0, combined_value / total_weight, np.nan)


def build_feature_history(history_df, config):
    """
    基于数据库缓存的行业时序，为全部历史日期构建 RPS 与双因子拥挤度特征。
    """
    if history_df.empty:
        return pd.DataFrame()

    working = history_df.copy()
    working['trade_date'] = working['trade_date'].astype(str)
    working = working.sort_values(['industry', 'trade_date'])

    lookback_days = config['data']['lookback_days']
    crowded_window = config['data']['crowded_window']

    working['ret_20d'] = (
        working.groupby('industry')['close']
        .pct_change(periods=lookback_days - 1)
    )
    working['ret_20d_clean'] = working.groupby('trade_date')['ret_20d'].transform(
        lambda series: series.fillna(series.dropna().min() if not series.dropna().empty else 0.0)
    )
    working['RPS'] = working.groupby('trade_date')['ret_20d_clean'].rank(pct=True) * 100
    working['amount_crowded_percentile'] = compute_group_trailing_percentiles(
        working,
        'industry',
        'amount_ratio',
        crowded_window,
    )
    working['turnover_crowded_percentile'] = compute_group_trailing_percentiles(
        working,
        'industry',
        'turnover_rate',
        crowded_window,
    )
    working['crowded_percentile'] = combine_crowded_percentiles(working, config)
    return working


def calculate_industry_metrics(feature_history_df, trade_date):
    """
    提取分析日的行业得分明细，作为最终展示与导出结果。
    """
    if feature_history_df.empty:
        return pd.DataFrame()

    latest_df = feature_history_df[feature_history_df['trade_date'] == trade_date].copy()
    if latest_df.empty:
        return pd.DataFrame()

    return latest_df.sort_values('RPS', ascending=False).reset_index(drop=True)


def compute_series_trailing_percentiles(series, window):
    """
    计算单变量时间序列相对于过去窗口的历史分位。
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.tolist()
    for idx, current_value in enumerate(values):
        if pd.isna(current_value):
            continue
        start_index = max(0, idx - window)
        history_values = [value for value in values[start_index:idx] if pd.notna(value)]
        if history_values:
            result.iat[idx] = compute_percentile(history_values, current_value)
    return result


def build_daily_state_metrics(feature_history_df, config):
    """
    将行业层特征聚合成每日市场结构状态指标，并计算动态阈值。
    """
    if feature_history_df.empty:
        return pd.DataFrame()

    state_config = config['state_model']
    strong_rps_threshold = state_config['strong_rps_threshold']
    history_window = state_config['history_window']
    min_history = state_config['min_history']

    rows = []
    for trade_date, group in feature_history_df.groupby('trade_date', sort=True):
        ret_series = group['ret_20d_clean']
        rows.append(
            {
                'trade_date': trade_date,
                'strong_ratio': float((group['RPS'] > strong_rps_threshold).mean() * 100),
                'rotation_score': float(min(100, ret_series.std() / (ret_series.abs().mean() + 1e-6) * 40)),
                'avg_crowded': float(group['crowded_percentile'].mean()),
                'max_crowded': float(group['crowded_percentile'].max()),
            }
        )

    daily_df = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True)
    for metric_name in ['strong_ratio', 'rotation_score', 'avg_crowded', 'max_crowded']:
        daily_df[f'{metric_name}_hist_pct'] = compute_series_trailing_percentiles(
            daily_df[metric_name],
            history_window,
        )

    shifted = daily_df.shift(1)
    daily_df['healthy_strong_threshold'] = shifted['strong_ratio'].rolling(history_window, min_periods=min_history).quantile(
        state_config['healthy_strong_quantile']
    )
    daily_df['healthy_rotation_threshold'] = shifted['rotation_score'].rolling(history_window, min_periods=min_history).quantile(
        state_config['healthy_rotation_quantile']
    )
    daily_df['healthy_crowded_threshold'] = shifted['max_crowded'].rolling(history_window, min_periods=min_history).quantile(
        state_config['healthy_crowded_quantile']
    )
    daily_df['chaos_strong_threshold'] = shifted['strong_ratio'].rolling(history_window, min_periods=min_history).quantile(
        state_config['chaos_strong_quantile']
    )
    daily_df['chaos_rotation_threshold'] = shifted['rotation_score'].rolling(history_window, min_periods=min_history).quantile(
        state_config['chaos_rotation_quantile']
    )
    daily_df['risk_strong_threshold'] = shifted['strong_ratio'].rolling(history_window, min_periods=min_history).quantile(
        state_config['risk_strong_quantile']
    )
    daily_df['risk_crowded_threshold'] = shifted['max_crowded'].rolling(history_window, min_periods=min_history).quantile(
        state_config['risk_crowded_quantile']
    )
    return daily_df


def evaluate_market_state(result_df, daily_state_df, trade_date, config):
    """
    基于滚动历史分位阈值，对分析日的市场结构状态做动态判断。
    """
    if result_df.empty or daily_state_df.empty:
        return {'市场状态': '数据不足'}

    state_row = daily_state_df[daily_state_df['trade_date'] == trade_date]
    if state_row.empty:
        return {'市场状态': '数据不足'}

    state_row = state_row.iloc[0]
    strong_ratio_hist_pct = float(state_row['strong_ratio_hist_pct']) if pd.notna(state_row['strong_ratio_hist_pct']) else np.nan
    rotation_score_hist_pct = float(state_row['rotation_score_hist_pct']) if pd.notna(state_row['rotation_score_hist_pct']) else np.nan
    avg_crowded_hist_pct = float(state_row['avg_crowded_hist_pct']) if pd.notna(state_row['avg_crowded_hist_pct']) else np.nan
    max_crowded_hist_pct = float(state_row['max_crowded_hist_pct']) if pd.notna(state_row['max_crowded_hist_pct']) else np.nan
    strong_ratio = float(state_row['strong_ratio'])
    rotation_score = float(state_row['rotation_score'])
    avg_crowded = float(state_row['avg_crowded'])
    max_crowded = float(state_row['max_crowded'])
    top_industry = result_df.loc[result_df['RPS'].idxmax(), 'industry']
    strong_healthy_gate = config['state_model']['healthy_strong_quantile'] * 100
    rotation_healthy_gate = config['state_model']['healthy_rotation_quantile'] * 100
    crowded_healthy_gate = config['state_model']['healthy_crowded_quantile'] * 100
    strong_chaos_gate = config['state_model']['chaos_strong_quantile'] * 100
    rotation_chaos_gate = config['state_model']['chaos_rotation_quantile'] * 100
    strong_risk_gate = config['state_model']['risk_strong_quantile'] * 100
    crowded_risk_gate = config['state_model']['risk_crowded_quantile'] * 100

    if (
        pd.notna(strong_ratio_hist_pct) and
        pd.notna(max_crowded_hist_pct) and
        strong_ratio_hist_pct >= strong_risk_gate and
        max_crowded_hist_pct >= crowded_risk_gate
    ):
        state = '主线过热（风险）'
    elif (
        pd.notna(strong_ratio_hist_pct) and
        pd.notna(rotation_score_hist_pct) and
        (
            strong_ratio_hist_pct <= strong_chaos_gate or
            rotation_score_hist_pct >= rotation_chaos_gate
        )
    ):
        state = '快速轮动（混沌）'
    elif (
        pd.notna(strong_ratio_hist_pct) and
        pd.notna(rotation_score_hist_pct) and
        pd.notna(max_crowded_hist_pct) and
        strong_ratio_hist_pct >= strong_healthy_gate and
        rotation_score_hist_pct <= rotation_healthy_gate and
        max_crowded_hist_pct <= crowded_healthy_gate
    ):
        state = '主线清晰（健康）'
    else:
        state = '震荡分化（中性）'

    return {
        '市场状态': state,
        '强势行业比例(%)': round(strong_ratio, 2),
        '轮动频率评分': round(rotation_score, 2),
        '平均拥挤度分位(%)': round(avg_crowded, 2),
        '最高拥挤度分位(%)': round(max_crowded, 2),
        '强势行业比例历史分位(%)': round(strong_ratio_hist_pct, 2),
        '轮动频率历史分位(%)': round(rotation_score_hist_pct, 2),
        '平均拥挤度历史分位(%)': round(avg_crowded_hist_pct, 2),
        '最高拥挤度历史分位(%)': round(max_crowded_hist_pct, 2),
        '健康强势阈值(%)': round(float(state_row['healthy_strong_threshold']) if pd.notna(state_row['healthy_strong_threshold']) else np.nan, 2),
        '健康轮动阈值(%)': round(float(state_row['healthy_rotation_threshold']) if pd.notna(state_row['healthy_rotation_threshold']) else np.nan, 2),
        '健康拥挤阈值(%)': round(float(state_row['healthy_crowded_threshold']) if pd.notna(state_row['healthy_crowded_threshold']) else np.nan, 2),
        '混沌强势阈值(%)': round(float(state_row['chaos_strong_threshold']) if pd.notna(state_row['chaos_strong_threshold']) else np.nan, 2),
        '混沌轮动阈值(%)': round(float(state_row['chaos_rotation_threshold']) if pd.notna(state_row['chaos_rotation_threshold']) else np.nan, 2),
        '过热强势阈值(%)': round(float(state_row['risk_strong_threshold']) if pd.notna(state_row['risk_strong_threshold']) else np.nan, 2),
        '过热拥挤阈值(%)': round(float(state_row['risk_crowded_threshold']) if pd.notna(state_row['risk_crowded_threshold']) else np.nan, 2),
        '最强行业': top_industry,
    }


def display_top_bottom(result_df, n=5):
    """
    输出最强和最弱行业榜单，便于快速识别主线与拖累板块。
    """
    top_df = result_df.nlargest(
        n,
        'RPS',
    )[['industry', 'RPS', 'crowded_percentile', 'amount_crowded_percentile', 'turnover_crowded_percentile', 'ret_20d']]
    bottom_df = result_df.nsmallest(
        n,
        'RPS',
    )[['industry', 'RPS', 'crowded_percentile', 'amount_crowded_percentile', 'turnover_crowded_percentile', 'ret_20d']]
    return top_df, bottom_df


def get_database_stats(db_path):
    """
    获取数据库累计交易日数和行业数，帮助判断历史样本丰富程度。
    """
    with get_db_connection(db_path) as connection:
        total_days = connection.execute('SELECT COUNT(DISTINCT trade_date) FROM industry_daily').fetchone()[0]
        total_industries = connection.execute('SELECT COUNT(DISTINCT industry) FROM industry_daily').fetchone()[0]
    return total_days, total_industries


def write_output_files(result_df, state_info, top_df, bottom_df, daily_state_df, config, trade_date, config_path, db_path, logger):
    """
    输出 CSV 和 JSON 结果，便于后续复核或前端调用。
    """
    output_dir = resolve_project_path(config['storage']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / config['storage']['csv_filename']
    json_path = output_dir / config['storage']['json_filename']

    if config['output']['write_csv']:
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f'已输出 CSV 结果: {csv_path}')

    if config['output']['write_json']:
        payload = {
            'analysis_date': trade_date,
            'config_path': str(config_path),
            'database_path': str(db_path),
            'state_info': state_info,
            'state_metrics_latest': daily_state_df[daily_state_df['trade_date'] == trade_date].to_dict(orient='records'),
            'top_industries': top_df.to_dict(orient='records'),
            'bottom_industries': bottom_df.to_dict(orient='records'),
            'industry_results': result_df.to_dict(orient='records'),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float),
            encoding='utf-8',
        )
        logger.info(f'已输出 JSON 结果: {json_path}')


def print_analysis_report(result_df, state_info, top_df, bottom_df, total_days, total_industries, trade_date, logger):
    """
    将核心分析结论打印到控制台，并同步写入日志。
    """
    logger.info('')
    logger.info(f'输入日期已校准到最近交易日: {trade_date}')
    logger.info('========== 各行业核心数据 ==========')
    logger.info(
        result_df[
            [
                'industry',
                'close',
                'amount_ratio',
                'turnover_rate',
                'RPS',
                'amount_crowded_percentile',
                'turnover_crowded_percentile',
                'crowded_percentile',
                'ret_20d',
            ]
        ]
        .to_string(index=False, float_format=lambda value: f'{value:.2f}')
    )

    logger.info('')
    logger.info('========== 市场状态分析 ==========')
    for key, value in state_info.items():
        logger.info(f'{key}: {value}')

    logger.info('')
    logger.info('========== 最强行业 TOP5 ==========')
    logger.info(top_df.to_string(index=False, float_format=lambda value: f'{value:.2f}'))

    logger.info('')
    logger.info('========== 最弱行业 BOTTOM5 ==========')
    logger.info(bottom_df.to_string(index=False, float_format=lambda value: f'{value:.2f}'))

    logger.info('')
    logger.info('========== 数据库状态 ==========')
    logger.info(f'已存储交易日数: {total_days}')
    logger.info(f'已存储行业数: {total_industries}')
    logger.info('建议：拥挤度分位数的可靠性会随历史样本积累而提高，建议每日收盘后运行一次。')


def add_chart_labels(ax, bars, values, text_format='{:.1f}', rotation=0):
    """
    为柱状图补充数值标签，便于快速读取高低分布。
    """
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            text_format.format(value),
            va='center',
            ha='left',
            fontsize=9,
            rotation=rotation,
        )


def generate_structure_charts(result_df, state_info, config, trade_date, logger):
    """
    生成行业 RPS 柱状图和 RPS-拥挤度四象限图。
    """
    if result_df.empty or not config['output'].get('write_charts', True):
        return []

    output_dir = resolve_project_path(config['storage']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    plot_df = result_df.sort_values('RPS', ascending=True).copy()
    colors = np.where(plot_df['RPS'] >= 80, '#d62728', '#4c78a8')
    rps_chart_path = output_dir / config['output']['rps_chart_filename']

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(plot_df['industry'], plot_df['RPS'], color=colors, alpha=0.9)
    ax.axvline(80, color='red', linestyle='--', linewidth=1, label='强势阈值 RPS=80')
    ax.axvline(50, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_title(f'市场结构行业 RPS 排名 ({trade_date})', fontsize=15)
    ax.set_xlabel('RPS')
    ax.set_ylabel('行业')
    ax.set_xlim(0, max(100, float(plot_df['RPS'].max()) + 5))
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    add_chart_labels(ax, bars, plot_df['RPS'].tolist())
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(rps_chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    generated_paths.append(rps_chart_path)

    quadrant_chart_path = output_dir / config['output']['quadrant_chart_filename']
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        result_df['crowded_percentile'],
        result_df['RPS'],
        s=np.clip(result_df['amount_ratio'] * 120, 60, 1200),
        c=result_df['turnover_rate'],
        cmap='RdYlBu_r',
        edgecolors='black',
        linewidths=0.6,
        alpha=0.85,
    )
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.axvline(50, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('综合拥挤度分位')
    ax.set_ylabel('RPS')
    ax.set_title(f'市场结构四象限图 ({trade_date})\n{state_info.get("市场状态", "未知状态")}', fontsize=15)
    ax.grid(alpha=0.18, linestyle='--')

    for row in result_df.itertuples(index=False):
        ax.annotate(
            row.industry,
            (row.crowded_percentile, row.RPS),
            textcoords='offset points',
            xytext=(5, 4),
            fontsize=9,
        )

    color_bar = plt.colorbar(scatter, ax=ax)
    color_bar.set_label('行业换手率(%)')

    ax.text(18, 92, '左上：强势但不拥挤', fontsize=10, color='#1f4e79')
    ax.text(62, 92, '右上：强势且拥挤', fontsize=10, color='#a61c00')
    ax.text(18, 8, '左下：弱势且不拥挤', fontsize=10, color='#38761d')
    ax.text(62, 8, '右下：弱势但拥挤', fontsize=10, color='#7f6000')

    plt.tight_layout()
    plt.savefig(quadrant_chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    generated_paths.append(quadrant_chart_path)

    for chart_path in generated_paths:
        logger.info(f'已输出图表: {chart_path}')
    return generated_paths


def generate_structure_timeseries_chart(daily_state_df, trade_date, config, logger):
    """
    绘制市场结构时间序列图，展示主线强度、轮动频率与拥挤度的历史演变。
    """
    if daily_state_df.empty or not config['output'].get('write_charts', True):
        return None

    plot_df = daily_state_df.copy()
    plot_df['trade_date'] = pd.to_datetime(plot_df['trade_date'])
    output_dir = resolve_project_path(config['storage']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / config['output']['timeseries_chart_filename']

    latest_rows = plot_df[plot_df['trade_date'] == pd.to_datetime(trade_date)]
    latest_row = latest_rows.iloc[-1] if not latest_rows.empty else plot_df.iloc[-1]
    latest_date = latest_row['trade_date']

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(plot_df['trade_date'], plot_df['strong_ratio'], color='#d62728', linewidth=1.6, label='强势行业比例')
    axes[0].plot(
        plot_df['trade_date'],
        plot_df['healthy_strong_threshold'],
        color='#ff9896',
        linestyle='--',
        linewidth=1.0,
        label='健康阈值',
    )
    axes[0].plot(
        plot_df['trade_date'],
        plot_df['risk_strong_threshold'],
        color='#7f0000',
        linestyle=':',
        linewidth=1.0,
        label='过热阈值',
    )
    axes[0].scatter(latest_date, latest_row['strong_ratio'], color='#d62728', s=60, zorder=5)
    axes[0].set_ylabel('强势比例(%)')
    axes[0].set_title('市场结构时间序列图', fontsize=16)
    axes[0].grid(alpha=0.2, linestyle='--')
    axes[0].legend(loc='upper left')

    axes[1].plot(plot_df['trade_date'], plot_df['rotation_score'], color='#1f77b4', linewidth=1.6, label='轮动频率评分')
    axes[1].plot(
        plot_df['trade_date'],
        plot_df['healthy_rotation_threshold'],
        color='#9ecae1',
        linestyle='--',
        linewidth=1.0,
        label='健康阈值',
    )
    axes[1].plot(
        plot_df['trade_date'],
        plot_df['chaos_rotation_threshold'],
        color='#08519c',
        linestyle=':',
        linewidth=1.0,
        label='混沌阈值',
    )
    axes[1].scatter(latest_date, latest_row['rotation_score'], color='#1f77b4', s=60, zorder=5)
    axes[1].set_ylabel('轮动评分')
    axes[1].grid(alpha=0.2, linestyle='--')
    axes[1].legend(loc='upper left')

    axes[2].plot(plot_df['trade_date'], plot_df['avg_crowded'], color='#2ca02c', linewidth=1.5, label='平均拥挤度')
    axes[2].plot(plot_df['trade_date'], plot_df['max_crowded'], color='#9467bd', linewidth=1.3, label='最高拥挤度')
    axes[2].plot(
        plot_df['trade_date'],
        plot_df['healthy_crowded_threshold'],
        color='#98df8a',
        linestyle='--',
        linewidth=1.0,
        label='健康阈值',
    )
    axes[2].plot(
        plot_df['trade_date'],
        plot_df['risk_crowded_threshold'],
        color='#6a3d9a',
        linestyle=':',
        linewidth=1.0,
        label='过热阈值',
    )
    axes[2].scatter(latest_date, latest_row['avg_crowded'], color='#2ca02c', s=50, zorder=5)
    axes[2].scatter(latest_date, latest_row['max_crowded'], color='#9467bd', s=50, zorder=5)
    axes[2].set_ylabel('拥挤度分位')
    axes[2].set_xlabel('时间')
    axes[2].grid(alpha=0.2, linestyle='--')
    axes[2].legend(loc='upper left', ncol=2)

    for ax in axes:
        ax.axvline(latest_date, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.tick_params(axis='x', rotation=0)

    axes[0].text(
        0.01,
        0.92,
        f'最新状态: {latest_row.get("trade_date").strftime("%Y-%m-%d")} / 强势 {latest_row["strong_ratio"]:.2f}% / 轮动 {latest_row["rotation_score"]:.2f} / 最高拥挤 {latest_row["max_crowded"]:.2f}',
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'},
    )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'已输出图表: {chart_path}')
    return chart_path


def main():
    """
    主流程：加载配置、更新历史、计算分析结果并输出结构化文件。
    """
    configure_console_encoding()
    args = parse_args()
    config, config_path = load_parameter_config(args.config)

    if args.date:
        config['data']['analyse_date'] = args.date

    db_path = resolve_project_path(config['storage']['db_path'])
    log_path = resolve_project_path(config['storage']['log_path'])
    logger = setup_logger(log_path, config['logging']['level'])
    logger.info(f'已加载参数文件: {config_path}')

    if args.skip_charts:
        logger.info('当前版本未输出图表，已忽略 --skip-charts 参数。')

    init_database(db_path)
    pro = init_tushare_client()
    logger.info('Tushare 初始化成功')

    trade_date = get_last_trading_day(pro, config['data']['analyse_date'], config, logger)
    logger.info(f'目标分析日期: {config["data"]["analyse_date"]}')
    logger.info(f'实际分析日期: {trade_date}')

    saved_rows, total_trade_days = update_industry_history(
        pro,
        config,
        db_path,
        trade_date,
        logger,
        force_refresh=args.refresh_history,
    )
    logger.info(f'本次补充写入行数: {saved_rows}')
    logger.info(f'纳入交易日总数: {total_trade_days}')

    history_df = load_history_dataframe(db_path, trade_date)
    feature_history_df = build_feature_history(history_df, config)
    result_df = calculate_industry_metrics(feature_history_df, trade_date)
    if result_df.empty:
        raise RuntimeError('分析日无有效行业数据，请检查 Tushare 权限、配置映射或历史更新是否成功。')

    daily_state_df = build_daily_state_metrics(feature_history_df, config)
    state_info = evaluate_market_state(result_df, daily_state_df, trade_date, config)
    top_df, bottom_df = display_top_bottom(result_df)
    total_days, total_industries = get_database_stats(db_path)

    print_analysis_report(result_df, state_info, top_df, bottom_df, total_days, total_industries, trade_date, logger)
    generate_structure_charts(result_df, state_info, config, trade_date, logger)
    generate_structure_timeseries_chart(daily_state_df, trade_date, config, logger)
    write_output_files(result_df, state_info, top_df, bottom_df, daily_state_df, config, trade_date, config_path, db_path, logger)


if __name__ == '__main__':
    main()

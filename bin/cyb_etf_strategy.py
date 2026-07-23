import argparse
import os
import sys
import tomllib
import warnings
from pathlib import Path

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
    配置控制台输出编码，避免 Windows 中文输出异常。
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')


def get_project_root():
    """
    获取项目根目录。
    """
    return Path(__file__).resolve().parent.parent


def build_output_path(filename):
    """
    构造项目输出文件路径。
    """
    output_dir = get_project_root() / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def resolve_path(path_str):
    """
    将相对路径解析为项目根目录下的绝对路径。
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (get_project_root() / path).resolve()


def load_env_value(key, env_path=None):
    """
    从 .env 文件读取配置项。
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


def load_parameter_config(config_path=None):
    """
    加载策略参数文件。
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'cyb_etf_strategy_params.toml'
    else:
        config_path = resolve_path(config_path)

    with config_path.open('rb') as file:
        config = tomllib.load(file)

    print(f'已加载参数文件: {config_path}')
    return config


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='创业板 ETF 温度交易系统回测')
    parser.add_argument('--config', help='策略参数文件路径')
    parser.add_argument('--start', help='分析开始日期，格式 YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--end', help='分析结束日期，格式 YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--input', help='市场温度时间序列 CSV 路径')
    return parser.parse_args()


def normalize_date(value):
    """
    规范日期字符串格式。
    """
    if not value:
        return None
    return pd.to_datetime(value).strftime('%Y%m%d')


def get_tushare_pro():
    """
    初始化 Tushare Pro。
    """
    token = os.getenv('TUSHARE_API_KEY') or load_env_value('TUSHARE_API_KEY')
    return ts.pro_api(token)


def load_temperature_series(input_path, start_date=None, end_date=None, rebound_days=5):
    """
    读取市场温度时间序列。
    """
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)

    if start_date:
        df = df[df['trade_date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['trade_date'] <= pd.to_datetime(end_date)]

    if len(df) == 0:
        raise ValueError('温度时间序列在指定区间内无数据')

    rebound_days = max(int(rebound_days), 1)
    df['composite_rebound'] = df['composite_score'] > df['composite_score'].shift(rebound_days)
    return df.reset_index(drop=True)


def get_etf_daily_data(pro, ts_code, start_date, end_date, ma_short_window=20, ma_long_window=60):
    """
    获取 ETF 日线与复权因子，并构造前复权开收盘价。
    """
    df_price = pro.fund_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    df_adj = pro.fund_adj(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )

    if df_price is None or len(df_price) == 0:
        raise ValueError(f'未获取到 ETF 日线数据: {ts_code}')
    if df_adj is None or len(df_adj) == 0:
        raise ValueError(f'未获取到 ETF 复权因子数据: {ts_code}')

    df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])
    df_adj['trade_date'] = pd.to_datetime(df_adj['trade_date'])

    df = df_price.merge(df_adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
    df = df.sort_values('trade_date').reset_index(drop=True)
    latest_factor = df['adj_factor'].iloc[-1]
    df['adj_open'] = df['open'] * df['adj_factor'] / latest_factor
    df['adj_close'] = df['close'] * df['adj_factor'] / latest_factor
    df['ma_short'] = df['adj_close'].rolling(ma_short_window, min_periods=ma_short_window).mean()
    df['ma_long'] = df['adj_close'].rolling(ma_long_window, min_periods=ma_long_window).mean()
    df['trend_confirmed'] = df['ma_short'] > df['ma_long']
    return df


def prepare_backtest_data(temp_df, etf_df):
    """
    合并温度与 ETF 行情数据。
    """
    merged = temp_df.merge(
        etf_df[['trade_date', 'open', 'close', 'adj_open', 'adj_close', 'ma_short', 'ma_long', 'trend_confirmed']],
        on='trade_date',
        how='inner'
    )
    if len(merged) < 30:
        raise ValueError('可用于回测的合并数据过少')

    merged = merged.sort_values('trade_date').reset_index(drop=True)
    return merged


def initialize_state(initial_capital):
    """
    初始化策略持仓状态。
    """
    return {
        'cash': float(initial_capital),
        'shares': 0,
        'avg_cost': 0.0,
        'holding_days': 0,
        'cooldown_days_left': 0,
        'target_weight_level': 0.0,
        'pending_target_weight': None,
        'pending_reason': None,
        'entry_date': None,
        'entry_nav': None,
        'round_trip_id': 0,
        'trade_count': 0,
        'round_trips': [],
    }


def get_confirmation_flag(row, strategy_config):
    """
    判断综合温度是否对当前价值信号形成确认。
    """
    composite_low = row['composite_score'] <= strategy_config['confirm_composite_low']
    composite_rebound = bool(row['composite_rebound']) if pd.notna(row['composite_rebound']) else False
    return composite_low or composite_rebound


def get_trend_confirmation_flag(row):
    """
    判断是否满足短期均线强于中期均线的趋势过滤条件。
    """
    return bool(row['trend_confirmed']) if pd.notna(row['trend_confirmed']) else False


def is_in_cooldown(state):
    """
    判断策略是否处于止损后的冷却期。
    """
    return int(state.get('cooldown_days_left', 0)) > 0


def advance_weight_towards_target(current_weight, target_weight, weight_levels):
    """
    按预设仓位层级逐步加仓，每次只提升一档。
    """
    valid_levels = sorted({float(level) for level in weight_levels if float(level) > 0})
    for level in valid_levels:
        if current_weight + 1e-9 < level <= target_weight + 1e-9:
            return level
    return current_weight


def decide_value_only_target(row, state, strategy_config, current_weight, current_return):
    """
    纯价值驱动策略：先试探仓，再逐步加到深度仓和趋势确认后的重仓。
    """
    anchor_score = row['anchor_score']
    trend_confirmed = get_trend_confirmation_flag(row)
    current_target = state['target_weight_level']

    if state['shares'] > 0:
        if current_return <= strategy_config['stop_loss']:
            return 0.0, 'stop_loss'
        if state['holding_days'] >= strategy_config['max_holding_days']:
            return 0.0, 'time_exit'
        if anchor_score >= strategy_config['exit_anchor']:
            return 0.0, 'value_repaired'
        target_cap = current_target
        if anchor_score <= strategy_config['entry_anchor']:
            target_cap = max(target_cap, float(strategy_config['initial_weight']))
        if anchor_score <= strategy_config['deep_anchor']:
            target_cap = max(target_cap, float(strategy_config['deep_weight']))
            if trend_confirmed:
                target_cap = max(target_cap, float(strategy_config['heavy_weight']))

        next_target = advance_weight_towards_target(
            current_weight=current_target,
            target_weight=target_cap,
            weight_levels=[
                strategy_config['initial_weight'],
                strategy_config['deep_weight'],
                strategy_config['heavy_weight'],
            ]
        )
        if next_target > current_target + 1e-9:
            if next_target >= float(strategy_config['heavy_weight']) - 1e-9:
                return next_target, 'trend_confirmed_add'
            if next_target >= float(strategy_config['deep_weight']) - 1e-9:
                return next_target, 'deep_value_add'
            return next_target, 'value_entry_probe'
        return current_target, 'hold'

    if is_in_cooldown(state):
        return 0.0, 'cooldown_wait'

    if anchor_score <= strategy_config['entry_anchor']:
        return float(strategy_config['initial_weight']), 'value_entry_probe'

    return 0.0, 'wait'



def decide_value_managed_target(row, state, strategy_config, current_weight, current_return):
    """
    价值驱动 + 综合温度仓位管理策略。
    """
    anchor_score = row['anchor_score']
    trend_confirmed = get_trend_confirmation_flag(row)
    current_target = state['target_weight_level']
    confirmed = get_confirmation_flag(row, strategy_config)

    if state['shares'] > 0:
        if current_return <= strategy_config['stop_loss']:
            return 0.0, 'stop_loss'
        if state['holding_days'] >= strategy_config['max_holding_days']:
            return 0.0, 'time_exit'
        if anchor_score >= strategy_config['exit_anchor']:
            return 0.0, 'value_repaired'
        target_cap = current_target
        if anchor_score <= strategy_config['entry_anchor']:
            base_weight = (
                float(strategy_config['initial_normal_weight'])
                if confirmed else float(strategy_config['initial_weak_weight'])
            )
            target_cap = max(target_cap, base_weight)
        if anchor_score <= strategy_config['deep_anchor']:
            target_cap = max(target_cap, float(strategy_config['deep_weight']))
            if confirmed and trend_confirmed:
                target_cap = max(target_cap, float(strategy_config['heavy_weight']))

        next_target = advance_weight_towards_target(
            current_weight=current_target,
            target_weight=target_cap,
            weight_levels=[
                strategy_config['initial_weak_weight'],
                strategy_config['initial_normal_weight'],
                strategy_config['deep_weight'],
                strategy_config['heavy_weight'],
            ]
        )
        if next_target > current_target + 1e-9:
            if next_target >= float(strategy_config['heavy_weight']) - 1e-9:
                return next_target, 'trend_confirmed_add'
            if next_target >= float(strategy_config['deep_weight']) - 1e-9:
                return next_target, 'deep_value_add'
            if next_target >= float(strategy_config['initial_normal_weight']) - 1e-9:
                return next_target, 'confirmation_add'
            return next_target, 'value_entry_probe'

        return current_target, 'hold'

    if is_in_cooldown(state):
        return 0.0, 'cooldown_wait'

    if anchor_score <= strategy_config['entry_anchor']:
        if confirmed:
            return float(strategy_config['initial_normal_weight']), 'value_entry_confirmed'
        return float(strategy_config['initial_weak_weight']), 'value_entry_probe'

    return 0.0, 'wait'



def execute_rebalance(state, row, target_weight, reason, trade_config, trade_records, strategy_name):
    """
    在次日开盘按目标仓位执行调仓。
    """
    open_price = float(row['adj_open'])
    lot_size = int(trade_config['lot_size'])
    buy_cost_rate = float(trade_config['buy_cost_rate'])
    sell_cost_rate = float(trade_config['sell_cost_rate'])
    slippage_rate = float(trade_config['slippage_rate'])

    if not np.isfinite(open_price) or open_price <= 0:
        return

    gross_asset = state['cash'] + state['shares'] * open_price
    target_shares = int((gross_asset * target_weight) / open_price / lot_size) * lot_size
    current_shares = int(state['shares'])

    if target_shares == current_shares:
        state['target_weight_level'] = target_weight if state['shares'] > 0 else 0.0
        return

    if target_shares > current_shares:
        shares_to_buy = target_shares - current_shares
        trade_price = open_price * (1 + slippage_rate)
        max_affordable = int(state['cash'] / (trade_price * (1 + buy_cost_rate)) / lot_size) * lot_size
        shares_to_buy = min(shares_to_buy, max_affordable)
        if shares_to_buy <= 0:
            return

        gross_amount = shares_to_buy * trade_price
        cost = gross_amount * buy_cost_rate
        total_outflow = gross_amount + cost
        previous_shares = state['shares']
        state['cash'] -= total_outflow
        state['shares'] += shares_to_buy
        state['avg_cost'] = (
            (state['avg_cost'] * previous_shares + total_outflow) / state['shares']
            if state['shares'] > 0 else 0.0
        )

        if previous_shares == 0 and state['shares'] > 0:
            state['entry_date'] = row['trade_date']
            state['entry_nav'] = gross_asset
            state['round_trip_id'] += 1

        state['trade_count'] += 1
        state['target_weight_level'] = target_weight
        trade_records.append({
            'strategy_name': strategy_name,
            'trade_date': row['trade_date'],
            'action': 'buy',
            'reason': reason,
            'price': trade_price,
            'shares': shares_to_buy,
            'cash_after': state['cash'],
            'shares_after': state['shares'],
            'target_weight': target_weight,
            'round_trip_id': state['round_trip_id'],
        })
        return

    shares_to_sell = current_shares - target_shares
    trade_price = open_price * (1 - slippage_rate)
    gross_amount = shares_to_sell * trade_price
    cost = gross_amount * sell_cost_rate
    net_inflow = gross_amount - cost
    state['cash'] += net_inflow
    state['shares'] -= shares_to_sell
    state['trade_count'] += 1
    state['target_weight_level'] = target_weight if state['shares'] > 0 else 0.0

    trade_records.append({
        'strategy_name': strategy_name,
        'trade_date': row['trade_date'],
        'action': 'sell',
        'reason': reason,
        'price': trade_price,
        'shares': shares_to_sell,
        'cash_after': state['cash'],
        'shares_after': state['shares'],
        'target_weight': target_weight,
        'round_trip_id': state['round_trip_id'],
    })

    if state['shares'] == 0:
        if state['entry_nav'] is not None:
            exit_nav = state['cash']
            state['round_trips'].append({
                'strategy_name': strategy_name,
                'round_trip_id': state['round_trip_id'],
                'entry_date': state['entry_date'],
                'exit_date': row['trade_date'],
                'holding_days': state['holding_days'],
                'trade_return': exit_nav / state['entry_nav'] - 1,
            })
        state['avg_cost'] = 0.0
        state['holding_days'] = 0
        state['entry_date'] = None
        state['entry_nav'] = None
        state['target_weight_level'] = 0.0


def run_strategy(df, initial_capital, trade_config, strategy_config, strategy_name, decide_target_func):
    """
    运行单个策略的事件驱动回测。
    """
    state = initialize_state(initial_capital)
    daily_records = []
    trade_records = []

    for index, row in df.iterrows():
        if state['pending_target_weight'] is not None:
            pending_reason = state['pending_reason']
            execute_rebalance(
                state=state,
                row=row,
                target_weight=state['pending_target_weight'],
                reason=pending_reason,
                trade_config=trade_config,
                trade_records=trade_records,
                strategy_name=strategy_name
            )
            if pending_reason == 'stop_loss' and state['shares'] == 0:
                state['cooldown_days_left'] = int(strategy_config.get('cooldown_days', 0))
            state['pending_target_weight'] = None
            state['pending_reason'] = None

        close_price = float(row['adj_close'])
        net_asset = state['cash'] + state['shares'] * close_price
        current_weight = (state['shares'] * close_price / net_asset) if net_asset > 0 else 0.0
        current_return = (close_price / state['avg_cost'] - 1) if state['shares'] > 0 and state['avg_cost'] > 0 else 0.0

        if state['shares'] > 0:
            state['holding_days'] += 1

        target_weight, reason = decide_target_func(
            row=row,
            state=state,
            strategy_config=strategy_config,
            current_weight=current_weight,
            current_return=current_return
        )

        if index < len(df) - 1 and abs(target_weight - state['target_weight_level']) > 1e-9:
            state['pending_target_weight'] = target_weight
            state['pending_reason'] = reason

        daily_records.append({
            'trade_date': row['trade_date'],
            'strategy_name': strategy_name,
            'net_asset': net_asset,
            'position_weight': current_weight,
            'cash': state['cash'],
            'shares': state['shares'],
            'anchor_score': row['anchor_score'],
            'composite_score': row['composite_score'],
            'adj_close': close_price,
            'ma_short': row.get('ma_short'),
            'ma_long': row.get('ma_long'),
            'trend_confirmed': row.get('trend_confirmed'),
            'cooldown_days_left': state['cooldown_days_left'],
            'signal_reason': reason,
            'holding_days': state['holding_days'],
        })

        if state['cooldown_days_left'] > 0:
            state['cooldown_days_left'] -= 1

    daily_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_records)
    round_trip_df = pd.DataFrame(state['round_trips'])
    return daily_df, trade_df, round_trip_df


def run_buy_and_hold(df, initial_capital, trade_config, benchmark_name):
    """
    回测 ETF 买入持有基准。
    """
    state = initialize_state(initial_capital)
    trade_records = []
    first_row = df.iloc[1].copy() if len(df) > 1 else df.iloc[0].copy()
    execute_rebalance(
        state=state,
        row=first_row,
        target_weight=1.0,
        reason='benchmark_buy',
        trade_config=trade_config,
        trade_records=trade_records,
        strategy_name=benchmark_name
    )

    daily_records = []
    for _, row in df.iterrows():
        close_price = float(row['adj_close'])
        net_asset = state['cash'] + state['shares'] * close_price
        position_weight = (state['shares'] * close_price / net_asset) if net_asset > 0 else 0.0
        daily_records.append({
            'trade_date': row['trade_date'],
            'strategy_name': benchmark_name,
            'net_asset': net_asset,
            'position_weight': position_weight,
            'cash': state['cash'],
            'shares': state['shares'],
            'adj_close': close_price,
            'signal_reason': 'benchmark_hold',
            'holding_days': 0,
        })

    daily_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_records)
    return daily_df, trade_df


def calculate_max_drawdown(net_value_series):
    """
    计算净值序列最大回撤。
    """
    rolling_max = net_value_series.cummax()
    drawdown = net_value_series / rolling_max - 1
    return drawdown.min()


def summarize_performance(daily_df, round_trip_df, initial_capital):
    """
    汇总策略绩效指标。
    """
    net_value = daily_df['net_asset'] / initial_capital
    daily_return = net_value.pct_change().dropna()
    total_return = net_value.iloc[-1] - 1
    periods = len(daily_df)
    annual_return = (net_value.iloc[-1] ** (252 / periods) - 1) if periods > 1 else np.nan
    max_drawdown = calculate_max_drawdown(net_value)
    sharpe = (
        daily_return.mean() / daily_return.std() * np.sqrt(252)
        if len(daily_return) > 1 and daily_return.std() > 0 else np.nan
    )
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    summary = {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'calmar': calmar,
        'avg_position_weight': daily_df['position_weight'].mean(),
        'trade_rounds': len(round_trip_df),
    }

    if len(round_trip_df) > 0:
        summary['trade_win_rate'] = (round_trip_df['trade_return'] > 0).mean()
        summary['avg_trade_return'] = round_trip_df['trade_return'].mean()
        summary['median_holding_days'] = round_trip_df['holding_days'].median()
    else:
        summary['trade_win_rate'] = np.nan
        summary['avg_trade_return'] = np.nan
        summary['median_holding_days'] = np.nan

    return summary


def merge_strategy_navs(strategy_daily_map, initial_capital):
    """
    合并各策略净值曲线。
    """
    merged = None
    for strategy_name, daily_df in strategy_daily_map.items():
        current = daily_df[['trade_date', 'net_asset', 'position_weight']].copy()
        current = current.rename(columns={
            'net_asset': f'{strategy_name}_net_asset',
            'position_weight': f'{strategy_name}_position_weight',
        })
        current[f'{strategy_name}_net_value'] = current[f'{strategy_name}_net_asset'] / initial_capital
        merged = current if merged is None else merged.merge(current, on='trade_date', how='outer')

    return merged.sort_values('trade_date').reset_index(drop=True)


def export_csv(df, filename):
    """
    导出 CSV 文件。
    """
    path = build_output_path(filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'已导出: {path}')


def plot_strategy_comparison(nav_df, benchmark_name):
    """
    绘制策略与基准净值、仓位对比图。
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    net_value_columns = [column for column in nav_df.columns if column.endswith('_net_value')]
    for column in net_value_columns:
        label = column.replace('_net_value', '')
        axes[0].plot(nav_df['trade_date'], nav_df[column], linewidth=2, label=label)

    axes[0].set_title('创业板ETF 温度交易系统净值对比')
    axes[0].set_ylabel('净值')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    position_columns = [
        column for column in nav_df.columns
        if column.endswith('_position_weight') and benchmark_name not in column
    ]
    for column in position_columns:
        label = column.replace('_position_weight', '')
        axes[1].plot(nav_df['trade_date'], nav_df[column] * 100, linewidth=1.8, label=label)

    axes[1].set_title('策略仓位路径')
    axes[1].set_ylabel('仓位（%）')
    axes[1].set_xlabel('时间')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    output_path = build_output_path('创业板ETF策略回测图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已生成图表: {output_path}')


def main():
    """
    主流程：读取温度序列与 ETF 行情，回测两套策略并输出结果。
    """
    configure_console_encoding()
    args = parse_args()
    config = load_parameter_config(args.config)
    data_config = config['data']
    trend_filter_config = config.get('trend_filter', {})
    managed_strategy_config = config['strategy']['value_managed']

    start_date = normalize_date(args.start or data_config['start_date'])
    end_date = normalize_date(args.end or data_config['end_date']) or pd.Timestamp.today().strftime('%Y%m%d')
    input_path = resolve_path(args.input or data_config['temperature_csv'])
    initial_capital = float(data_config['initial_capital'])

    print('=' * 70)
    print('创业板 ETF 温度交易系统回测')
    print(f'温度序列: {input_path}')
    print(f'ETF 标的: {data_config["etf_code"]}')
    print(f'分析区间: {start_date} ~ {end_date}')
    print('=' * 70)

    temp_df = load_temperature_series(
        input_path,
        start_date=start_date,
        end_date=end_date,
        rebound_days=managed_strategy_config.get('confirm_rebound_days', 5)
    )
    pro = get_tushare_pro()
    etf_df = get_etf_daily_data(
        pro=pro,
        ts_code=data_config['etf_code'],
        start_date=start_date,
        end_date=end_date,
        ma_short_window=trend_filter_config.get('short_ma_window', 20),
        ma_long_window=trend_filter_config.get('long_ma_window', 60)
    )
    backtest_df = prepare_backtest_data(temp_df, etf_df)

    value_only_daily, value_only_trades, value_only_rounds = run_strategy(
        df=backtest_df,
        initial_capital=initial_capital,
        trade_config=config['trade'],
        strategy_config=config['strategy']['value_only'],
        strategy_name='value_only',
        decide_target_func=decide_value_only_target
    )
    managed_daily, managed_trades, managed_rounds = run_strategy(
        df=backtest_df,
        initial_capital=initial_capital,
        trade_config=config['trade'],
        strategy_config=managed_strategy_config,
        strategy_name='value_managed',
        decide_target_func=decide_value_managed_target
    )
    benchmark_daily, benchmark_trades = run_buy_and_hold(
        df=backtest_df,
        initial_capital=initial_capital,
        trade_config=config['trade'],
        benchmark_name=config['plot']['benchmark_name']
    )

    nav_df = merge_strategy_navs({
        'value_only': value_only_daily,
        'value_managed': managed_daily,
        config['plot']['benchmark_name']: benchmark_daily,
    }, initial_capital)

    performance_rows = []
    for strategy_name, daily_df, round_df in [
        ('value_only', value_only_daily, value_only_rounds),
        ('value_managed', managed_daily, managed_rounds),
    ]:
        summary = summarize_performance(daily_df, round_df, initial_capital)
        performance_rows.append({'strategy_name': strategy_name, **summary})

    benchmark_summary = summarize_performance(
        benchmark_daily,
        pd.DataFrame(),
        initial_capital
    )
    performance_rows.append({'strategy_name': config['plot']['benchmark_name'], **benchmark_summary})
    performance_df = pd.DataFrame(performance_rows)

    trade_df = pd.concat([
        value_only_trades,
        managed_trades,
        benchmark_trades
    ], ignore_index=True)
    round_trip_df = pd.concat([
        value_only_rounds,
        managed_rounds
    ], ignore_index=True)

    export_csv(nav_df, '创业板ETF策略净值.csv')
    export_csv(performance_df, '创业板ETF策略绩效.csv')
    export_csv(trade_df, '创业板ETF策略交易明细.csv')
    export_csv(round_trip_df, '创业板ETF策略轮次明细.csv')
    plot_strategy_comparison(nav_df, config['plot']['benchmark_name'])

    print('=' * 70)
    print('策略绩效摘要')
    print('=' * 70)
    for _, row in performance_df.iterrows():
        sharpe_text = f"{row['sharpe']:.2f}" if pd.notna(row['sharpe']) else 'nan'
        print(
            f"{row['strategy_name']}: 总收益 {row['total_return'] * 100:.2f}% | "
            f"年化 {row['annual_return'] * 100:.2f}% | "
            f"最大回撤 {row['max_drawdown'] * 100:.2f}% | "
            f"夏普 {sharpe_text}"
        )


if __name__ == '__main__':
    main()

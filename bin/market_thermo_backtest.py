import argparse
import sys
import tomllib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def configure_console_encoding():
    """
    配置控制台编码，避免 Windows 下中文输出报错。
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
    生成输出文件绝对路径。
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


def load_parameter_config(config_path=None):
    """
    加载回测参数文件。
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'market_thermo_backtest_params.toml'
    else:
        config_path = resolve_path(config_path)

    with config_path.open('rb') as file:
        config = tomllib.load(file)

    print(f'已加载回测参数文件: {config_path}')
    return config


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='市场温度有效性检验')
    parser.add_argument('--input', help='温度时间序列 CSV 路径')
    parser.add_argument('--start', help='分析开始日期，格式 YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--end', help='分析结束日期，格式 YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--config', help='回测参数文件路径')
    return parser.parse_args()


def normalize_date(value):
    """
    规范日期字符串格式。
    """
    if value is None:
        return None
    return pd.to_datetime(value).strftime('%Y-%m-%d')


def load_temperature_series(input_path, start_date=None, end_date=None):
    """
    读取市场温度时间序列数据。
    """
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)

    if start_date:
        df = df[df['trade_date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['trade_date'] <= pd.to_datetime(end_date)]

    if len(df) == 0:
        raise ValueError('筛选后无有效时间序列数据')

    return df.reset_index(drop=True)


def compute_forward_metrics(price_series, horizon):
    """
    计算给定持有期下的前瞻收益、最大上涨和最大回撤。
    """
    values = price_series.to_numpy(dtype=float)
    length = len(values)
    forward_return = np.full(length, np.nan)
    forward_max_gain = np.full(length, np.nan)
    forward_max_drawdown = np.full(length, np.nan)

    for index in range(length - horizon):
        entry = values[index]
        future_window = values[index + 1:index + horizon + 1]
        relative_window = future_window / entry - 1
        forward_return[index] = values[index + horizon] / entry - 1
        forward_max_gain[index] = np.nanmax(relative_window)
        forward_max_drawdown[index] = np.nanmin(relative_window)

    return forward_return, forward_max_gain, forward_max_drawdown


def add_forward_columns(df, index_columns, forward_days):
    """
    为各指数添加前瞻收益与路径风险指标。
    """
    result = df.copy()
    for index_column in index_columns:
        for horizon in forward_days:
            forward_return, forward_max_gain, forward_max_drawdown = compute_forward_metrics(
                result[index_column],
                horizon
            )
            result[f'{index_column}_ret_{horizon}'] = forward_return
            result[f'{index_column}_max_gain_{horizon}'] = forward_max_gain
            result[f'{index_column}_max_drawdown_{horizon}'] = forward_max_drawdown
    return result


def assign_quantile_bucket(series, group_count):
    """
    将分数按全样本历史位置划分为若干分组。
    """
    rank_pct = series.rank(method='average', pct=True)
    bucket = np.ceil(rank_pct * group_count).clip(1, group_count)
    return bucket.astype('Int64')


def summarize_returns(frame, return_col, max_gain_col, max_drawdown_col):
    """
    汇总一组样本的收益、胜率和风险统计。
    """
    valid = frame[[return_col, max_gain_col, max_drawdown_col]].dropna()
    if len(valid) == 0:
        return {
            'sample_count': 0,
            'avg_return': np.nan,
            'median_return': np.nan,
            'win_rate': np.nan,
            'avg_max_gain': np.nan,
            'avg_max_drawdown': np.nan,
            'profit_loss_ratio': np.nan,
        }

    positive = valid[valid[return_col] > 0][return_col]
    negative = valid[valid[return_col] < 0][return_col]
    if len(positive) > 0 and len(negative) > 0:
        profit_loss_ratio = positive.mean() / abs(negative.mean())
    else:
        profit_loss_ratio = np.nan

    return {
        'sample_count': int(len(valid)),
        'avg_return': valid[return_col].mean(),
        'median_return': valid[return_col].median(),
        'win_rate': (valid[return_col] > 0).mean(),
        'avg_max_gain': valid[max_gain_col].mean(),
        'avg_max_drawdown': valid[max_drawdown_col].mean(),
        'profit_loss_ratio': profit_loss_ratio,
    }


def build_group_analysis(df, score_columns, index_columns, forward_days, group_count):
    """
    按温度分组统计未来收益表现。
    """
    rows = []
    for score_column in score_columns:
        bucket_column = f'{score_column}_bucket'
        working = df.copy()
        working[bucket_column] = assign_quantile_bucket(working[score_column], group_count)

        for index_column in index_columns:
            for horizon in forward_days:
                return_col = f'{index_column}_ret_{horizon}'
                max_gain_col = f'{index_column}_max_gain_{horizon}'
                max_drawdown_col = f'{index_column}_max_drawdown_{horizon}'

                for bucket in range(1, group_count + 1):
                    subset = working[working[bucket_column] == bucket]
                    stats = summarize_returns(subset, return_col, max_gain_col, max_drawdown_col)
                    rows.append({
                        'score_column': score_column,
                        'index_column': index_column,
                        'horizon_days': horizon,
                        'bucket': bucket,
                        **stats,
                    })

    return pd.DataFrame(rows)


def build_threshold_analysis(df, score_columns, index_columns, forward_days, config):
    """
    按绝对阈值和历史分位阈值检验低温区、高温区的后续表现。
    """
    rows = []

    for score_column in score_columns:
        score_series = df[score_column]

        for threshold in config['thresholds']['absolute_low']:
            mask = score_series <= threshold
            rows.extend(
                collect_threshold_rows(
                    df,
                    score_column,
                    index_columns,
                    forward_days,
                    'absolute',
                    f'<= {threshold}',
                    mask,
                    config
                )
            )

        for threshold in config['thresholds']['absolute_high']:
            mask = score_series >= threshold
            rows.extend(
                collect_threshold_rows(
                    df,
                    score_column,
                    index_columns,
                    forward_days,
                    'absolute',
                    f'>= {threshold}',
                    mask,
                    config
                )
            )

        for quantile in config['thresholds']['quantile_low']:
            level = score_series.quantile(quantile)
            mask = score_series <= level
            rows.extend(
                collect_threshold_rows(
                    df,
                    score_column,
                    index_columns,
                    forward_days,
                    'quantile',
                    f'<= Q{int(quantile * 100)} ({level:.1f})',
                    mask,
                    config
                )
            )

        for quantile in config['thresholds']['quantile_high']:
            level = score_series.quantile(quantile)
            mask = score_series >= level
            rows.extend(
                collect_threshold_rows(
                    df,
                    score_column,
                    index_columns,
                    forward_days,
                    'quantile',
                    f'>= Q{int(quantile * 100)} ({level:.1f})',
                    mask,
                    config
                )
            )

    return pd.DataFrame(rows)


def get_effective_cooldown_days(config, horizon_days):
    """
    获取事件去重窗口。
    """
    cooldown_days = int(config.get('signals', {}).get('cooldown_days', 0))
    return cooldown_days if cooldown_days > 0 else horizon_days


def collect_threshold_rows(df, score_column, index_columns, forward_days, threshold_type, threshold_label, mask, config):
    """
    汇总单个阈值条件下的各指数统计结果。
    """
    rows = []
    raw_signal_count = int(mask.sum())

    for index_column in index_columns:
        for horizon in forward_days:
            cooldown_days = get_effective_cooldown_days(config, horizon)
            signal_index = select_non_overlapping_signals(mask, cooldown_days)
            subset = df.loc[signal_index].copy()
            return_col = f'{index_column}_ret_{horizon}'
            max_gain_col = f'{index_column}_max_gain_{horizon}'
            max_drawdown_col = f'{index_column}_max_drawdown_{horizon}'
            stats = summarize_returns(subset, return_col, max_gain_col, max_drawdown_col)
            rows.append({
                'score_column': score_column,
                'index_column': index_column,
                'horizon_days': horizon,
                'threshold_type': threshold_type,
                'threshold_label': threshold_label,
                'raw_signal_count': raw_signal_count,
                'cooldown_days': cooldown_days,
                **stats,
            })

    return rows


def select_non_overlapping_signals(mask, cooldown_days):
    """
    对连续信号做去重，避免同一段低温/高温被重复统计。
    """
    signal_positions = np.flatnonzero(mask.to_numpy())
    selected_positions = []
    last_position = -cooldown_days

    for position in signal_positions:
        if position - last_position >= cooldown_days:
            selected_positions.append(position)
            last_position = position

    return mask.index[selected_positions]


def get_score_display_name(score_column):
    """
    返回温度列的显示名称。
    """
    return {
        'composite_score': '综合温度',
        'anchor_score': '价值温度',
    }.get(score_column, score_column)


def get_index_display_name(index_column):
    """
    返回指数列的显示名称。
    """
    return {
        'hs300_close': '沪深300',
        'cyb_close': '创业板指',
    }.get(index_column, index_column)


def build_event_path_analysis(df, config):
    """
    构建低温/高温事件触发后的平均累计收益路径。
    """
    event_config = config['event_path']
    rows = []

    for score_column in event_config['score_columns']:
        score_series = df[score_column]

        for index_column in event_config['index_columns']:
            for threshold in event_config['absolute_low']:
                rows.extend(
                    collect_event_path_rows(
                        df=df,
                        score_column=score_column,
                        index_column=index_column,
                        path_days=event_config['path_days'],
                        threshold_type='absolute',
                        threshold_label=f'<= {threshold}',
                        direction='low',
                        mask=score_series <= threshold,
                        config=config
                    )
                )

            for threshold in event_config['absolute_high']:
                rows.extend(
                    collect_event_path_rows(
                        df=df,
                        score_column=score_column,
                        index_column=index_column,
                        path_days=event_config['path_days'],
                        threshold_type='absolute',
                        threshold_label=f'>= {threshold}',
                        direction='high',
                        mask=score_series >= threshold,
                        config=config
                    )
                )

            for quantile in event_config['quantile_low']:
                level = score_series.quantile(quantile)
                rows.extend(
                    collect_event_path_rows(
                        df=df,
                        score_column=score_column,
                        index_column=index_column,
                        path_days=event_config['path_days'],
                        threshold_type='quantile',
                        threshold_label=f'<= Q{int(quantile * 100)} ({level:.1f})',
                        direction='low',
                        mask=score_series <= level,
                        config=config
                    )
                )

            for quantile in event_config['quantile_high']:
                level = score_series.quantile(quantile)
                rows.extend(
                    collect_event_path_rows(
                        df=df,
                        score_column=score_column,
                        index_column=index_column,
                        path_days=event_config['path_days'],
                        threshold_type='quantile',
                        threshold_label=f'>= Q{int(quantile * 100)} ({level:.1f})',
                        direction='high',
                        mask=score_series >= level,
                        config=config
                    )
                )

    return pd.DataFrame(rows)


def collect_event_path_rows(df, score_column, index_column, path_days, threshold_type, threshold_label, direction, mask, config):
    """
    汇总单个阈值事件的平均累计收益路径。
    """
    raw_signal_count = int(mask.sum())
    cooldown_days = get_effective_cooldown_days(config, path_days)
    signal_index = select_non_overlapping_signals(mask, cooldown_days)
    price_values = df[index_column].to_numpy(dtype=float)
    paths = []

    for signal_label in signal_index:
        signal_position = df.index.get_loc(signal_label)
        if signal_position + path_days >= len(df):
            continue

        base_price = price_values[signal_position]
        future_prices = price_values[signal_position:signal_position + path_days + 1]
        paths.append(future_prices / base_price - 1)

    if not paths:
        return []

    path_matrix = np.vstack(paths)
    avg_path = np.nanmean(path_matrix, axis=0)
    median_path = np.nanmedian(path_matrix, axis=0)
    rows = []

    for day_index, (avg_value, median_value) in enumerate(zip(avg_path, median_path)):
        rows.append({
            'score_column': score_column,
            'index_column': index_column,
            'direction': direction,
            'threshold_type': threshold_type,
            'threshold_label': threshold_label,
            'path_days': path_days,
            'day': day_index,
            'raw_signal_count': raw_signal_count,
            'sample_count': len(paths),
            'cooldown_days': cooldown_days,
            'avg_cumulative_return': avg_value,
            'median_cumulative_return': median_value,
        })

    return rows


def export_csv(df, filename):
    """
    导出 CSV 文件。
    """
    path = build_output_path(filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'已导出: {path}')


def plot_backtest_dashboard(group_df, threshold_df, display_horizon, output_path='市场温度有效性检验图.png'):
    """
    绘制分组收益与阈值检验统计图。
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    plot_group_panel(
        axes[0, 0],
        group_df,
        score_column='composite_score',
        index_column='hs300_close',
        title='综合温度十分位与沪深300前瞻收益'
    )
    plot_group_panel(
        axes[0, 1],
        group_df,
        score_column='anchor_score',
        index_column='hs300_close',
        title='价值温度十分位与沪深300前瞻收益'
    )
    plot_threshold_panel(
        axes[1, 0],
        threshold_df,
        index_column='hs300_close',
        display_horizon=display_horizon,
        title=f'沪深300阈值检验（{display_horizon}日）'
    )
    plot_threshold_panel(
        axes[1, 1],
        threshold_df,
        index_column='cyb_close',
        display_horizon=display_horizon,
        title=f'创业板指阈值检验（{display_horizon}日）'
    )

    plt.tight_layout()
    plt.savefig(build_output_path(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已生成图表: {build_output_path(output_path)}')


def plot_group_panel(ax, group_df, score_column, index_column, title):
    """
    绘制十分位与未来收益的关系图。
    """
    filtered = group_df[
        (group_df['score_column'] == score_column) &
        (group_df['index_column'] == index_column)
    ].copy()
    if len(filtered) == 0:
        ax.set_title(title)
        return

    pivot = filtered.pivot(index='bucket', columns='horizon_days', values='avg_return') * 100
    for horizon in pivot.columns:
        ax.plot(pivot.index, pivot[horizon], marker='o', linewidth=2, label=f'{horizon}日')

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('温度分组（1=最低温，10=最高温）')
    ax.set_ylabel('平均前瞻收益率（%）')
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_threshold_panel(ax, threshold_df, index_column, display_horizon, title):
    """
    绘制阈值事件在指定持有期下的平均收益图。
    """
    filtered = threshold_df[
        (threshold_df['index_column'] == index_column) &
        (threshold_df['horizon_days'] == display_horizon)
    ].copy()
    if len(filtered) == 0:
        ax.set_title(title)
        return

    filtered['label'] = filtered['score_column'].map({
        'composite_score': '综合',
        'anchor_score': '价值'
    }) + ' ' + filtered['threshold_label']
    filtered = filtered.sort_values(['score_column', 'threshold_type', 'threshold_label']).reset_index(drop=True)

    bars = ax.bar(filtered['label'], filtered['avg_return'] * 100, color='steelblue', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_ylabel('平均前瞻收益率（%）')
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars, filtered['avg_return'] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)


def plot_event_path_dashboard(event_path_df, output_path='市场温度事件路径图.png'):
    """
    绘制低温/高温事件触发后的平均累计收益路径图。
    """
    if len(event_path_df) == 0:
        return

    index_columns = event_path_df['index_column'].dropna().unique().tolist()
    path_days = int(event_path_df['path_days'].max())
    fig, axes = plt.subplots(
        nrows=len(index_columns),
        ncols=2,
        figsize=(18, 5 * len(index_columns)),
        sharex=True
    )

    axes = np.atleast_2d(axes)
    directions = [('low', '低温信号后路径'), ('high', '高温信号后路径')]

    for row_idx, index_column in enumerate(index_columns):
        index_name = get_index_display_name(index_column)

        for col_idx, (direction, title_prefix) in enumerate(directions):
            axis = axes[row_idx, col_idx]
            filtered = event_path_df[
                (event_path_df['index_column'] == index_column) &
                (event_path_df['direction'] == direction)
            ].copy()

            if len(filtered) == 0:
                axis.set_title(f'{title_prefix} - {index_name}')
                axis.axhline(0, color='gray', linestyle='--', linewidth=1)
                axis.grid(True, alpha=0.25)
                continue

            for _, scenario_df in filtered.groupby(['score_column', 'threshold_type', 'threshold_label']):
                scenario_df = scenario_df.sort_values('day')
                sample_count = int(scenario_df['sample_count'].iloc[0])
                if sample_count == 0:
                    continue

                score_name = get_score_display_name(scenario_df['score_column'].iloc[0])
                threshold_label = scenario_df['threshold_label'].iloc[0]
                label = f'{score_name} {threshold_label} (n={sample_count})'
                axis.plot(
                    scenario_df['day'],
                    scenario_df['avg_cumulative_return'] * 100,
                    linewidth=2,
                    label=label
                )

            axis.axhline(0, color='gray', linestyle='--', linewidth=1)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel('平均累计收益率（%）')
            axis.set_title(f'{title_prefix} - {index_name}')
            axis.legend(loc='best')

    for axis in axes[-1]:
        axis.set_xlabel(f'触发后交易日（0 ~ {path_days}）')

    plt.tight_layout()
    plt.savefig(build_output_path(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已生成图表: {build_output_path(output_path)}')


def print_summary(threshold_df):
    """
    打印最关键的阈值检验摘要。
    """
    focus = threshold_df[
        (threshold_df['index_column'] == 'hs300_close') &
        (threshold_df['horizon_days'] == 60)
    ].copy()
    if len(focus) == 0:
        return

    best_low = focus[focus['threshold_label'].str.startswith('<=')].sort_values('avg_return', ascending=False).head(1)
    best_high = focus[focus['threshold_label'].str.startswith('>=')].sort_values('avg_return', ascending=True).head(1)

    print('=' * 70)
    print('温度有效性检验摘要（沪深300，60日持有期）')
    print('=' * 70)
    if len(best_low) > 0:
        row = best_low.iloc[0]
        print(
            f"最佳低温信号: {row['score_column']} {row['threshold_label']} | "
            f"原始信号 {int(row['raw_signal_count'])} 次，去重后样本 {int(row['sample_count'])} 次 | "
            f"胜率 {row['win_rate'] * 100:.1f}% | "
            f"平均收益 {row['avg_return'] * 100:.2f}% | 平均最大回撤 {row['avg_max_drawdown'] * 100:.2f}%"
        )
    if len(best_high) > 0:
        row = best_high.iloc[0]
        print(
            f"最佳高温反向信号: {row['score_column']} {row['threshold_label']} | "
            f"原始信号 {int(row['raw_signal_count'])} 次，去重后样本 {int(row['sample_count'])} 次 | "
            f"胜率 {row['win_rate'] * 100:.1f}% | "
            f"平均收益 {row['avg_return'] * 100:.2f}% | 平均最大回撤 {row['avg_max_drawdown'] * 100:.2f}%"
        )
    print('=' * 70)


def main():
    """
    主流程：读取温度序列，完成分组回测、阈值检验并输出图表与 CSV。
    """
    configure_console_encoding()
    args = parse_args()
    config = load_parameter_config(args.config)

    input_path = resolve_path(args.input or config['data']['input_csv'])
    start_date = normalize_date(args.start)
    end_date = normalize_date(args.end)
    analysis_config = config['analysis']

    print('=' * 70)
    print('市场温度有效性检验')
    print(f'输入文件: {input_path}')
    if start_date or end_date:
        print(f'筛选区间: {start_date or "起始"} ~ {end_date or "结束"}')
    print('=' * 70)

    df = load_temperature_series(input_path, start_date, end_date)
    df = add_forward_columns(
        df,
        analysis_config['index_columns'],
        analysis_config['forward_days']
    )

    group_df = build_group_analysis(
        df,
        analysis_config['score_columns'],
        analysis_config['index_columns'],
        analysis_config['forward_days'],
        analysis_config['group_count']
    )
    threshold_df = build_threshold_analysis(
        df,
        analysis_config['score_columns'],
        analysis_config['index_columns'],
        analysis_config['forward_days'],
        config
    )
    event_path_df = build_event_path_analysis(df, config)

    export_csv(df, '市场温度回测明细.csv')
    export_csv(group_df, '市场温度分组前瞻收益.csv')
    export_csv(threshold_df, '市场温度阈值检验.csv')
    export_csv(event_path_df, '市场温度事件路径.csv')
    plot_backtest_dashboard(
        group_df,
        threshold_df,
        config['plot']['display_horizon']
    )
    plot_event_path_dashboard(event_path_df)
    print_summary(threshold_df)


if __name__ == '__main__':
    main()

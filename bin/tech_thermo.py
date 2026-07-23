# ============================================================
# 技术广度维度指标研究（五维框架 - 权重15%）
# 核心指标：
#   1. 归一化新高-新低线（累积差值比例）
#   2. 麦克莱伦震荡指标（19日EMA - 39日EMA）
#   3. 阿姆氏指标（Arms Index，TRIN）
# 数据源：Tushare Pro
# ============================================================

import argparse
import os
from pathlib import Path
import sys
import tomllib
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
warnings.filterwarnings('ignore')

def configure_console_encoding():
    """
    配置控制台输出编码，避免 Windows 下打印 emoji 时报错。
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')


def load_env_value(key, env_path=None):
    """
    从 .env 文件读取指定环境变量。
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


def get_project_root():
    """
    获取项目根目录，便于统一定位 .env 和参数文件。
    """
    script_path = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / 'bin' / 'tech_thermo.py'
    return script_path.parent.parent


def build_output_path(filename):
    """
    构造输出文件路径，确保图表统一写到 outputs 目录。
    """
    output_dir = get_project_root() / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def load_parameter_config(config_path=None):
    """
    加载技术广度分析参数文件。
    """
    if config_path is None:
        config_path = get_project_root() / 'config' / 'tech_thermo_params.toml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f'未找到参数文件: {config_path}')

    with config_path.open('rb') as file:
        config = tomllib.load(file)

    print(f"✅ 已加载参数文件: {config_path}")
    return config

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一步：初始化 Tushare
# ============================================================

configure_console_encoding()
TOKEN = os.getenv('TUSHARE_API_KEY') or load_env_value('TUSHARE_API_KEY')
pro = ts.pro_api(TOKEN)

print("✅ Tushare 初始化成功")
print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ============================================================
# 第二步：获取全市场日线数据（用于计算广度指标）
# ============================================================

def get_all_stock_daily(start_date='20150101', end_date=None):
    """
    获取全市场个股的日度交易数据
    用于计算：涨跌家数、52周新高/新低、年线位置等

    数据源：Tushare daily（日度行情）
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"\n📊 获取全市场日线数据: {start_date} ~ {end_date}")
    print("   ⚠️ 数据量较大，可能需要较长时间...")
    
    try:
        # 获取股票列表（剔除北交所）
        stock_list = pro.stock_basic(
            exchange='', 
            list_status='L', 
            fields='ts_code,symbol,name,market'
        )
        
        # 过滤：只保留主板、创业板、科创板（剔除北交所）
        stock_list = stock_list[~stock_list['market'].str.contains('北交所', na=False)]
        
        # 进一步过滤：剔除ST、*ST股票
        stock_list = stock_list[~stock_list['name'].str.contains('ST|退', na=False)]
        
        print(f"   ✅ 股票列表: {len(stock_list)} 只")
        
        # 获取交易日历，按交易日逐日拉取全市场日线，避免批量 ts_code 接口拿不到所需字段。
        trade_cal = pro.trade_cal(
            exchange='',
            start_date=start_date,
            end_date=end_date,
            fields='cal_date,is_open'
        )
        trade_dates = (
            trade_cal[trade_cal['is_open'] == 1]['cal_date']
            .sort_values()
            .tolist()
        )
        print(f"   ✅ 交易日数量: {len(trade_dates)}")

        stock_codes = set(stock_list['ts_code'])
        all_data = []

        for i, trade_date in enumerate(trade_dates, start=1):
            try:
                df_day = pro.daily(
                    trade_date=trade_date,
                    fields='ts_code,trade_date,close,vol,amount,pct_chg'
                )

                if df_day is None or len(df_day) == 0:
                    continue

                df_day = df_day[df_day['ts_code'].isin(stock_codes)].copy()
                if len(df_day) == 0:
                    continue

                df_day = df_day.rename(columns={'vol': 'volume'})
                all_data.append(df_day)

                if i % 50 == 0 or i == len(trade_dates):
                    print(f"   📈 进度: {i}/{len(trade_dates)}")

            except Exception as e:
                print(f"   ⚠️ {trade_date} 获取失败: {e}")
                continue
        
        if not all_data:
            print("   ❌ 未获取到任何数据")
            return None
        
        # 合并所有数据
        df_all = pd.concat(all_data, ignore_index=True)
        df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
        df_all = df_all.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        print(f"\n   ✅ 获取到 {len(df_all)} 条日度记录")
        print(f"   日期范围: {df_all['trade_date'].min().strftime('%Y-%m-%d')} ~ {df_all['trade_date'].max().strftime('%Y-%m-%d')}")
        
        return df_all
        
    except Exception as e:
        print(f"   ❌ 获取数据失败: {e}")
        return None


def get_index_price(start_date='20150101', end_date=None):
    """
    获取沪深300指数收盘价（用于叠加对比图）
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        df_idx = pro.index_daily(
            ts_code='000300.SH',
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,close'
        )
        df_idx['trade_date'] = pd.to_datetime(df_idx['trade_date'])
        df_idx = df_idx.sort_values('trade_date').reset_index(drop=True)
        return df_idx
    except Exception as e:
        print(f"   ⚠️ 获取指数数据失败: {e}")
        return None


def rolling_percentile(series, window, min_periods=None):
    """
    计算滚动窗口内的历史分位数，结果范围为 0-100。
    """
    def _calc(values):
        current = values[-1]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return (valid <= current).sum() / len(valid) * 100

    if min_periods is None:
        min_periods = max(30, window // 5)

    return series.rolling(window=window, min_periods=min_periods).apply(_calc, raw=True)


# ============================================================
# 第三步：计算技术广度指标
# ============================================================

def compute_breadth_indicators(df_daily, config):
    """
    计算三个技术广度指标：
    1. 新高-新低累积广度线及其滚动历史分位
    2. 中长期麦克莱伦震荡指标（39日EMA - 78日EMA）
    3. 阿姆氏指标（Arms Index，TRIN，含去极值和平滑）
    
    参数：
    - df_daily: 全市场日线数据（含ts_code, trade_date, close, volume, amount等）
    
    返回：
    - df_breadth: 按日期聚合的广度指标DataFrame
    """
    
    print("\n" + "=" * 70)
    print("🔄 计算技术广度指标...")
    print("=" * 70)
    
    # 获取所有交易日期
    dates = df_daily['trade_date'].unique()
    dates = np.sort(dates)
    
    breadths = []
    
    high_low_config = config['high_low']
    mcclellan_config = config['mcclellan']
    arms_config = config['arms']

    window_52w = high_low_config['lookback_days']
    
    for i, date in enumerate(dates):
        # 获取当日所有股票数据
        df_day = df_daily[df_daily['trade_date'] == date]
        total_stocks = len(df_day)
        
        if total_stocks == 0:
            continue
        
        # ----- 指标1：归一化新高-新低线 -----
        # 获取过去252个交易日的数据（含当日）
        start_idx = max(0, i - window_52w + 1)
        df_window = df_daily[df_daily['trade_date'].isin(dates[start_idx:i+1])]
        
        # 计算每只股票的52周最高价和最低价
        df_high_low = df_window.groupby('ts_code').agg({
            'close': ['max', 'min']
        }).reset_index()
        df_high_low.columns = ['ts_code', 'high_52w', 'low_52w']
        
        # 合并当日价格
        df_day_with_hl = pd.merge(df_day[['ts_code', 'close']], df_high_low, on='ts_code', how='left')
        
        # 判断创新高/新低
        new_high_count = len(
            df_day_with_hl[df_day_with_hl['close'] >= df_day_with_hl['high_52w'] * high_low_config['new_high_tolerance']]
        )
        new_low_count = len(
            df_day_with_hl[df_day_with_hl['close'] <= df_day_with_hl['low_52w'] * high_low_config['new_low_tolerance']]
        )
        
        # 归一化：计算比例差值
        pct_new_high = new_high_count / total_stocks
        pct_new_low = new_low_count / total_stocks
        diff_pct = pct_new_high - pct_new_low  # 范围[-1, 1]
        
        # ----- 指标2：麦克莱伦震荡指标（基于当日涨跌幅） -----
        advances = len(df_day[df_day['pct_chg'] > 0])
        declines = len(df_day[df_day['pct_chg'] < 0])
        
        # ----- 指标3：阿姆氏指标（需要成交量和涨跌家数） -----
        if 'volume' in df_day.columns and advances > 0 and declines > 0 and total_stocks > 0:
            adv_volume = df_day[df_day['pct_chg'] > 0]['volume'].sum()
            dec_volume = df_day[df_day['pct_chg'] < 0]['volume'].sum()

            if dec_volume > 0 and adv_volume > 0:
                arms_index = (advances / declines) / (adv_volume / dec_volume)
            else:
                arms_index = np.nan
        else:
            arms_index = np.nan
        
        # 记录结果
        breadths.append({
            'trade_date': date,
            'total_stocks': total_stocks,
            'new_high_count': new_high_count,
            'new_low_count': new_low_count,
            'pct_new_high': pct_new_high * 100,  # 转为百分比
            'pct_new_low': pct_new_low * 100,
            'diff_pct': diff_pct * 100,  # 转为百分比
            'advances': advances,
            'declines': declines,
            'arms_index': arms_index
        })
        
        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"   📊 进度: {i+1}/{len(dates)}")
    
    df_breadth = pd.DataFrame(breadths)
    print(f"   ✅ 生成 {len(df_breadth)} 个交易日的广度数据")
    
    # ----- 计算累积新高-新低线 -----
    df_breadth['cumulative_diff'] = df_breadth['diff_pct'].cumsum()
    # 用 3 年滚动分位表示其历史位置，真正归一化到 0-100。
    df_breadth['cumulative_diff_norm'] = rolling_percentile(
        df_breadth['cumulative_diff'],
        window=high_low_config['norm_window_days'],
        min_periods=high_low_config['norm_min_periods']
    )
    
    # ----- 计算麦克莱伦震荡指标 -----
    # 使用上涨/下跌家数占全部股票数的净差，降低市场扩容带来的量纲影响。
    df_breadth['net_advance_ratio'] = (
        (df_breadth['advances'] - df_breadth['declines']) / df_breadth['total_stocks'] * 100
    )
    # 用更慢的 39/78 EMA 参数，更适合中长期冷热判断。
    fast_span = mcclellan_config['fast_ema_span']
    slow_span = mcclellan_config['slow_ema_span']
    df_breadth['mcclellan_fast_ema'] = df_breadth['net_advance_ratio'].ewm(span=fast_span, adjust=False).mean()
    df_breadth['mcclellan_slow_ema'] = df_breadth['net_advance_ratio'].ewm(span=slow_span, adjust=False).mean()
    df_breadth['mcclellan'] = df_breadth['mcclellan_fast_ema'] - df_breadth['mcclellan_slow_ema']

    # ----- 计算阿姆氏指标的去极值结果 -----
    arms_valid = df_breadth['arms_index'].dropna()
    if len(arms_valid) >= 20:
        arms_clip_low = arms_valid.quantile(arms_config['clip_quantile_low'])
        arms_clip_high = arms_valid.quantile(arms_config['clip_quantile_high'])
    else:
        arms_clip_low = 0.5
        arms_clip_high = 2.5

    df_breadth['arms_index_clipped'] = df_breadth['arms_index'].clip(lower=arms_clip_low, upper=arms_clip_high)
    arms_ma_window = arms_config['ma_window']
    arms_ma_min_periods = min(arms_config['ma_min_periods'], arms_ma_window)
    df_breadth['arms_ma'] = df_breadth['arms_index_clipped'].rolling(
        window=arms_ma_window,
        min_periods=arms_ma_min_periods
    ).mean()
    
    return df_breadth


# ============================================================
# 第四步：计算技术广度综合得分
# ============================================================

def compute_breadth_score(df_breadth, config):
    """
    计算技术广度综合得分（0-100分）
    
    三个子指标：
    1. 归一化新高-新低线（权重40%）
    2. 麦克莱伦震荡指标（权重35%）
    3. 阿姆氏指标（权重25%）
    
    极寒（底部）信号：三个指标均处于历史低位，且与指数形成背离
    极热（顶部）信号：三个指标均处于历史高位，且与指数形成背离
    """
    
    print("\n" + "=" * 70)
    print("🌡️ 计算技术广度综合得分...")
    print("=" * 70)
    
    if df_breadth is None or len(df_breadth) == 0:
        print("❌ 无数据")
        return None
    
    # 计算各指标的历史分位数（基于全部历史数据）
    def get_percentile(series, value):
        """计算值在历史序列中的分位数"""
        return (series <= value).sum() / len(series) * 100

    mcclellan_config = config['mcclellan']
    arms_config = config['arms']
    weights = config['scoring']['weights']
    
    # ---------- 子指标1：归一化新高-新低线得分 ----------
    # 逻辑：值越高 → 市场越热（得分越高）
    # 值越低 → 市场越冷（得分越低）
    current_cum_diff = df_breadth['cumulative_diff'].iloc[-1]
    current_cum_diff_norm = df_breadth['cumulative_diff_norm'].iloc[-1]
    pct_cum_diff = get_percentile(df_breadth['cumulative_diff'], current_cum_diff)
    score_cum_diff = current_cum_diff_norm if pd.notna(current_cum_diff_norm) else pct_cum_diff
    
    # ---------- 子指标2：麦克莱伦震荡指标得分 ----------
    # 逻辑：值越高 → 市场越热
    # 注意：麦克莱伦指标一般在-100到+100之间波动
    current_mcclellan = df_breadth['mcclellan'].iloc[-1]
    pct_mcclellan = get_percentile(df_breadth['mcclellan'].dropna(), current_mcclellan)
    score_mcclellan = pct_mcclellan
    
    # ---------- 子指标3：阿姆氏指标得分 ----------
    # 逻辑：阿姆氏指标越高 → 市场越恐慌（得分越低）
    # 阿姆氏指标越低 → 市场越亢奋（得分越高）
    # 使用去极值后的均线来判断冷热，而不是受毛刺影响更大的原始值。
    current_arms = df_breadth['arms_ma'].iloc[-1]
    if pd.notna(current_arms):
        pct_arms = get_percentile(df_breadth['arms_ma'].dropna(), current_arms)
        arms_score = 100 - pct_arms
        score_arms = arms_score
    else:
        pct_arms = np.nan
        score_arms = 50
    
    # ---------- 综合得分（加权平均） ----------
    total_score = (
        score_cum_diff * weights['cum_diff'] +
        score_mcclellan * weights['mcclellan'] +
        score_arms * weights['arms']
    )
    
    result = {
        'total_score': total_score,
        'score_cum_diff': score_cum_diff,
        'score_mcclellan': score_mcclellan,
        'score_arms': score_arms,
        'current_cum_diff': current_cum_diff,
        'current_cum_diff_norm': current_cum_diff_norm,
        'current_mcclellan': current_mcclellan,
        'current_arms': current_arms,
        'pct_cum_diff': pct_cum_diff,
        'pct_mcclellan': pct_mcclellan,
        'pct_arms': pct_arms,
        'mcclellan_fast_span': mcclellan_config['fast_ema_span'],
        'mcclellan_slow_span': mcclellan_config['slow_ema_span'],
        'arms_ma_window': arms_config['ma_window']
    }
    
    print(f"\n📊 当前技术广度指标:")
    print(f"   📈 新高-新低累积广度线: {current_cum_diff:.2f} (全样本分位: {pct_cum_diff:.1f}%, 3年归一化: {current_cum_diff_norm:.1f}) → 得分: {score_cum_diff:.1f}")
    print(f"   📈 麦克莱伦震荡指标({mcclellan_config['fast_ema_span']}EMA-{mcclellan_config['slow_ema_span']}EMA): {current_mcclellan:.2f} (分位: {pct_mcclellan:.1f}%) → 得分: {score_mcclellan:.1f}")
    print(f"   📈 阿姆氏指标(去极值{arms_config['ma_window']}日MA): {current_arms:.2f} (分位: {pct_arms:.1f}%) → 得分: {score_arms:.1f}")
    print(f"   ─────────────────────")
    print(f"   🌡️ 技术广度综合得分: {total_score:.1f} / 100")
    
    return result


# ============================================================
# 第五步：信号判断
# ============================================================

def judge_breadth_signal(df_breadth, breadth_score):
    """
    判断技术广度信号，特别关注背离信号
    """
    
    print("\n" + "=" * 70)
    print("📊 技术广度信号判断")
    print("=" * 70)
    
    if df_breadth is None or len(df_breadth) < 60:
        print("⚠️ 数据不足，无法判断信号")
        return
    
    # 获取最近60个交易日的数据
    recent = df_breadth.tail(60).copy()
    recent_idx = df_breadth.tail(60)['cumulative_diff']
    
    # ----- 检测背离信号 -----
    # 需要沪深300指数数据配合
    try:
        df_idx = get_index_price()
        if df_idx is not None:
            # 合并指数数据
            merged = pd.merge(
                recent, 
                df_idx[df_idx['trade_date'].isin(recent['trade_date'])],
                on='trade_date',
                how='inner'
            )
            
            if len(merged) > 20:
                # 顶部背离：指数创新高，但累积广度线没有创新高
                idx_highest = merged['close'].iloc[-20:].max()
                current_idx = merged['close'].iloc[-1]
                cumdiff_highest = merged['cumulative_diff'].iloc[-20:].max()
                current_cumdiff = merged['cumulative_diff'].iloc[-1]
                
                # 判断逻辑：当前指数接近近期高点，但累积广度线明显低于近期高点
                if current_idx >= idx_highest * 0.98:
                    if current_cumdiff < cumdiff_highest * 0.95:
                        print("   🚨 检测到【顶部背离】信号！")
                        print("      指数接近近期高点，但新高-新低线明显下降")
                        print("      → 上涨动能衰竭，警惕回调风险")
                
                # 底部背离：指数创新低，但累积广度线没有创新低
                idx_lowest = merged['close'].iloc[-20:].min()
                cumdiff_lowest = merged['cumulative_diff'].iloc[-20:].min()
                
                if current_idx <= idx_lowest * 1.02:
                    if current_cumdiff > cumdiff_lowest * 1.05:
                        print("   🚨 检测到【底部背离】信号！")
                        print("      指数接近近期低点，但新高-新低线明显回升")
                        print("      → 下跌动能衰竭，关注反弹机会")
                    
    except Exception as e:
        print(f"   ⚠️ 背离检测失败: {e}")
    
    # ----- 基础信号判断 -----
    total_score = breadth_score['total_score']
    
    print(f"\n   🌡️ 技术广度综合得分: {total_score:.1f} / 100")
    
    if total_score < 20:
        print("   🧊 状态: 极度冰点（市场广度极差，绝大多数个股在下跌）")
        print("   💡 含义: 市场处于系统性恐慌中，历史大底区域的特征")
    elif total_score < 40:
        print("   🌧️ 状态: 低温磨底（广度较差，但已脱离极端区域）")
        print("   💡 含义: 多数个股仍在下行，但下跌动能已减弱")
    elif total_score < 60:
        print("   ⛅ 状态: 常温震荡（市场广度中性）")
        print("   💡 含义: 涨跌个股分布均衡，无明显趋势信号")
    elif total_score < 80:
        print("   ☀️ 状态: 高温过热（市场广度良好，多数个股在上涨）")
        print("   💡 含义: 市场处于健康上涨状态，但要警惕过热")
    else:
        print("   🔥 状态: 极度沸腾（市场广度极好，几乎所有个股都在涨）")
        print("   💡 含义: 市场极度亢奋，历史大顶区域的特征")
    
    print("=" * 70)


# ============================================================
# 第六步：可视化
# ============================================================

def plot_breadth(df_breadth, df_idx, breadth_score, config):
    """
    生成技术广度可视化图表
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # ----- 图1：归一化新高-新低线 vs 沪深300 -----
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    high_low_config = config['high_low']
    mcclellan_config = config['mcclellan']
    arms_config = config['arms']

    ax1.plot(
        df_breadth['trade_date'],
        df_breadth['cumulative_diff_norm'],
        color='blue',
        linewidth=1.5,
        label=f"新高-新低线({high_low_config['norm_window_days'] // 252}年分位归一化)"
    )
    ax1.axhline(50, color='gray', linestyle='--', linewidth=0.8)
    ax1.axhline(80, color='red', linestyle='--', linewidth=1, alpha=0.4)
    ax1.axhline(20, color='green', linestyle='--', linewidth=1, alpha=0.4)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('历史位置(0-100)', fontsize=11, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    if df_idx is not None:
        merged = pd.merge(df_breadth, df_idx, on='trade_date', how='inner')
        ax1_twin.plot(merged['trade_date'], merged['close'], 
                     color='red', linewidth=1.5, alpha=0.7, label='沪深300')
        ax1_twin.set_ylabel('沪深300指数', fontsize=11, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('归一化新高-新低线 vs 沪深300', fontsize=13)
    ax1.legend(loc='upper left')
    
    # ----- 图2：新高/新低占比 -----
    ax2 = axes[0, 1]
    ax2.fill_between(df_breadth['trade_date'], 0, df_breadth['pct_new_high'], 
                     color='green', alpha=0.5, label='创新高占比')
    ax2.fill_between(df_breadth['trade_date'], -df_breadth['pct_new_low'], 0, 
                     color='red', alpha=0.5, label='创新低占比')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('占比 (%)', fontsize=11)
    ax2.set_title('每日创新高/创新低个股占比', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ----- 图3：麦克莱伦震荡指标 -----
    ax3 = axes[1, 0]
    mcclellan_valid = df_breadth['mcclellan'].dropna()
    if len(mcclellan_valid) > 10:
        mcclellan_upper = mcclellan_valid.quantile(mcclellan_config['hot_quantile'])
        mcclellan_lower = mcclellan_valid.quantile(mcclellan_config['cold_quantile'])
    else:
        mcclellan_upper = 5
        mcclellan_lower = -5

    ax3.plot(df_breadth['trade_date'], df_breadth['mcclellan'], 
             color='purple', linewidth=1.5, label='麦克莱伦指标')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax3.axhline(mcclellan_upper, color='red', linestyle='--', linewidth=1, alpha=0.6, label=f'高热阈值({mcclellan_upper:.1f})')
    ax3.axhline(mcclellan_lower, color='green', linestyle='--', linewidth=1, alpha=0.6, label=f'低温阈值({mcclellan_lower:.1f})')
    ax3.fill_between(df_breadth['trade_date'], mcclellan_upper, df_breadth['mcclellan'], 
                     where=(df_breadth['mcclellan'] > mcclellan_upper), color='red', alpha=0.2)
    ax3.fill_between(df_breadth['trade_date'], mcclellan_lower, df_breadth['mcclellan'], 
                     where=(df_breadth['mcclellan'] < mcclellan_lower), color='green', alpha=0.2)
    ax3.set_ylabel('指标值', fontsize=11)
    ax3.set_title(
        f"麦克莱伦震荡指标 ({mcclellan_config['fast_ema_span']}EMA-{mcclellan_config['slow_ema_span']}EMA)",
        fontsize=13
    )
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # ----- 图4：阿姆氏指标（双坐标） -----
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    arms_display = df_breadth['arms_index_clipped']
    arms_ma = df_breadth['arms_ma']
    arms_valid = arms_display.dropna()
    if len(arms_valid) > 10:
        y_min = max(0, arms_valid.quantile(arms_config['display_y_low_quantile']) * 0.9)
        y_max = arms_valid.quantile(arms_config['display_y_high_quantile']) * 1.1
    else:
        y_min = 0
        y_max = 3

    arms_ma_valid = arms_ma.dropna()
    if len(arms_ma_valid) > 10:
        arms_hot = arms_ma_valid.quantile(arms_config['hot_quantile'])
        arms_cold = arms_ma_valid.quantile(arms_config['cold_quantile'])
        ma_y_min = max(0, min(arms_hot, arms_cold, arms_config['neutral_value']) * 0.9)
        ma_y_max = max(arms_hot, arms_cold, arms_config['neutral_value']) * 1.1
    else:
        arms_hot = 0.8
        arms_cold = 1.2
        ma_y_min = 0.6
        ma_y_max = 1.4

    raw_line, = ax4.plot(
        df_breadth['trade_date'],
        arms_display,
        color='orange',
        linewidth=0.8,
        alpha=0.45,
        label='去极值后原始值'
    )
    ma_line, = ax4_twin.plot(
        df_breadth['trade_date'],
        arms_ma,
        color='darkorange',
        linewidth=2,
        label=f"{arms_config['ma_window']}日均线"
    )
    neutral_line = ax4_twin.axhline(
        arms_config['neutral_value'],
        color='gray',
        linestyle='--',
        linewidth=0.8,
        label=f"中性({arms_config['neutral_value']:.2f})"
    )
    cold_line = ax4_twin.axhline(
        arms_cold,
        color='green',
        linestyle='--',
        linewidth=1,
        label=f"偏冷阈值({arms_cold:.2f})"
    )
    hot_line = ax4_twin.axhline(
        arms_hot,
        color='red',
        linestyle='--',
        linewidth=1,
        label=f"偏热阈值({arms_hot:.2f})"
    )

    ax4.set_ylim(y_min, y_max)
    ax4_twin.set_ylim(ma_y_min, ma_y_max)
    ax4.set_ylabel('ARMS原始值', fontsize=11, color='orange')
    ax4_twin.set_ylabel('ARMS均线与阈值', fontsize=11, color='darkorange')
    ax4.tick_params(axis='y', labelcolor='orange')
    ax4_twin.tick_params(axis='y', labelcolor='darkorange')
    ax4.set_title('阿姆氏指标（双坐标：原始值 vs 均线）', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(
        [raw_line, ma_line, neutral_line, cold_line, hot_line],
        [
            raw_line.get_label(),
            ma_line.get_label(),
            neutral_line.get_label(),
            cold_line.get_label(),
            hot_line.get_label()
        ],
        loc='upper left'
    )
    
    # ----- 图5：三个指标得分雷达图 -----
    ax5 = axes[2, 0]
    
    scores = [
        breadth_score['score_cum_diff'] / 100,
        breadth_score['score_mcclellan'] / 100,
        breadth_score['score_arms'] / 100
    ]
    
    categories = ['新高-新低线', '麦克莱伦', '阿姆氏']
    
    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]  # 闭合
    angles_plot = angles + [angles[0]]
    
    ax5.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='blue')
    ax5.fill(angles_plot, scores_plot, alpha=0.25, color='blue')
    ax5.set_xticks(angles)
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_title('技术广度子指标得分雷达图', fontsize=13)
    
    # ----- 图6：综合得分仪表盘 -----
    ax6 = axes[2, 1]
    total_score = breadth_score['total_score']
    
    theta = np.linspace(np.pi, 0, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax6.plot(x, y, color='lightgray', linewidth=15, solid_capstyle='round')
    color_map = plt.cm.RdYlGn_r
    color = color_map(total_score / 100)
    ax6.plot(x, y, color=color, linewidth=12, solid_capstyle='round')
    
    angle = np.pi * (1 - total_score / 100)
    ax6.arrow(0, 0, 0.8 * np.cos(angle), 0.8 * np.sin(angle), 
              head_width=0.08, head_length=0.08, fc='black', ec='black')
    
    ax6.text(0, -0.2, f'{total_score:.1f}°C', 
             ha='center', va='center', fontsize=20)
    ax6.text(0, -0.4, '技术广度维度', ha='center', va='center', fontsize=12)
    ax6.text(-1.2, -0.1, '🧊 冰点', fontsize=10, ha='center')
    ax6.text(1.2, -0.1, '🔥 沸腾', fontsize=10, ha='center')
    ax6.text(0, 1.1, '⛅ 中性', fontsize=10, ha='center')
    
    ax6.set_xlim(-1.3, 1.3)
    ax6.set_ylim(-0.5, 1.2)
    ax6.set_aspect('equal')
    ax6.axis('off')
    ax6.set_title('技术广度综合得分仪表盘', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(build_output_path('技术广度分析图.png'), dpi=300, bbox_inches='tight')
    
    print("\n✅ 图表已保存为: 技术广度分析图.png")


def parse_args():
    """
    解析命令行参数，支持自定义分析起止日期。
    """
    parser = argparse.ArgumentParser(description='技术广度维度市场温度计')
    parser.add_argument('--start', default='20180101', help='分析开始日期，格式为 YYYYMMDD')
    parser.add_argument('--end', default=None, help='分析结束日期，格式为 YYYYMMDD，默认使用当天')
    parser.add_argument('--config', default='config/tech_thermo_params.toml', help='参数文件路径，默认使用 config/tech_thermo_params.toml')
    args = parser.parse_args()

    for field_name in ['start', 'end']:
        value = getattr(args, field_name)
        if value is None:
            continue
        try:
            datetime.strptime(value, '%Y%m%d')
        except ValueError as exc:
            raise SystemExit(f'❌ 参数 --{field_name} 格式错误，应为 YYYYMMDD') from exc

    if args.end is not None and args.start > args.end:
        raise SystemExit('❌ 参数错误：--start 不能晚于 --end')

    return args


def main(start_date='20180101', end_date=None, config_path=None):
    """
    执行技术广度维度分析主流程。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    config = load_parameter_config(config_path)

    print("\n" + "=" * 70)
    print("🌡️ 技术广度维度指标研究（五维框架 - 权重15%）")
    print("   核心指标：新高-新低线 + 麦克莱伦 + 阿姆氏")
    print("=" * 70)

    df_daily = get_all_stock_daily(start_date, end_date)

    if df_daily is None or len(df_daily) == 0:
        print("\n❌ 数据获取失败")
        raise SystemExit("程序终止")

    df_breadth = compute_breadth_indicators(df_daily, config)

    if df_breadth is None or len(df_breadth) == 0:
        print("\n❌ 计算广度指标失败")
        raise SystemExit("程序终止")

    df_idx = get_index_price(start_date, end_date)
    breadth_score = compute_breadth_score(df_breadth, config)
    judge_breadth_signal(df_breadth, breadth_score)
    plot_breadth(df_breadth, df_idx, breadth_score, config)

    print("\n" + "=" * 70)
    print("✅ 技术广度维度分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    cli_args = parse_args()
    main(cli_args.start, cli_args.end, cli_args.config)

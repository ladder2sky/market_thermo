# ============================================================
# 情绪维度指标研究（五维框架 - 权重25%）
# 核心指标：成交额萎缩率 + 月度全市场换手率 + 融资买入额占成交额
# 数据源：Tushare Pro + AkShare
# ============================================================

import os
from pathlib import Path
import sys
import warnings
from datetime import datetime

import akshare as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


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
        env_path = Path(__file__).resolve().parent.parent / '.env'

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


def build_output_path(filename):
    """
    构造输出文件路径，确保图表和 CSV 统一写到 outputs 目录。
    """
    output_dir = Path(__file__).resolve().parent.parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename

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
# 第二步：获取情绪维度核心数据
# ============================================================

def get_market_turnover(start_date='20100101', end_date=None):
    """
    指标1：两市成交额（日度）
    极寒信号：成交额较前期高点萎缩 > 60%
    极热信号：成交额持续放大创新高
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"\n📊 [指标1] 获取两市成交额（日度）: {start_date} ~ {end_date}")
    
    try:
        # Tushare 指数日线的 amount 单位为千元，这里合并上证指数和深证成指作为两市成交额代理。
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

        if df_sh is not None and len(df_sh) > 0 and df_sz is not None and len(df_sz) > 0:
            df_sh['trade_date'] = pd.to_datetime(df_sh['trade_date'])
            df_sz['trade_date'] = pd.to_datetime(df_sz['trade_date'])
            df = pd.merge(
                df_sh.rename(columns={'amount': 'amount_sh'}),
                df_sz.rename(columns={'amount': 'amount_sz'}),
                on='trade_date',
                how='inner'
            )
            df = df.sort_values('trade_date').reset_index(drop=True)
            df['turnover_thousand_yuan'] = df['amount_sh'] + df['amount_sz']
            df['turnover_billion'] = df['turnover_thousand_yuan'] / 100000
            print(f"   ✅ 获取到 {len(df)} 条两市成交额数据")
            return df[['trade_date', 'turnover_thousand_yuan', 'turnover_billion']]

    except Exception as e:
        print(f"   ⚠️ Tushare获取失败: {e}")

    return None


def get_market_turnover_rate_monthly():
    """
    指标2：月度全市场换手率
    使用 AKShare 宏观数据构造：(两市月成交额 / 两市月总市值) * 100
    """
    try:
        print(f"\n📊 [指标2] 获取月度全市场换手率...")
        df = ak.macro_china_stock_market_cap()
        if df is None or len(df) == 0:
            print("   ❌ 未获取到市场总市值数据")
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
        numeric_cols = ['market_cap_sh', 'market_cap_sz', 'turnover_sh', 'turnover_sz']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['trade_date', 'market_cap_sh', 'market_cap_sz', 'turnover_sh', 'turnover_sz'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['market_cap_total'] = df['market_cap_sh'] + df['market_cap_sz']
        df['turnover_total'] = df['turnover_sh'] + df['turnover_sz']
        df['market_turnover_rate'] = df['turnover_total'] / df['market_cap_total'] * 100
        print(f"   ✅ 获取到 {len(df)} 条月度换手率数据")
        return df[['trade_date', 'market_turnover_rate', 'turnover_total', 'market_cap_total']]

    except Exception as e:
        print(f"   ❌ 获取月度换手率失败: {e}")
        return None


def get_margin_buy_ratio(df_turnover, start_date='20100101', end_date=None):
    """
    指标3：融资买入额占两市成交额比例（日度）
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"\n📊 [指标3] 获取融资买入额占成交额比例: {start_date} ~ {end_date}")
    
    try:
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
        print(f"   ✅ 获取到 {len(df)} 条融资买入占成交额数据")
        return df[['trade_date', 'margin_buy_yuan', 'margin_buy_ratio', 'margin_buy_ratio_ma20']]

    except Exception as e:
        print(f"   ❌ AkShare获取失败: {e}")
    
    return None


# ============================================================
# 第三步：计算情绪维度子指标
# ============================================================

def calculate_emotion_indicators(df_turnover, df_turnover_rate, df_margin):
    """
    将原始数据转化为标准化的情绪信号
    """
    
    print("\n" + "=" * 70)
    print("🔄 计算情绪指标信号...")
    print("=" * 70)
    
    # ---------- 指标1：成交额萎缩率 ----------
    df_t = df_turnover.copy()
    df_t['high_1y'] = df_t['turnover_billion'].rolling(window=252, min_periods=100).max()
    df_t['turnover_decline_rate'] = (df_t['turnover_billion'] / df_t['high_1y'] - 1) * 100
    df_t['turnover_decline_rate'] = df_t['turnover_decline_rate'].clip(-100, 0)
    df_t['high_3y'] = df_t['turnover_billion'].rolling(window=252 * 3, min_periods=252).max()
    df_t['turnover_decline_rate_3y'] = (df_t['turnover_billion'] / df_t['high_3y'] - 1) * 100
    df_t['turnover_decline_rate_3y'] = df_t['turnover_decline_rate_3y'].clip(-100, 0)
    
    # ---------- 指标2：月度全市场换手率 ----------
    df_r = None
    if df_turnover_rate is not None:
        df_r = df_turnover_rate.copy()
        df_r['market_turnover_rate_ma3'] = df_r['market_turnover_rate'].rolling(window=3, min_periods=2).mean()
        df_r['market_turnover_rate_percentile'] = rolling_percentile(df_r['market_turnover_rate'], window=36, min_periods=12)
    
    # ---------- 指标3：融资买入额占成交额 ----------
    df_m = df_margin.copy()
    df_m['margin_buy_ratio_percentile'] = rolling_percentile(df_m['margin_buy_ratio'], window=252 * 3, min_periods=252)
    
    return df_t, df_r, df_m


def rolling_percentile(series, window, min_periods):
    """
    计算滚动窗口内当前位置的百分位，输出范围 0~100。
    """
    def _last_percentile(values):
        s = pd.Series(values)
        return s.rank(pct=True).iloc[-1] * 100
    return series.rolling(window=window, min_periods=min_periods).apply(_last_percentile, raw=False)


def compute_emotion_score(df_t, df_r, df_m, current_date=None):
    """
    计算综合情绪得分（0-100分）
    成交额萎缩率权重 60%，融资买入额占成交额权重 40%。
    """
    if current_date is None:
        current_date = df_t['trade_date'].iloc[-1]
    
    # 获取最新数据
    latest_t = df_t[df_t['trade_date'] == current_date]
    if len(latest_t) == 0:
        latest_t = df_t.iloc[-1:]
    
    # 1. 成交额萎缩率 -> 情绪得分 (0-100)
    decline_rate = latest_t['turnover_decline_rate'].iloc[-1]
    decline_rate_3y = latest_t['turnover_decline_rate_3y'].iloc[-1]
    score_turnover = decline_rate + 100
    score_turnover = max(0, min(100, score_turnover))

    # 2. 融资买入额占成交额 -> 情绪得分
    latest_m = df_m[df_m['trade_date'] == current_date]
    if len(latest_m) == 0:
        latest_m = df_m.iloc[-1:]
    margin_buy_ratio = latest_m['margin_buy_ratio'].iloc[-1] if len(latest_m) > 0 else np.nan
    margin_percentile = latest_m['margin_buy_ratio_percentile'].iloc[-1] if len(latest_m) > 0 else np.nan
    if pd.isna(margin_percentile):
        margin_percentile = df_m['margin_buy_ratio'].rank(pct=True).iloc[-1] * 100
    score_margin = max(0, min(100, margin_percentile))

    latest_r = None
    if df_r is not None:
        latest_r = df_r[df_r['trade_date'] <= current_date]
        if len(latest_r) > 0:
            latest_r = latest_r.iloc[-1:]

    market_turnover_rate = latest_r['market_turnover_rate'].iloc[-1] if latest_r is not None and len(latest_r) > 0 else np.nan

    total_score = score_turnover * 0.6 + score_margin * 0.4
    
    return {
        'total_score': total_score,
        'score_turnover': score_turnover,
        'score_margin': score_margin,
        'decline_rate': decline_rate,
        'decline_rate_3y': decline_rate_3y,
        'market_turnover_rate': market_turnover_rate,
        'margin_buy_ratio': margin_buy_ratio,
        'margin_percentile': margin_percentile,
        'current_date': current_date
    }


def export_emotion_data_csv(df_t, df_r, df_m, output_path='情绪维度每日数据.csv'):
    """
    导出情绪维度分析的日度 CSV 数据，月度换手率按日期向前对齐。
    """
    export_df = pd.merge(
        df_t[
            [
                'trade_date',
                'turnover_billion',
                'turnover_decline_rate',
                'turnover_decline_rate_3y',
            ]
        ],
        df_m[
            [
                'trade_date',
                'margin_buy_yuan',
                'margin_buy_ratio',
                'margin_buy_ratio_ma20',
                'margin_buy_ratio_percentile',
            ]
        ],
        on='trade_date',
        how='inner'
    )

    if df_r is not None and len(df_r) > 0:
        export_df = pd.merge_asof(
            export_df.sort_values('trade_date'),
            df_r[
                [
                    'trade_date',
                    'market_turnover_rate',
                    'market_turnover_rate_ma3',
                    'market_turnover_rate_percentile',
                ]
            ].sort_values('trade_date'),
            on='trade_date',
            direction='backward'
        )

    export_df = export_df.rename(
        columns={
            'trade_date': '日期',
            'turnover_billion': '两市成交额_亿元',
            'turnover_decline_rate': '成交额萎缩率_1年高点口径',
            'turnover_decline_rate_3y': '成交额萎缩率_3年高点口径',
            'margin_buy_yuan': '融资买入额_元',
            'margin_buy_ratio': '融资买入额占成交额_百分比',
            'margin_buy_ratio_ma20': '融资买入额占成交额_20日均线',
            'margin_buy_ratio_percentile': '融资买入额占成交额_历史分位',
            'market_turnover_rate': '月度全市场换手率_百分比',
            'market_turnover_rate_ma3': '月度全市场换手率_3月均线',
            'market_turnover_rate_percentile': '月度全市场换手率_历史分位',
        }
    )
    export_df['日期'] = export_df['日期'].dt.strftime('%Y-%m-%d')
    export_df.to_csv(build_output_path(output_path), index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV已保存为: {output_path}")
    return export_df


# ============================================================
# 第四步：主程序执行
# ============================================================

print("\n" + "=" * 70)
print("🌡️ 情绪维度指标研究（五维框架 - 权重25%）")
print("   ✅ 核心指标: 成交额萎缩率 + 月度全市场换手率 + 融资买入额占成交额")
print("=" * 70)

START_DATE = '20150101'
END_DATE = datetime.now().strftime('%Y%m%d')

# 1. 获取原始数据
df_turnover = get_market_turnover(START_DATE, END_DATE)
df_turnover_rate = get_market_turnover_rate_monthly()

if df_turnover is None:
    raise SystemExit("❌ 获取成交额数据失败，程序终止")
if df_turnover_rate is None:
    raise SystemExit("❌ 获取月度换手率数据失败，程序终止")
df_margin = get_margin_buy_ratio(df_turnover, START_DATE, END_DATE)
if df_margin is None:
    raise SystemExit("❌ 获取融资买入额占成交额数据失败，程序终止")

# 2. 计算情绪指标
df_t, df_r, df_m = calculate_emotion_indicators(df_turnover, df_turnover_rate, df_margin)

# 3. 计算当前情绪得分
emotion_result = compute_emotion_score(df_t, df_r, df_m)

print("\n" + "=" * 70)
print("📊 当前情绪维度分析结果")
print("=" * 70)
print(f"  评估日期:     {emotion_result['current_date'].strftime('%Y-%m-%d')}")
print("-" * 70)
print(f"  📈 成交额萎缩率:        {emotion_result['decline_rate']:.1f}%")
print(f"  📈 3年成交额萎缩率:     {emotion_result['decline_rate_3y']:.1f}%")
print(f"  📈 月度全市场换手率:    {emotion_result['market_turnover_rate']:.2f}%")
print(f"  📈 融资买入额占成交额:  {emotion_result['margin_buy_ratio']:.2f}%")
print("-" * 70)
print(f"  🔥 成交额情绪得分:      {emotion_result['score_turnover']:.1f} / 100")
print(f"  🔥 两融情绪得分:        {emotion_result['score_margin']:.1f} / 100")
print("-" * 70)
print(f"  🌡️ 综合情绪得分:        {emotion_result['total_score']:.1f} / 100")

score = emotion_result['total_score']
if score < 20:
    state = "❄️ 极度冰点（极寒底部区域）"
elif score < 40:
    state = "🌧️ 低温磨底（熊市中后期）"
elif score < 60:
    state = "⛅ 常温震荡（均衡市）"
elif score < 80:
    state = "☀️ 高温过热（牛市后期）"
else:
    state = "💀 极度沸腾（历史大顶区域）"

print("-" * 70)
print(f"  🏷️ 市场状态: {state}")
print("=" * 70)

export_emotion_data_csv(df_t, df_r, df_m)


# ============================================================
# 第五步：生成可视化图表
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(15, 14))

# ---- 图1：月度全市场换手率 ----
ax1 = axes[0, 0]
ax1.plot(df_r['trade_date'], df_r['market_turnover_rate'], color='blue', linewidth=1.5, label='月度换手率')
ax1.plot(df_r['trade_date'], df_r['market_turnover_rate_ma3'], color='navy', linewidth=2, label='3月均线')
ax1.set_ylabel('换手率 (%)', fontsize=11)
ax1.set_title('月度全市场换手率', fontsize=13)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# ---- 图2：成交额萎缩率 ----
ax2 = axes[0, 1]
ax2.plot(df_t['trade_date'], df_t['turnover_decline_rate'], 
         color='red', linewidth=1.5, label='1年高点口径')
ax2.plot(df_t['trade_date'], df_t['turnover_decline_rate_3y'], 
         color='darkred', linewidth=1.0, linestyle='--', label='3年高点口径')
ax2.axhline(-60, color='green', linestyle='--', linewidth=1.5, 
            label='极寒阈值 (-60%)')
ax2.axhline(0, color='black', linestyle=':', linewidth=0.8)
ax2.fill_between(df_t['trade_date'], -100, df_t['turnover_decline_rate'], 
                  where=(df_t['turnover_decline_rate'] < -60), 
                  color='green', alpha=0.3, label='极寒区域')
ax2.set_ylabel('萎缩率 (%)', fontsize=11)
ax2.set_title('成交额萎缩率（相对过去1年高点）', fontsize=13)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# ---- 图3：融资买入额占成交额 ----
ax3 = axes[1, 0]
ax3.plot(df_m['trade_date'], df_m['margin_buy_ratio'], color='orange', linewidth=1.0, alpha=0.6, label='日度值')
ax3.plot(df_m['trade_date'], df_m['margin_buy_ratio_ma20'], color='darkorange', linewidth=2, label='20日均线')
ax3.set_ylabel('占成交额比例 (%)', fontsize=11)
ax3.set_title('融资买入额占两市成交额', fontsize=13)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# ---- 图4：融资买入额占成交额分位 ----
ax4 = axes[1, 1]
ax4.plot(df_m['trade_date'], df_m['margin_buy_ratio_percentile'], 
         color='green', linewidth=1.5)
ax4.fill_between(df_m['trade_date'], 0, df_m['margin_buy_ratio_percentile'], 
                 alpha=0.3, color='green')
ax4.axhline(20, color='green', linestyle='--', linewidth=1, label='低热度阈值')
ax4.axhline(80, color='red', linestyle='--', linewidth=1, label='高热度阈值')
ax4.set_ylabel('历史分位 (0-100)', fontsize=11)
ax4.set_title('融资买入额占成交额分位', fontsize=13)
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

# ---- 图5：核心情绪得分对比 ----
ax5 = axes[2, 0]
labels = ['成交额萎缩率', '融资买入占比']
scores = [emotion_result['score_turnover'], 
          emotion_result['score_margin']]
colors_bar = ['blue', 'green']
bars = ax5.bar(labels, scores, color=colors_bar, alpha=0.7)
ax5.axhline(20, color='green', linestyle='--', linewidth=1, label='极寒阈值 (20分)')
ax5.axhline(80, color='red', linestyle='--', linewidth=1, label='极热阈值 (80分)')
ax5.set_ylim(0, 100)
ax5.set_ylabel('情绪得分', fontsize=11)
ax5.set_title('各维度情绪得分对比', fontsize=13)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{score:.1f}', ha='center', va='bottom', fontsize=10)

# ---- 图6：综合情绪得分仪表盘 ----
ax6 = axes[2, 1]
total_score = emotion_result['total_score']

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
ax6.text(0, -0.4, '市场情绪', ha='center', va='center', fontsize=12)
ax6.text(-1.2, -0.1, '❄️ 冰点', fontsize=10, ha='center')
ax6.text(1.2, -0.1, '💀 沸腾', fontsize=10, ha='center')
ax6.text(0, 1.1, '⛅ 中性', fontsize=10, ha='center')

ax6.set_xlim(-1.3, 1.3)
ax6.set_ylim(-0.5, 1.2)
ax6.set_aspect('equal')
ax6.axis('off')
ax6.set_title('综合情绪得分仪表盘', fontsize=13)

plt.tight_layout()
plt.savefig(build_output_path('情绪维度分析图_真实数据.png'), dpi=300, bbox_inches='tight')

print("\n✅ 图表已保存为: 情绪维度分析图_真实数据.png")
print("=" * 70)

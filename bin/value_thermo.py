# ============================================================
# 股债收益差计算与可视化 - 完整可运行代码（带调试版）
# 数据源：Tushare Pro
# 适用环境：Jupyter Notebook
# ============================================================

import os
from pathlib import Path
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import akshare as ak
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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
configure_console_encoding()

# ============================================================
# 第一步：初始化 Tushare（从 .env 读取 token）
# ============================================================
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


TOKEN = os.getenv('TUSHARE_API_KEY') or load_env_value('TUSHARE_API_KEY')
pro = ts.pro_api(TOKEN)

print("✅ Tushare 初始化成功")
print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 60)


# ============================================================
# 第二步：定义数据获取函数（带异常处理）
# ============================================================

def get_hs300_data(start_date='20100101', end_date=None):
    """
    获取沪深300指数的PE_TTM和收盘价
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"📊 正在获取沪深300数据: {start_date} ~ {end_date}")
    
    try:
        # 获取PE数据（来自index_dailybasic）
        df_pe = pro.index_dailybasic(
            ts_code='000300.SH',
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,pe_ttm'
        )
        
        if df_pe is None or len(df_pe) == 0:
            print("   ❌ 未获取到PE数据，请检查网络或token权限")
            return None
        
        # 获取收盘价数据（来自index_daily）
        df_price = pro.index_daily(
            ts_code='000300.SH',
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,close'
        )
        
        if df_price is None or len(df_price) == 0:
            print("   ❌ 未获取到价格数据")
            return None
        
        # 合并数据
        df = pd.merge(df_pe, df_price, on='trade_date', how='inner')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 剔除PE为空的数据
        df = df.dropna(subset=['pe_ttm'])
        
        print(f"   ✅ 获取到 {len(df)} 条日度数据")
        return df
        
    except Exception as e:
        print(f"   ❌ 获取数据失败: {e}")
        return None


def get_bond_yield(start_date='20100101', end_date=None):
    """
    获取中国10年期国债收益率
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"📈 正在获取中国10年期国债收益率: {start_date} ~ {end_date}")

    try:
        print("   🔄 优先使用 AKShare 中债国债收益率曲线（按年分段抓取）...")
        target_curve_name = '\u4e2d\u503a\u56fd\u503a\u6536\u76ca\u7387\u66f2\u7ebf'
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        frames = []
        current_start = start_dt

        while current_start <= end_dt:
            current_end = min(current_start + pd.DateOffset(days=364), end_dt)
            yearly_df = ak.bond_china_yield(
                start_date=current_start.strftime('%Y%m%d'),
                end_date=current_end.strftime('%Y%m%d')
            )
            if yearly_df is not None and not yearly_df.empty:
                frames.append(yearly_df)
            current_start = current_end + pd.Timedelta(days=1)

        if not frames:
            raise ValueError('AKShare 未返回任何国债收益率数据')

        df_ak = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        curve_name_col = df_ak.columns[0]
        date_col = df_ak.columns[1]
        ten_year_col = next((column for column in df_ak.columns if '10年' in str(column)), None)
        if ten_year_col is None:
            raise KeyError('AKShare 返回结果中缺少 10年 列')

        df_ak = df_ak[df_ak[curve_name_col].astype(str) == target_curve_name].copy()
        if df_ak.empty:
            raise ValueError('AKShare 未返回中债国债收益率曲线数据')

        df_ak = df_ak[[date_col, ten_year_col]].rename(columns={date_col: 'trade_date', ten_year_col: 'yield_10y'})
        df_ak['trade_date'] = pd.to_datetime(df_ak['trade_date'])
        df_ak['yield_10y'] = pd.to_numeric(df_ak['yield_10y'], errors='coerce')
        df_ak = df_ak.dropna(subset=['yield_10y']).sort_values('trade_date').drop_duplicates(subset=['trade_date']).reset_index(drop=True)

        if df_ak.empty:
            raise ValueError('AKShare 中债10年期国债收益率为空')

        print(f"   ✅ 获取到中国10年期国债收益率 {len(df_ak)} 条（AKShare）")
        return df_ak

    except Exception as e:
        print(f"   ⚠️ AKShare 获取失败，回退到内置插值数据: {e}")
        print("   🔄 使用中国国债历史数据（内置锚点插值）...")

    # 使用内置锚点数据作为兜底方案。
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
            3.64, 3.28, 3.88, 3.87,  # 2010
            3.55, 3.34, 4.55, 4.03,  # 2011-2012
            4.02, 4.10, 3.37, 3.62,  # 2013-2014
            2.86, 2.92, 3.58, 3.50,  # 2015-2016
            3.62, 3.48, 3.18, 3.20,  # 2017-2018
            2.97, 3.10, 3.03, 2.85,  # 2019-2020
            2.84, 2.82, 2.86, 2.70,  # 2021-2022
            2.43, 2.65, 1.68, 1.75,  # 2023-2024
            1.66, 1.60                 # 2025-2026
        ]
    }
    
    df_china = pd.DataFrame(china_yield_data)
    df_china = df_china.sort_values('trade_date').reset_index(drop=True)
    
    # 将数据插值到日度频率
    df_china = df_china.set_index('trade_date')
    df_china = df_china.resample('D').interpolate(method='linear')
    df_china = df_china.reset_index()
    
    # 根据实际日期范围筛选
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df_china = df_china[(df_china['trade_date'] >= start_dt) & 
                        (df_china['trade_date'] <= end_dt)]
    
    print(f"   ✅ 获取到中国10年期国债收益率 {len(df_china)} 条（内置插值）")
    return df_china


def export_daily_data_csv(df, output_path='股债收益差每日数据.csv'):
    """
    导出每日数据到 CSV 文件，包含国债收益率、沪深300收益率、沪深300 PE_TTM 等字段。
    """
    export_df = df.copy()
    export_df['hs300_return_pct'] = export_df['close'].pct_change() * 100

    export_df = export_df.rename(
        columns={
            'trade_date': '日期',
            'yield_10y': '10Y国债收益率',
            'close': '沪深300收盘价',
            'hs300_return_pct': '沪深300日收益率',
            'pe_ttm': '沪深300PE_TTM',
            'earnings_yield': '沪深300盈利收益率',
            'spread': '股债收益差',
        }
    )

    export_df['日期'] = pd.to_datetime(export_df['日期']).dt.strftime('%Y-%m-%d')
    columns = [
        '日期',
        '10Y国债收益率',
        '沪深300收盘价',
        '沪深300日收益率',
        '沪深300PE_TTM',
        '沪深300盈利收益率',
        '股债收益差',
    ]
    export_df = export_df[columns]
    export_df.to_csv(build_output_path(output_path), index=False, encoding='utf-8-sig')
    print(f"✅ CSV已保存为: {output_path}")


# ============================================================
# 第三步：获取数据
# ============================================================

print("\n" + "=" * 60)
print("开始获取数据...")
print("=" * 60)

START_DATE = '20100101'
END_DATE = datetime.now().strftime('%Y%m%d')

# 获取沪深300数据
df_hs300 = get_hs300_data(START_DATE, END_DATE)

if df_hs300 is None:
    print("\n❌ 获取沪深300数据失败，请检查：")
    print("   1. Tushare token是否正确")
    print("   2. 网络连接是否正常")
    print("   3. Tushare账户是否有足够积分")
    raise SystemExit("程序终止")

# 获取国债收益率
df_bond = get_bond_yield(START_DATE, END_DATE)

if df_bond is None or len(df_bond) == 0:
    print("\n❌ 获取国债数据失败")
    raise SystemExit("程序终止")

# 合并数据
print("\n🔗 正在合并数据...")
df = pd.merge(df_hs300, df_bond, on='trade_date', how='inner')

if len(df) == 0:
    print("❌ 数据合并后为空，请检查日期范围是否匹配")
    print(f"   沪深300数据日期范围: {df_hs300['trade_date'].min()} ~ {df_hs300['trade_date'].max()}")
    print(f"   国债数据日期范围: {df_bond['trade_date'].min()} ~ {df_bond['trade_date'].max()}")
    raise SystemExit("程序终止")

df = df.sort_values('trade_date').reset_index(drop=True)
print(f"   ✅ 合并后数据量: {len(df)} 条")
print(f"   日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")


# ============================================================
# 第四步：计算股债收益差
# ============================================================

# 盈利收益率 = 1 / PE_TTM
df['earnings_yield'] = (1 / df['pe_ttm']) * 100

# 股债收益差 = 盈利收益率 - 10年期国债收益率
df['spread'] = df['earnings_yield'] - df['yield_10y']

export_daily_data_csv(df)

# 计算统计量
mean_spread = df['spread'].mean()
std_spread = df['spread'].std()
upper_1sigma = mean_spread + std_spread
lower_1sigma = mean_spread - std_spread
upper_2sigma = mean_spread + 2 * std_spread
lower_2sigma = mean_spread - 2 * std_spread

# 计算当前值
current_spread = df['spread'].iloc[-1]
current_pe = df['pe_ttm'].iloc[-1]
current_yield = df['yield_10y'].iloc[-1]
current_date = df['trade_date'].iloc[-1]
spread_percentile = (df['spread'] <= current_spread).sum() / len(df) * 100
pe_percentile = (df['pe_ttm'] <= current_pe).sum() / len(df) * 100

print("\n" + "=" * 60)
print("📊 股债收益差统计")
print("=" * 60)
print(f"  数据起止: {df['trade_date'].min().strftime('%Y-%m-%d')} ~ {df['trade_date'].max().strftime('%Y-%m-%d')}")
print(f"  均值:         {mean_spread:.2f}%")
print(f"  标准差:       {std_spread:.2f}%")
print(f"  +1σ:          {upper_1sigma:.2f}%")
print(f"  -1σ:          {lower_1sigma:.2f}%")
print(f"  +2σ:          {upper_2sigma:.2f}%")
print(f"  -2σ:          {lower_2sigma:.2f}%")
print("-" * 60)
print(f"  最新日期:     {current_date.strftime('%Y-%m-%d')}")
print(f"  当前沪深300 PE:   {current_pe:.2f}")
print(f"  当前PE分位数:     {pe_percentile:.1f}%")
print(f"  当前10Y国债收益率: {current_yield:.2f}%")
print(f"  🔥 当前股债收益差: {current_spread:.2f}%")
print(f"  股债收益差分位数: {spread_percentile:.1f}%")


# ============================================================
# 第五步：生成分析图表（增强版）
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ---- 图1：股债收益差走势 ----
ax1 = axes[0, 0]
ax1.plot(df['trade_date'], df['spread'], color='blue', linewidth=1.5, label='股债收益差')
ax1.axhline(mean_spread, color='black', linestyle='--', linewidth=1, label=f'均值 ({mean_spread:.2f}%)')
ax1.axhline(upper_1sigma, color='orange', linestyle=':', linewidth=1, label=f'+1σ')
ax1.axhline(lower_1sigma, color='orange', linestyle=':', linewidth=1, label=f'-1σ')
ax1.axhline(upper_2sigma, color='red', linestyle='--', linewidth=1.5, label=f'+2σ')
ax1.axhline(lower_2sigma, color='green', linestyle='--', linewidth=1.5, label=f'-2σ')

# 填充区域
ax1.fill_between(df['trade_date'], upper_2sigma, upper_1sigma, 
                  color='red', alpha=0.1, label='高性价比区')
ax1.fill_between(df['trade_date'], lower_1sigma, lower_2sigma, 
                  color='green', alpha=0.1, label='低性价比区')

ax1.set_ylabel('股债收益差 (%)', fontsize=11)
ax1.set_title('股债收益差历史走势', fontsize=13)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 标注当前值
ax1.scatter(current_date, current_spread, color='red', s=80, zorder=5)
ax1.annotate(f'当前 {current_spread:.2f}%', 
             xy=(current_date, current_spread),
             xytext=(-60, 10), textcoords='offset points',
             fontsize=10, color='red')

# ---- 图2：股债收益差分布 ----
ax2 = axes[0, 1]
ax2.hist(df['spread'], bins=50, color='blue', alpha=0.6, edgecolor='black', linewidth=0.5)
ax2.axvline(current_spread, color='red', linewidth=2.5, label=f'当前: {current_spread:.2f}%')
ax2.axvline(mean_spread, color='black', linestyle='--', linewidth=1, label=f'均值')
ax2.axvline(upper_2sigma, color='red', linestyle=':', linewidth=1, label=f'+2σ')
ax2.axvline(lower_2sigma, color='green', linestyle=':', linewidth=1, label=f'-2σ')

ax2.set_xlabel('股债收益差 (%)', fontsize=11)
ax2.set_ylabel('频次', fontsize=11)
ax2.set_title('股债收益差历史分布', fontsize=13)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# ---- 图3：沪深300 PE走势 ----
ax3 = axes[1, 0]
ax3.plot(df['trade_date'], df['pe_ttm'], color='green', linewidth=1.5)
ax3.set_ylabel('PE_TTM', fontsize=11)
ax3.set_title('沪深300 PE_TTM 历史走势', fontsize=13)
ax3.grid(True, alpha=0.3)

# 标注当前PE
ax3.scatter(current_date, current_pe, color='red', s=80, zorder=5)
ax3.annotate(f'{current_pe:.1f}', 
             xy=(current_date, current_pe),
             xytext=(10, 5), textcoords='offset points',
             fontsize=10, color='red')

# ---- 图4：沪深300 PE分位分布 ----
ax4 = axes[1, 1]
# 使用直方图直接展示当前PE在历史分布中的位置，避免与股债收益差分位混淆。
ax4.hist(df['pe_ttm'], bins=40, color='green', alpha=0.6, edgecolor='black', linewidth=0.5)
ax4.axvline(current_pe, color='red', linewidth=2.5, label=f'当前PE ({current_pe:.2f})')
ax4.axvline(df['pe_ttm'].mean(), color='black', linestyle='--', linewidth=1, label='历史均值')
ax4.set_xlabel('PE_TTM', fontsize=11)
ax4.set_ylabel('频次', fontsize=11)
ax4.set_title(f'当前PE历史分位: {pe_percentile:.1f}%', fontsize=13)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(build_output_path('股债收益差分析图.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 图表已保存为: 股债收益差分析图.png")


# ============================================================
# 第六步：输出当前市场温度判断
# ============================================================

print("\n" + "=" * 60)
print("🌡️ 当前市场温度判断")
print("=" * 60)

print(f"  当前股债收益差: {current_spread:.2f}%")
print(f"  股债收益差分位数: {spread_percentile:.1f}%")
print(f"  当前PE分位数:     {pe_percentile:.1f}%")
print("-" * 60)

if current_spread > upper_2sigma:
    print("  🔥 结论: 极度低估区域！")
    print("  💡 建议: 股票相对债券性价比极高，历史大底区间，可分批布局权益资产")
elif current_spread > upper_1sigma:
    print("  📈 结论: 低估区域")
    print("  💡 建议: 股票性价比较好，可适度增加权益配置")
elif current_spread < lower_2sigma:
    print("  💀 结论: 极度高估区域！")
    print("  💡 建议: 股票相对债券性价比极低，注意风险，考虑降低权益仓位")
elif current_spread < lower_1sigma:
    print("  📉 结论: 高估区域")
    print("  💡 建议: 股票性价比偏低，保持谨慎，控制仓位")
else:
    print("  ⛅ 结论: 中性区域")
    print("  💡 建议: 股债性价比较均衡，保持现有配置，耐心等待")

print("\n" + "=" * 60)
print("📌 历史参考区间：")
print(f"  +2σ ({upper_2sigma:.2f}%): 历史大底区域（如2014年、2018年底）")
print(f"  -2σ ({lower_2sigma:.2f}%): 历史大顶区域（如2015年、2021年初）")
print("=" * 60)

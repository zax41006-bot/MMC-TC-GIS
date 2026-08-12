import time
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import tcmarkers

# ==================== 1. 氣旋與 GitHub Pages 網址設定 ====================
TC_ID = "G"
TC_NAME = "浪卡"

# 你的 GitHub Pages 基礎網址
SITE_BASE_URL = "https://zax41006-bot.github.io/TC-Track"

PAST_CSV_URL = f"{SITE_BASE_URL}/past_track_{TC_ID}.csv"
FORE_CSV_URL = f"{SITE_BASE_URL}/forecast_track_{TC_ID}.csv"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_IMG = os.path.join(BASE_PATH, f"TC_forecast_{TC_ID}.png")

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

MACAO_LON, MACAO_LAT = 113.55, 22.17
LAT_TO_KM = 110.574
MACAO_ALERT_CIRCLES = [(100, "#A0A0A0", 0.3), (200, "#808080", 0.4), (400, "#808080", 0.5), (600, "#A0A0A0", 0.6), (800, "#FF4D4D", 0.7)]

def lon_to_km_factor(lat): 
    return 111.320 * np.cos(np.radians(lat))

def get_intensity_info(wind, cyc_type="tropical"):
    if cyc_type == "EX": return "溫帶氣旋", "#BDBDBD", tcmarkers.HU
    if wind < 41: return "低壓區", "#BDBDBD", tcmarkers.HU
    elif 41 <= wind <= 62: return "熱帶低氣壓", "#FFF176", tcmarkers.HU
    elif 63 <= wind <= 87: return "熱帶風暴", "#64B5F6", tcmarkers.HU
    elif 88 <= wind <= 117: return "強烈熱帶風暴", "#4CAF50", tcmarkers.HU
    elif 118 <= wind <= 149: return "颱風", "#FFB74D", tcmarkers.HU
    elif 150 <= wind <= 184: return "強颱風", "#FF7043", tcmarkers.HU
    else: return "超強颱風", "#BA68C8", tcmarkers.HU

def draw_chart():
    print(f"[{time.strftime('%H:%M:%S')}] 正在從 GitHub Pages 下載數據並生成預報圖 ({TC_NAME})...")
    
    try:
        # 從 GitHub Pages 讀取 CSV
        df_past = pd.read_csv(PAST_CSV_URL)
        df_fore = pd.read_csv(FORE_CSV_URL)
        
        past_data = df_past[['datetime', 'lng', 'lat', 'wind', 'minimum central pressure']].values.tolist()
        curr = past_data[-1]
        
        forecast_data = []
        for _, row in df_fore.iterrows():
            h = int(str(row['f_time']).replace('hr', ''))
            forecast_data.append([row['f_time'], row['lng'], row['lat'], row['wind'], h, row['minimum central pressure'], row.get('type', 'tropical')])

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # 根據過去與預報經緯度動態或固定邊界
        lon_min, lon_max, lat_min, lat_max = 101.0, 178.0, 1.0, 58.0
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#F5F5DC", edgecolor="#795548", linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="#E3F2FD", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#4E342E', zorder=2)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#757575', alpha=0.6, linestyle='--', zorder=1)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.MultipleLocator(5)
        gl.ylocator = mticker.MultipleLocator(5)

        for r_km, col, alph in MACAO_ALERT_CIRCLES:
            lat_r, lon_r = r_km / LAT_TO_KM, r_km / lon_to_km_factor(MACAO_LAT)
            ax.add_patch(Ellipse((MACAO_LON, MACAO_LAT), 2 * lon_r, 2 * lat_r, fc='none', ec=col, alpha=alph, lw=0.8, transform=ccrs.PlateCarree(), zorder=2))

            angle = np.radians(270)
            tx = MACAO_LON + lon_r * np.cos(angle)
            ty = MACAO_LAT + lat_r * np.sin(angle)

            if lat_min <= ty <= lat_max and lon_min <= tx <= lon_max:
                ax.text(tx, ty, f"{r_km}km", color=col, fontsize=8.5, fontweight='bold', alpha=alph + 0.3,
                        ha='center', va='top', transform=ccrs.PlateCarree(), zorder=2)

        # 澳門位置標記 (ms=7.0)
        ax.plot(MACAO_LON, MACAO_LAT, '*', color="#E64A19", ms=7.0, mec='#3E2723', mew=1.0, zorder=12)

        # 過去路徑線
        ax.plot([d[1] for d in past_data], [d[2] for d in past_data], color="#43A047", lw=2.2, zorder=4)
        
        # 預報誤差扇形/橢圓區域
        f_hs = [d[4] for d in forecast_data]
        all_h, all_ln, all_lt = [0] + f_hs, [curr[1]] + [d[1] for d in forecast_data], [curr[2]] + [d[2] for d in forecast_data]
        all_er = [0] + [((h // 24) * 100 + (h % 24) * (100 / 24)) * (1 / 111) for h in f_hs]
        ih = np.linspace(0, max(all_h), 100)
        xi, yi, ri = PchipInterpolator(all_h, all_ln)(ih), PchipInterpolator(all_h, all_lt)(ih), PchipInterpolator(all_h, all_er)(ih)
        ps = [Polygon(np.dstack((xi[i] + ri[i] * np.cos(np.linspace(0, 2 * np.pi, 360)), yi[i] + ri[i] * np.sin(np.linspace(0, 2 * np.pi, 360))))[0]) for i in range(len(ih))]
        ax.add_geometries([unary_union([MultiPolygon([ps[i], ps[i + 1]]).convex_hull for i in range(len(ps) - 1)])], 
                          ccrs.PlateCarree(), fc="#FFF5D7", alpha=0.45, ec="#FFD180", lw=0.7, zorder=3)
        ax.plot(xi, yi, color="#1976D2", lw=2.2, ls='--', zorder=4)

        # 預報點 ICON（縮小至 ms=6.0，12hr 節點 ms=4.0）
        for d in forecast_data:
            _, ln, lt, wd, h, _, cyc = d
            _, col, m = get_intensity_info(wd, cyc)
            if h in {24, 48, 72, 96, 120}:
                ax.plot(ln, lt, marker=m, ms=6.0, color=col, mec='k', mew=0.6, zorder=10)
            else:
                ax.plot(ln, lt, marker='x', ms=4.0, color="#1976D2", mew=0.8, zorder=9)

        # 現時位置 ICON（縮小至 ms=7.5）
        _, c_col, c_m = get_intensity_info(curr[3])
        ax.plot(curr[1], curr[2], marker=c_m, ms=7.5, color=c_col, mec='k', mew=0.8, zorder=10)

        # 標題與資訊欄
        fig.text(0.5, 0.94, f"熱帶氣旋 “{TC_NAME}” 路徑預報圖", ha='center', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.91, f"預報時效：{max(f_hs)} 小時", ha='center', fontsize=13, color='#424242')

        info_txt = f"現時位置資料\n時間：{curr[0]}\n強度：{get_intensity_info(curr[3])[0]}\n近中心最大風速：{curr[3]}kph  中心氣壓：{curr[4]}hPa\n現時位置：{curr[2]:.1f}°N, {curr[1]:.1f}°E"
        ax.text(0.03, 0.96, info_txt, transform=ax.transAxes, va='top', fontsize=9.0, fontweight='bold', linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8, ec="#8D6E63", lw=1.0), zorder=20)
        ax.text(0.98, 0.98, "澳門氣象中心MMC 發佈", transform=ax.transAxes, ha='right', va='top', fontsize=11.0, fontweight='bold',
                color='#3E2723', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"), zorder=20)

        # 圖例（對應縮小圖示標記）
        leg_core = [Line2D([0], [0], color="#43A047", lw=2.2, label='過去路徑'), 
                    Line2D([0], [0], color="#1976D2", lw=2.2, ls='--', label='預報路徑'), 
                    plt.Rectangle((0, 0), 1, 1, fc="#FFF5D7", alpha=0.45, ec="#FFD180", label='預報誤差範圍')]
        
        leg_int = [Line2D([0], [0], marker=tcmarkers.HU, c=get_intensity_info(v)[1], label=get_intensity_info(v)[0], ms=4.0, mec='k', mew=0.5, ls='') for v in [30, 50, 75, 100, 130, 160, 200]]
        
        leg_node = [Line2D([0], [0], marker=tcmarkers.HU, color="#1976D2", ms=4.5, mec='#333', mew=0.5, ls='', label='24小時預報節點'), 
                    Line2D([0], [0], marker='x', color="#1976D2", ms=3.5, mew=0.8, ls='', label='12小時預報節點')]

        leg_params = dict(loc='lower center', frameon=True, edgecolor='#8D6E63', facecolor='white', framealpha=0.85)

        fig.legend(handles=leg_core, ncol=3, bbox_to_anchor=(0.5, 0.13), fontsize=8.5, **leg_params)
        fig.legend(handles=leg_int, ncol=7, bbox_to_anchor=(0.5, 0.08), fontsize=8.0, **leg_params)
        fig.legend(handles=leg_node, ncol=2, bbox_to_anchor=(0.5, 0.04), fontsize=8.5, **leg_params)

        plt.subplots_adjust(bottom=0.2, top=0.88)
        plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[{time.strftime('%H:%M:%S')}] √ 預報圖生成成功 ({OUTPUT_IMG})")

    except Exception as e: 
        print(f"[{time.strftime('%H:%M:%S')}] × 讀取或繪圖失敗: {e}")

if __name__ == "__main__":
    draw_chart()
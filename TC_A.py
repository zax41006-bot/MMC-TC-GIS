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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tcmarkers

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PAST_CSV = os.path.join(BASE_PATH, "past_track_A.csv")
FORE_CSV = os.path.join(BASE_PATH, "forecast_track_A.csv")
OUTPUT_IMG = os.path.join(BASE_PATH, "TC_forecast_A.png")

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

MACAO_LON, MACAO_LAT = 113.55, 22.17
LAT_TO_KM = 110.574
MACAO_ALERT_CIRCLES = [(100, "#A0A0A0", 0.3), (200, "#808080", 0.4), (400, "#808080", 0.5), (600, "#A0A0A0", 0.6), (800, "#FF4D4D", 0.7)]

def lon_to_km_factor(lat): return 111.320 * np.cos(np.radians(lat))

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
    print(f"[{time.strftime('%H:%M:%S')}] 正在生成預報...")
    try:
        df_past = pd.read_csv(PAST_CSV)
        df_fore = pd.read_csv(FORE_CSV)
        past_data = df_past[['datetime', 'lng', 'lat', 'wind', 'minimum central pressure']].values.tolist()
        curr = past_data[-1]
        
        forecast_data = []
        for _, row in df_fore.iterrows():
            h = int(str(row['f_time']).replace('hr', ''))
            forecast_data.append([row['f_time'], row['lng'], row['lat'], row['wind'], h, row['minimum central pressure'], row.get('type', 'tropical')])

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([105.0, 140.0, 10.0, 35.0], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#F5F5DC", edgecolor="#795548", linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="#E3F2FD", zorder=0)

        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#4E342E', zorder=2)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#757575', alpha=0.6, linestyle='--', zorder=1)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.MultipleLocator(5)
        gl.ylocator = mticker.MultipleLocator(5)

        for r_km, col, alph in MACAO_ALERT_CIRCLES:
            lat_r, lon_r = r_km/LAT_TO_KM, r_km/lon_to_km_factor(MACAO_LAT)
            ax.add_patch(Ellipse((MACAO_LON, MACAO_LAT), 2*lon_r, 2*lat_r, fc='none', ec=col, alpha=alph, lw=0.8, transform=ccrs.PlateCarree(), zorder=2))

            angle = np.radians(270)
            tx = MACAO_LON + lon_r * np.cos(angle)
            ty = MACAO_LAT + lat_r * np.sin(angle)

            ax.text(tx, ty, f"{r_km}km", color=col, fontsize=8.5, fontweight='bold', alpha=alph+0.3,
                    ha='center', va='top', transform=ccrs.PlateCarree(), zorder=15)

        ax.plot(MACAO_LON, MACAO_LAT, '*', color="#E64A19", ms=10, mec='#3E2723', mew=1.2, zorder=12)

        ax.plot([d[1] for d in past_data], [d[2] for d in past_data], color="#43A047", lw=2.5, zorder=4)
        f_hs = [d[4] for d in forecast_data]
        all_h, all_ln, all_lt = [0]+f_hs, [curr[1]]+[d[1] for d in forecast_data], [curr[2]]+[d[2] for d in forecast_data]
        all_er = [0]+[((h//24)*100 + (h%24)*(100/24))*(1/111) for h in f_hs]
        ih = np.linspace(0, max(all_h), 100)
        xi, yi, ri = PchipInterpolator(all_h, all_ln)(ih), PchipInterpolator(all_h, all_lt)(ih), PchipInterpolator(all_h, all_er)(ih)
        ps = [Polygon(np.dstack((xi[i]+ri[i]*np.cos(np.linspace(0, 2*np.pi, 360)), yi[i]+ri[i]*np.sin(np.linspace(0, 2*np.pi, 360))))[0]) for i in range(len(ih))]
        ax.add_geometries([unary_union([MultiPolygon([ps[i], ps[i+1]]).convex_hull for i in range(len(ps)-1)])], 
                          ccrs.PlateCarree(), fc="#FFF5D7", alpha=0.45, ec="#FFD180", lw=0.7, zorder=3)
        ax.plot(xi, yi, color="#1976D2", lw=2.5, ls='--', zorder=4)

        for d in forecast_data:
            _, ln, lt, wd, h, _, cyc = d
            _, col, m = get_intensity_info(wd, cyc)
            if h in {24, 48, 72, 96, 120}:
                ax.plot(ln, lt, marker=m, ms=7.5, color=col, mec='k', mew=0.8, zorder=10)
            else:
                ax.plot(ln, lt, marker='x', ms=4.5, color="#1976D2", mew=1.0, zorder=9)

        _, c_col, c_m = get_intensity_info(curr[3])
        ax.plot(curr[1], curr[2], marker=c_m, ms=8.5, color=c_col, mec='k', mew=1.1, zorder=10)

        fig.text(0.5, 0.94, "菲律賓以東海域的熱帶低氣壓 路徑預報圖", ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.905, f"預報時效：{max(f_hs)} 小時", ha='center', fontsize=14, color='#424242')

        info_txt = f"現時位置資料\n時間：{curr[0]}\n強度：{get_intensity_info(curr[3])[0]}\n近中心最大風速：{curr[3]}kph  中心氣壓：{curr[4]}hPa\n現時位置：{curr[2]:.1f}°N, {curr[1]:.1f}°E"
        ax.text(0.03, 0.96, info_txt, transform=ax.transAxes, va='top', fontsize=9.5, fontweight='bold', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.75, ec="#8D6E63", lw=1.2), zorder=20)
        ax.text(0.98, 0.98, "澳門氣象中心MMC 發佈", transform=ax.transAxes, ha='right', va='top', fontsize=11.5, fontweight='bold',
                color='#3E2723', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"), zorder=20)

        leg_core = [Line2D([0],[0],color="#43A047",lw=2.5,label='過去路徑'), Line2D([0],[0],color="#1976D2",lw=2.5,ls='--',label='預報路徑'), plt.Rectangle((0,0),1,1,fc="#FFF5D7",alpha=0.45,ec="#FFD180",label='預報誤差範圍')]
        leg_int = [Line2D([0],[0],marker=tcmarkers.HU,c=get_intensity_info(v)[1],label=get_intensity_info(v)[0],ms=5.5,mec='k',ls='') for v in [30, 50, 75, 100, 130, 160, 200]]
        leg_node = [Line2D([0],[0],marker=tcmarkers.HU,color="#1976D2",ms=6,mec='#333',ls='',label='24小時預報節點'), Line2D([0],[0],marker='x',color="#1976D2",ms=5,mew=1.0,ls='',label='12小時預報節點')]

        leg_params = dict(loc='lower center', frameon=True, edgecolor='#8D6E63', facecolor='white', framealpha=0.8)

        fig.legend(handles=leg_core, ncol=3, bbox_to_anchor=(0.5, 0.13), fontsize=9, **leg_params)
        fig.legend(handles=leg_int, ncol=7, bbox_to_anchor=(0.5, 0.08), fontsize=8.5, **leg_params)
        fig.legend(handles=leg_node, ncol=2, bbox_to_anchor=(0.5, 0.04), fontsize=9, **leg_params)

        plt.subplots_adjust(bottom=0.2, top=0.88)
        plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[{time.strftime('%H:%M:%S')}] √ 預報圖生成成功")

    except Exception as e: 
        print(f"[{time.strftime('%H:%M:%S')}] × 出錯了: {e}")

class CSVHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".csv"): 
            time.sleep(0.5)
            draw_chart()

if __name__ == "__main__":
    draw_chart()
    obs = Observer()
    obs.schedule(CSVHandler(), BASE_PATH, recursive=False)
    obs.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: 
        obs.stop()
    obs.join()
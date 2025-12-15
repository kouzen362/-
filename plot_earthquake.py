# -*- coding: utf-8 -*-
"""
震源数据可视化脚本
使用科研绘图风格绘制日本及周边地区地震分布图

特点:
- 空心圆表示震源位置
- 圆圈大小表示震级（最大考虑8级）
- 颜色表示深度（最大考虑700km）
- 绘制日本及周边国家省级行政区划
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 读取震源数据
# =============================================================================
def read_earthquake_data(filepath):
    """
    读取震源数据文件，支持两种格式：
    
    格式1 (Hi-net格式):
    ------Origin Time--------OTerr----Lat---Yerr---Long---Xerr---Dep---Derr--Mag
    2025-12-14 00:04:28.160  0.180   41.0208  0.6  142.4567  0.9   13.3  1.5   2.2
    
    格式2 (JMA格式):
    -----Origin Time--------OTerr---Lat---LatErr---Long--LonErr---Dep--DepErr-Mag---------Region----Flag
    2025-12-13 00:00:06.64  0.39   34.733  1.30  141.130  1.06     9.1   3.8  1.9v        SE OFF...  A
    
    JMA Flag说明:
    K: 气象厅震源 (高精度)
    k: 简易气象厅震源 (较高精度)
    A: 自动气象厅震源 (较高精度)
    S: 参考震源 (低精度)
    s: 简易参考震源 (低精度)
    a: 自动参考震源 (低精度)
    N: 震源固定/不定/未计算
    F: 远地
    
    返回:
        (earthquakes, is_jma_format): 地震数据列表和是否为JMA格式的标志
    """
    earthquakes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检测格式：查看表头是否包含 "Region" 或 "Flag"
    header = lines[0] if lines else ""
    is_jma_format = "Region" in header or "Flag" in header
    
    for line in lines[1:]:  # 跳过标题行
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        parts = line_stripped.split()
        if len(parts) < 10:
            continue
            
        try:
            # 两种格式的前10列结构相同:
            # [0]: 日期 (2025-12-14)
            # [1]: 时间 (00:04:28.160)
            # [2]: OTerr
            # [3]: Lat
            # [4]: LatErr
            # [5]: Long
            # [6]: LonErr
            # [7]: Dep
            # [8]: DepErr (可能缺失)
            # [9]: Mag (新格式可能带v/V后缀)
            
            lat = float(parts[3])
            lon = float(parts[5])
            
            # 深度处理
            depth_str = parts[7]
            depth = float(depth_str)
            
            # 震级处理：可能带有字母后缀 (如 1.9v, 2.1V)
            mag_str = parts[9] if len(parts) > 9 else parts[8]
            
            # 移除可能的字母后缀
            mag_clean = ''.join(c for c in mag_str if c.isdigit() or c == '.' or c == '-')
            if mag_clean:
                mag = float(mag_clean)
            else:
                continue
            
            # 提取Flag (JMA格式最后一列)
            flag = None
            if is_jma_format and len(parts) >= 2:
                # Flag通常是最后一个字段，是单个字母
                last_part = parts[-1]
                if len(last_part) == 1 and last_part in 'KkSsAaNF':
                    flag = last_part
            
            earthquakes.append({
                'lat': lat,
                'lon': lon,
                'depth': depth,
                'mag': mag,
                'flag': flag  # None for Hi-net format, letter for JMA format
            })
        except (ValueError, IndexError):
            continue
    
    return earthquakes, is_jma_format


def filter_earthquakes(earthquakes, mag_min=None, mag_max=None, 
                       depth_min=None, depth_max=None, flags=None):
    """
    过滤地震数据
    
    参数:
        earthquakes: 地震数据列表
        mag_min: 最小震级 (包含)
        mag_max: 最大震级 (包含)
        depth_min: 最小深度 (包含)
        depth_max: 最大深度 (包含)
        flags: 允许的Flag列表，如 ['K', 'k', 'A']
    
    返回:
        过滤后的地震数据列表
    """
    filtered = []
    for eq in earthquakes:
        # 震级过滤
        if mag_min is not None and eq['mag'] < mag_min:
            continue
        if mag_max is not None and eq['mag'] > mag_max:
            continue
        
        # 深度过滤
        if depth_min is not None and eq['depth'] < depth_min:
            continue
        if depth_max is not None and eq['depth'] > depth_max:
            continue
        
        # Flag过滤 (仅对JMA格式有效)
        if flags is not None and eq.get('flag') is not None:
            if eq['flag'] not in flags:
                continue
        
        filtered.append(eq)
    
    return filtered

# =============================================================================
# 2. 设置科研绘图风格
# =============================================================================
def setup_science_style():
    """设置科研论文风格的绘图参数"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Microsoft YaHei'],
        'font.size': 10,
        
        # 坐标轴设置
        'axes.linewidth': 1.2,
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.facecolor': '#E8F4FA',  # 浅蓝色背景表示海洋
        
        # 刻度设置
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        
        # 图例设置
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.3',
        
        # 图形设置
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# =============================================================================
# 3. 创建深度颜色映射
# =============================================================================
def create_depth_colormap():
    """
    创建平滑过渡的深度颜色映射
    通过控制颜色点的位置，实现非均匀的线性插值
    """
    # 定义控制点 (深度值)
    # 0-10km (浅源): 红色 -> 橙色
    # 10-30km: 橙色 -> 黄色
    # 30-100km: 黄色 -> 绿色
    # 100-200km: 绿色 -> 青色
    # 200-700km (深源): 青色 -> 蓝色 -> 深蓝
    
    max_depth = 700.0
    
    # 定义 (位置, 颜色) 元组
    # 位置必须归一化到 0-1 之间
    colors_data = [
        (0/max_depth,   '#FF0000'),  # 0 km: 红色
        (10/max_depth,  '#FF5500'),  # 10 km: 红橙
        (30/max_depth,  '#FFAA00'),  # 30 km: 橙色
        (80/max_depth,  '#FFFF00'),  # 80 km: 黄色
        (150/max_depth, '#00FF00'),  # 150 km: 绿色
        (300/max_depth, '#00FFFF'),  # 300 km: 青色
        (500/max_depth, '#0000FF'),  # 500 km: 蓝色
        (700/max_depth, '#000080'),  # 700 km: 深蓝
    ]
    
    # 创建平滑过渡的colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('depth_smooth', colors_data, N=512)
    
    # 返回 colormap 和 标准norm (0-700)
    norm = mcolors.Normalize(vmin=0, vmax=max_depth)
    
    return cmap, norm

# =============================================================================
# 4. 计算震级对应的圆圈大小
# =============================================================================
def magnitude_to_size(mag, mag_min=0, mag_max=8, size_min=15, size_max=600):
    """
    将震级转换为圆圈大小
    使用指数关系使差异更明显
    """
    normalized = (mag - mag_min) / (mag_max - mag_min)
    normalized = np.clip(normalized, 0, 1)
    
    # 使用二次函数使大小差异更明显
    size = size_min + (size_max - size_min) * (normalized ** 2)
    return size

# =============================================================================
# 5. 下载并缓存Natural Earth数据
# =============================================================================
def get_natural_earth_data():
    """获取Natural Earth地理数据"""
    import os
    import urllib.request
    import zipfile
    
    cache_dir = os.path.join(os.path.dirname(__file__), 'ne_data')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 国家边界数据
    countries_file = os.path.join(cache_dir, 'ne_10m_admin_0_countries.shp')
    # 省级行政区划数据
    provinces_file = os.path.join(cache_dir, 'ne_10m_admin_1_states_provinces.shp')
    
    # 如果文件不存在，下载它们
    if not os.path.exists(countries_file):
        print("正在下载国家边界数据...")
        url = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        zip_path = os.path.join(cache_dir, 'countries.zip')
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(cache_dir)
            os.remove(zip_path)
        except Exception as e:
            print(f"下载失败: {e}")
            return None, None
    
    if not os.path.exists(provinces_file):
        print("正在下载省级行政区划数据...")
        url = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip'
        zip_path = os.path.join(cache_dir, 'provinces.zip')
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(cache_dir)
            os.remove(zip_path)
        except Exception as e:
            print(f"下载失败: {e}")
            return None, None
    
    try:
        countries = gpd.read_file(countries_file)
        provinces = gpd.read_file(provinces_file)
        return countries, provinces
    except Exception as e:
        print(f"读取shapefile失败: {e}")
        return None, None

# =============================================================================
# 6. 主绑图函数
# =============================================================================
def plot_earthquakes(earthquakes, output_path='earthquake_map.png', 
                     map_extent=None, map_name='Japan Region', filter_info=None):
    """
    绑制地震分布图
    
    参数:
        earthquakes: 地震数据列表 (已过滤)
        output_path: 输出文件路径
        map_extent: (lon_min, lon_max, lat_min, lat_max) 地图范围
        map_name: 地图名称，用于标题
        filter_info: 过滤条件字典，用于在标题中显示
            {
                'mag_min': float, 'mag_max': float,
                'depth_min': float, 'depth_max': float,
                'flags': list, 'is_jma': bool
            }
    """
    
    setup_science_style()
    
    # 提取数据
    lats = np.array([eq['lat'] for eq in earthquakes])
    lons = np.array([eq['lon'] for eq in earthquakes])
    depths = np.array([eq['depth'] for eq in earthquakes])
    mags = np.array([eq['mag'] for eq in earthquakes])
    
    # 计算圆圈大小
    sizes = np.array([magnitude_to_size(m) for m in mags])
    
    # 创建颜色映射
    depth_cmap, depth_norm = create_depth_colormap()
    
    # 设置地图范围
    if map_extent is None:
        lon_min, lon_max = 122, 150
        lat_min, lat_max = 24, 50
    else:
        lon_min, lon_max, lat_min, lat_max = map_extent
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 设置背景色为海洋蓝
    ax.set_facecolor('#D4E8F2')
    
    # =================================
    # 加载地理数据
    # =================================
    print("正在加载地理数据...")
    countries, provinces = get_natural_earth_data()
    
    if countries is not None and provinces is not None:
        # 裁剪到显示范围内的区域
        from shapely.geometry import box
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        
        # 筛选日本及周边国家
        target_countries = ['Japan', 'China', 'South Korea', 'North Korea', 'Russia', 'Taiwan', 'Philippines']
        
        # 绑制陆地（国家多边形）
        countries_clipped = countries.clip(bbox)
        countries_clipped.plot(
            ax=ax,
            facecolor='#F5F5F5',  # 浅灰色陆地
            edgecolor='#444444',  # 深灰色边界
            linewidth=0.8,
            zorder=1
        )
        
        # 绑制省级边界
        provinces_clipped = provinces.clip(bbox)
        provinces_clipped.boundary.plot(
            ax=ax,
            edgecolor='#888888',
            linewidth=0.4,
            linestyle='-',
            zorder=2
        )
        
        print("地理数据加载完成!")
    else:
        print("未能加载地理数据，使用简化版地图...")
    
    # =================================
    # 添加经纬网格
    # =================================
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.6, zorder=3)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.4, zorder=3)
    
    # =================================
    # 绑制地震散点 (空心圆)
    # =================================
    
    # 按深度排序，深层先画，浅层后画（浅层显示在上面）
    sort_idx = np.argsort(-depths)
    
    # 获取排序后的颜色
    colors = depth_cmap(depth_norm(depths[sort_idx]))
    
    # 绑制空心圆
    scatter = ax.scatter(
        lons[sort_idx], 
        lats[sort_idx],
        s=sizes[sort_idx],
        facecolors='none',  # 空心
        edgecolors=colors,
        linewidths=0.8,
        alpha=0.85,
        zorder=10
    )
    
    # =================================
    # 设置坐标轴范围和标签
    # =================================
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    
    # 格式化刻度标签
    ax.set_xticklabels([f'{int(x)}°E' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{int(y)}°N' for y in ax.get_yticks()])
    
    # =================================
    # 添加颜色条 (深度)
    # =================================
    # 创建颜色条
    sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=depth_norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02,
                        aspect=30)
    cbar.set_label('Depth (km)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # =================================
    # 添加震级图例
    # =================================
    legend_mags = [2, 4, 6, 8]
    legend_sizes = [magnitude_to_size(m) for m in legend_mags]
    
    legend_elements = []
    for mag, size in zip(legend_mags, legend_sizes):
        legend_elements.append(
            Line2D([0], [0], 
                   marker='o', 
                   color='w',
                   markerfacecolor='none',
                   markeredgecolor='#333333',
                   markeredgewidth=0.8,
                   markersize=np.sqrt(size) * 0.7,
                   label=f'M {mag}',
                   linestyle='None')
        )
    
    legend = ax.legend(
        handles=legend_elements,
        title='Magnitude',
        title_fontsize=10,
        loc='upper left',  # 移到左上角
        frameon=True,
        framealpha=0.95,
        edgecolor='#333333',
        borderpad=1.0,
        labelspacing=1.5,
        handletextpad=1.0
    )
    legend.get_title().set_fontweight('bold')
    
    # =================================
    # 添加标题和信息
    # =================================
    # 筛选显示范围内的地震数量
    in_range = ((lons >= lon_min) & (lons <= lon_max) & 
                (lats >= lat_min) & (lats <= lat_max))
    n_in_range = np.sum(in_range)
    depths_in_range = depths[in_range]
    mags_in_range = mags[in_range]
    
    # 构建过滤条件字符串
    filter_str_parts = []
    if filter_info:
        if filter_info.get('mag_min') is not None or filter_info.get('mag_max') is not None:
            mag_min_val = filter_info.get('mag_min', 'any')
            mag_max_val = filter_info.get('mag_max', 'any')
            if mag_min_val is not None and mag_max_val is not None:
                filter_str_parts.append(f"Mag: {mag_min_val}~{mag_max_val}")
            elif mag_min_val is not None:
                filter_str_parts.append(f"Mag ≥ {mag_min_val}")
            elif mag_max_val is not None:
                filter_str_parts.append(f"Mag ≤ {mag_max_val}")
        
        if filter_info.get('depth_min') is not None or filter_info.get('depth_max') is not None:
            depth_min_val = filter_info.get('depth_min')
            depth_max_val = filter_info.get('depth_max')
            if depth_min_val is not None and depth_max_val is not None:
                filter_str_parts.append(f"Depth: {depth_min_val}~{depth_max_val}km")
            elif depth_min_val is not None:
                filter_str_parts.append(f"Depth ≥ {depth_min_val}km")
            elif depth_max_val is not None:
                filter_str_parts.append(f"Depth ≤ {depth_max_val}km")
        
        if filter_info.get('flags') and filter_info.get('is_jma'):
            flags_str = ','.join(filter_info['flags'])
            filter_str_parts.append(f"JMA Flag: [{flags_str}]")
    
    filter_line = ''
    if filter_str_parts:
        filter_line = '\nFilter: ' + ' | '.join(filter_str_parts)
    
    # 设置标题
    if len(depths_in_range) > 0 and len(mags_in_range) > 0:
        ax.set_title(
            f'Earthquake Distribution - {map_name}{filter_line}\n'
            f'(N = {n_in_range}, Depth: {depths_in_range.min():.1f} - {depths_in_range.max():.1f} km, '
            f'Magnitude: {mags_in_range.min():.1f} - {mags_in_range.max():.1f})',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
    else:
        ax.set_title(
            f'Earthquake Distribution - {map_name}{filter_line}\n'
            f'(N = {n_in_range})',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
    
    # 添加数据来源标注（移到右下角）
    ax.text(
        0.99, 0.01,
        'Data: Earthquake Catalog\nProjection: Plate Carrée',
        transform=ax.transAxes,
        fontsize=7,
        color='#666666',
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
        zorder=20
    )
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=500, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"地图已保存至: {output_path}")
    
    # 显示图像
    plt.show()
    
    return fig, ax

# =============================================================================
# 7. 主程序
# =============================================================================
if __name__ == '__main__':
    # 读取数据
    data_file = r'd:\Antigravity\docu\震源.txt'
    earthquakes, is_jma_format = read_earthquake_data(data_file)
    
    print(f"共读取 {len(earthquakes)} 条地震记录")
    print(f"数据格式: {'JMA格式' if is_jma_format else 'Hi-net格式'}")
    
    # 统计信息
    depths = [eq['depth'] for eq in earthquakes]
    mags = [eq['mag'] for eq in earthquakes]
    
    print(f"震源深度范围: {min(depths):.1f} - {max(depths):.1f} km")
    print(f"震级范围: {min(mags):.1f} - {max(mags):.1f}")
    
    # =================================
    # 过滤设置 (可根据需要修改)
    # =================================
    
    # 震级过滤: 只绘制0级以上地震
    MAG_MIN = 0
    MAG_MAX = None  # None表示不限制
    
    # 深度过滤: 例如只绘制10-100km地震，设置为None表示不过滤
    DEPTH_MIN = None  # 例如: 10
    DEPTH_MAX = None  # 例如: 100
    
    # JMA Flag过滤: 只绘制高精度震源 (K, k, A)
    # K: 气象厅震源, k: 简易气象厅震源, A: 自动气象厅震源
    JMA_FLAGS = ['K', 'k', 'A']  # 设置为None表示不过滤
    
    # =================================
    # 应用过滤
    # =================================
    filtered_earthquakes = filter_earthquakes(
        earthquakes,
        mag_min=MAG_MIN,
        mag_max=MAG_MAX,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        flags=JMA_FLAGS if is_jma_format else None
    )
    
    print(f"\n过滤后剩余 {len(filtered_earthquakes)} 条地震记录")
    
    # 过滤条件信息 (传递给绑图函数用于标题显示)
    filter_info = {
        'mag_min': MAG_MIN,
        'mag_max': MAG_MAX,
        'depth_min': DEPTH_MIN,
        'depth_max': DEPTH_MAX,
        'flags': JMA_FLAGS if is_jma_format else None,
        'is_jma': is_jma_format
    }
    
    # =================================
    # 定义两种地图范围
    # =================================
    
    # 地图1: 日本本土四岛 (北海道、本州、四国、九州)
    extent_main_islands = (129, 146, 30, 46)
    
    # 地图2: 日本全境 (包括台湾附近和小笠原群岛)
    extent_full_territory = (119, 152, 20, 50)
    
    # =================================
    # 绘制两张地图
    # =================================
    
    print("\n--- 绘制地图1: 日本本土四岛 ---")
    output_path1 = r'd:\Antigravity\docu\earthquake_map_main_islands.png'
    fig1, ax1 = plot_earthquakes(
        filtered_earthquakes, 
        output_path=output_path1,
        map_extent=extent_main_islands,
        map_name='Japanese Main Islands',
        filter_info=filter_info
    )
    
    print("\n--- 绘制地图2: 日本全境 ---")
    output_path2 = r'd:\Antigravity\docu\earthquake_map_full_territory.png'
    fig2, ax2 = plot_earthquakes(
        filtered_earthquakes,
        output_path=output_path2,
        map_extent=extent_full_territory,
        map_name='Full Territory',
        filter_info=filter_info
    )
    
    print("\n两张地图均已生成完成!")


"""
Analysis utilities for PyHEA output, including Short Range Order (SRO) parameter calculations.
"""
import os
import dpdata
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
import WarrenCowleyParameters as wc
from apollox2.utils import logger
from scipy.spatial import cKDTree
from pymatgen.core import Structure
from collections import defaultdict
def read_poscar_cartesian(filename):
    """
    读取POSCAR文件，返回笛卡尔坐标和元素符号列表。
    假设：
    - 第6行为元素符号列表
    - 第7行为对应元素数量
    - 第9行开始为坐标，笛卡尔坐标（无选择性动力学行）
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    elem_line = lines[5].strip().split()
    elem_num_line = lines[6].strip().split()
    elem_num = [int(x) for x in elem_num_line]
    natoms = sum(elem_num)

    coord_lines = lines[8:8+natoms]
    coords = []
    for line in coord_lines:
        parts = line.strip().split()
        x, y, z = map(float, parts[:3])
        coords.append([x, y, z])

    species = []
    for elem, count in zip(elem_line, elem_num):
        species.extend([elem]*count)

    return np.array(coords), species

import numpy as np
from scipy.spatial import cKDTree
def calculate_sro(filename: str, cutoff_radii: list[float]):
    """
    以与C++代码相同的、更严谨的方式计算多壳层Warren-Cowley短程有序参数。
    该方法能够正确处理每个原子的配位数不相等的复杂情况。
    
    计算逻辑:
    1. 对每个中心原子m，单独计算其在各壳层s中的邻居分布，得到局部配位数 Z_s(m) 和 j邻居数 N_js(m)。
    2. 计算局部的条件概率 P(j|m)_s = N_js(m) / Z_s(m)。
    3. 将所有i类型原子的局部概率 P(j|m)_s 相加，然后除以i原子的总数 N_i，得到最终的平均条件概率 P_j|i_s。
    4. 使用标准公式 alpha_ij_s = 1 - P_j|i_s / c_j 计算SRO参数。

    参数:
        filename (str): 结构文件路径 (如 POSCAR, cif 等)
        cutoff_radii (list of float): 按升序排列的邻居截断半径列表

    返回:
        sro (ndarray): shape=(num_shells, n_types, n_types) 的 SRO (alpha_ij) 矩阵
        elements (list): 按字母顺序排列的元素符号列表，其顺序与SRO矩阵的索引一一对应
    """
    # 1. 初始化和数据准备
    # pymatgen 在处理 VASP 文件时，通常更倾向于直接读取 POSCAR/CONTCAR，而不是目录
    # 这里我们假设 filename 是一个标准的文件路径
    if filename.endswith('vasp/poscar'): 
         filename = filename.replace('vasp/poscar', 'vasp')
    structure = Structure.from_file(filename)
    
    # 获取元素种类、数量和浓度 (保证顺序一致性)
    composition = structure.composition
    # 获取元素符号并按字母排序，以确保索引的唯一性和对应性
    elements = [el.symbol for el in composition.elements]
    element_to_idx = {el: idx for idx, el in enumerate(elements)}
    n_types = len(elements)
    
    concentrations = {el: composition.get_atomic_fraction(el) for el in elements}
    atom_counts = {el: composition[el] for el in elements}
    
    num_shells = len(cutoff_radii)
    # 这个数组将用来存储所有i类型原子的局部概率之和: Σ P(j|m)
    # 这是计算最终 P_j|i 的分子部分
    sum_of_local_probabilities = np.zeros((num_shells, n_types, n_types))
    
    # 2. 遍历每个原子，计算局部概率并累加
    all_neighbors = structure.get_all_neighbors(cutoff_radii[-1])
    
    for i, site_neighbors in enumerate(all_neighbors):
        type_i = structure[i].species_string
        # 如果中心原子类型不在我们的目标元素列表中（例如，在部分占据位点中），则跳过
        if type_i not in element_to_idx:
            continue
        idx_i = element_to_idx[type_i]
        
        # --- 为当前中心原子 i 计算其局部的邻居统计信息 ---
        # 局部配位数 Z_s(m)
        local_coordination_per_shell = defaultdict(int)
        # 局部j邻居数 N_js(m)
        local_neighbor_counts_per_shell = defaultdict(lambda: defaultdict(int))
        
        for neighbor_site, dist, _, _ in site_neighbors:
            type_j = neighbor_site.species_string
            if type_j not in element_to_idx:
                continue

            # 确定邻居属于哪个壳层
            # 从内到外检查，邻居属于它所满足的第一个（最小的）截断半径
            shell_idx = -1
            for s, cutoff in enumerate(cutoff_radii):
                if dist <= cutoff:
                    shell_idx = s
                    break
            if shell_idx == -1:
                continue
            
            # 由于 cutoff_radii 是累积的，我们需要确保每个邻居只被计数一次
            # 这里的逻辑是正确的，因为它会break
            
            # 累加壳层 s 的总配位数和特定类型邻居数
            prev_cutoff = cutoff_radii[shell_idx-1] if shell_idx > 0 else 0
            if prev_cutoff < dist <= cutoff_radii[shell_idx]:
                 local_coordination_per_shell[shell_idx] += 1
                 local_neighbor_counts_per_shell[shell_idx][type_j] += 1

        # --- 计算该原子的局部概率 P(j|m)，并将其累加到全局总和中 ---
        for s in range(num_shells):
            total_neighbors_in_shell = local_coordination_per_shell[s]
            if total_neighbors_in_shell > 0:
                for type_j, count_j in local_neighbor_counts_per_shell[s].items():
                    # *** 修正错误：使用 element_to_idx 而不是 elem_to_idx ***
                    idx_j = element_to_idx[type_j]
                    local_prob = count_j / total_neighbors_in_shell
                    sum_of_local_probabilities[s, idx_i, idx_j] += local_prob

    # 3. 计算最终的SRO参数
    sro = np.zeros((num_shells, n_types, n_types))
    
    for s in range(num_shells):
        for i_idx, i_type in enumerate(elements):
            N_i = atom_counts[i_type]
            if N_i == 0:
                continue
                
            for j_idx, j_type in enumerate(elements):
                c_j = concentrations[j_type]
                if c_j == 0:
                    continue
                
                # 计算平均条件概率 P_j|i = (Σ P(j|m)) / N_i
                sum_probs = sum_of_local_probabilities[s, i_idx, j_idx]
                P_j_given_i = sum_probs / N_i
                
                # 计算 alpha_ij
                alpha_ij = 1 - (P_j_given_i / c_j)
                sro[s, i_idx, j_idx] = alpha_ij
                
    return sro, elements


def plot_sro_heatmap(sro_values, atom_labels, output_file='sro_heatmap.png'):
    """Plot SRO parameters as a heatmap.
    
    @param sro_values ndarray Matrix of SRO values
    @param atom_labels list List of atom type labels
    @param output_file str Output file path for the plot
    """
    plt.figure(figsize=(9, 7))  # Increased figure size
    plt.imshow(sro_values, cmap='RdBu', vmin=-1, vmax=1)
    cbar = plt.colorbar(label='Warren-Cowley Parameter')
    cbar.ax.tick_params(labelsize=14)  # Colorbar tick size
    cbar.set_label('Warren-Cowley Parameter', size=16)  # Colorbar label size
    
    # Add labels with increased font sizes
    plt.xticks(range(len(atom_labels)), atom_labels, fontsize=14)
    plt.yticks(range(len(atom_labels)), atom_labels, fontsize=14)
    plt.xlabel('Atom Type', fontsize=16)
    plt.ylabel('Atom Type', fontsize=16)
    plt.title('Warren-Cowley Parameters (First Shell)', fontsize=18)
    
    # Add value annotations with increased font size
    for i in range(len(atom_labels)):
        for j in range(len(atom_labels)):
            plt.text(j, i, f'{sro_values[i,j]:.2f}', 
                    ha='center', va='center', fontsize=13)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)  # Added tight layout and increased DPI
    plt.close()

def analyze_result(output_file, target_sro, element_types,cutoffs):
    TYPE_LIST = element_types
    result_sro,_ = calculate_sro(output_file,cutoffs)
    
    # reshape target_sro to (num_shells, n_types, n_types)
    target_sro = target_sro.reshape(len(cutoffs), len(element_types), len(element_types))
    
    logger.info(f"target_sro shape: {target_sro.shape}, result_sro shape: {result_sro.shape}")
    
    for shell in range(len(cutoffs)):
        logger.info(f"Shell {shell+1} (cutoff={cutoffs[shell]}):")
        logger.info(f"Warren-Cowley Parameters (Result): {result_sro[shell].tolist()}")
        logger.info(f"Target SRO: {target_sro[shell].tolist()}")

        # Plot heatmap (你已有的plot_sro_heatmap函数)
        plot_sro_heatmap(result_sro[shell], TYPE_LIST, f'heatmap_shell{shell+1}.png')

        sro_diff = result_sro[shell] - target_sro[shell]
        
        logger.info("Type  |  Result SRO  |  Target SRO  |  Difference:")
        logger.info("==================================================")
        for i in range(len(element_types)):
            for j in range(len(element_types)):
                logger.info(
                    f" {TYPE_LIST[i]}-{TYPE_LIST[j]}  | "
                    f"{result_sro[shell][i, j]:>9.3f}    | "
                    f"{target_sro[shell][i, j]:>9.3f}    | "
                    f"{sro_diff[i, j]:>8.3f}"
                )
        logger.info("==================================================")

    # 全局误差可以根据所有壳层计算，比如均值
    total_diff = result_sro - target_sro
    mae = np.mean(np.abs(total_diff))
    rmse = np.sqrt(np.mean(total_diff**2))

    logger.info("Overall Error Metrics:")
    logger.info(f"Mean Absolute Error: {mae:.3f}")
    logger.info(f"Root Mean Square Error: {rmse:.3f}")

    return result_sro, mae, rmse

def analyze_structure(structure_file, latt_type='FCC', element_types=None, output_file=None):
    """Analyze the SRO parameters of a given structure file and generate visualization.
    
    @param structure_file str Path to the structure file (LAMMPS .lmp or VASP POSCAR)
    @param latt_type str Lattice type ('FCC' or 'BCC')
    @param element_types list List of element types in the structure
    @param output_file str Optional output filename for the heatmap. If None, generates default name
    @return tuple SRO values and visualization file path
    """
    # Calculate SRO parameters
    sro_values = calculate_sro(structure_file, latt_type)
    
    # If element types not provided, use generic labels
    if element_types is None:
        n_types = len(sro_values[0])
        element_types = TYPE_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I"][:n_types]
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(structure_file))[0]
        output_file = f'heatmap.png'
    
    # Plot heatmap
    plot_sro_heatmap(sro_values[0], element_types, output_file)
    
    return sro_values[0], output_file

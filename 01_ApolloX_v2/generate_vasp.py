import os
import csv
import re
import sys
import argparse

try:
    import yaml
except ImportError:
    print("错误：PyYAML模块未找到。请先通过 'pip install pyyaml' 安装它。")
    sys.exit(1)

# --- 数据结构 ---

class Lattice:
    """一个简单的类，用于保存晶格信息。"""
    def __init__(self, latt_con, latt_vec, coords):
        self.latt_con = latt_con  # 晶格常数 (缩放因子)
        self.latt_vec = latt_vec  # 3x3 的晶格矢量列表
        self.coords = coords     # 所有原子的坐标列表

# --- 核心功能 ---

def read_poscar_for_lattice(filename):
    """
    从一个POSCAR文件中读取晶格结构信息。

    @param filename str POSCAR文件名
    @return tuple (Lattice object, list of element labels)
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        comment = lines[0].strip()
        latt_con = float(lines[1].strip())

        latt_vec = []
        for i in range(2, 5):
            latt_vec.append([float(v) for v in lines[i].strip().split()])

        element_labels = lines[5].strip().split()

        # 确认坐标类型
        coord_type_line_index = 7
        if not lines[6].strip().isdigit():
             # 如果第7行不是原子数（例如Selective dynamics），坐标类型在第8行
            coord_type_line_index = 8

        coord_type = lines[coord_type_line_index-1].strip()
        if not coord_type.lower().startswith('cart'):
            print(f"错误: 脚本目前只支持'Cartesian'坐标。在'{filename}'中找到的是'{coord_type}'。")
            return None, None

        # 读取坐标
        coords = []
        for i in range(coord_type_line_index, len(lines)):
            line = lines[i].strip()
            if line:
                coords.append([float(c) for c in line.split()[:3]]) # 只取前三列

        print(f"成功从 '{filename}' 读取晶格信息。")
        print(f"  - 元素: {element_labels}")
        print(f"  - 读取到 {len(coords)} 个原子坐标。")

        return Lattice(latt_con, latt_vec, coords), element_labels

    except FileNotFoundError:
        print(f"错误: 结构文件 '{filename}' 未找到。")
        return None, None
    except (ValueError, IndexError) as e:
        print(f"错误: 解析文件 '{filename}' 时出错。请确保其为有效的POSCAR格式。错误: {e}")
        return None, None

def write_poscar(nest, latt, ntyp, element_labels, file):
    """以VASP POSCAR格式写入原子结构数据。"""
    natm = len(nest)
    # 确保latt.coords包含足够多的坐标
    if len(latt.coords) < natm:
        print(f"错误: 参考结构文件只提供了 {len(latt.coords)} 个坐标, 但当前结构需要 {natm} 个。")
        return False # 返回失败状态

    # VASP要求原子按元素类型分组，所以我们需要先排序
    # 创建一个 (类型, 坐标) 的元组列表
    # 我们只使用参考结构中的前 natm 个坐标
    sorted_atoms = sorted(zip(nest, latt.coords[:natm]), key=lambda pair: pair[0])

    # 解包排序后的列表
    sorted_nest, sorted_coords = zip(*sorted_atoms)

    # 统计每种类型的原子数
    atom_counts = [0] * ntyp
    for atom_type in sorted_nest:
        atom_counts[atom_type] += 1

    with open(file, "w") as final_coords:
        # 写入文件头
        final_coords.write(f"ApolloX v2.0.0\n")
        # 写入缩放因子
        final_coords.write(f"   {latt.latt_con}\n")
        # 写入晶格矢量
        for vec in latt.latt_vec:
            final_coords.write(f"  {vec[0]:.8f}   {vec[1]:.8f}   {vec[2]:.8f}\n")
        # 写入元素标签
        final_coords.write("   " + "  ".join(element_labels) + "\n")
        # 写入原子数量
        final_coords.write("   " + "  ".join(map(str, atom_counts)) + "\n")
        # 写入坐标类型
        final_coords.write("Cartesian\n")

        # 写入排序后的坐标
        for coord in sorted_coords:
            final_coords.write(f"  {coord[0]:.8f}   {coord[1]:.8f}   {coord[2]:.8f}\n")
    return True # 返回成功状态

def parse_and_generate_files(input_file, output_folder, latt_obj, element_labels, min_fitness, max_fitness, outcsv, top_n):
    """
    解析演化数据文件，提取所有结构，按Fitness排序，然后只为前top_n个结构生成文件。
    """
    print(f"\n正在解析 '{input_file}'...")

    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return

    # 1. 使用正则表达式查找所有晶格数据块并提取
    lattice_blocks = re.finditer(
        r"Lattice (\d+) Fitness: ([\d.eE+-]+)\nData: ([\d\s]+)",
        content
    )

    all_extracted_structures = []

    for match in lattice_blocks:
        lattice_idx, fitness_str, data_str = match.groups()
        fitness = float(fitness_str)

        # 按fitness范围初步过滤 (如果你仍然想保留这个功能)
        if not (min_fitness <= fitness <= max_fitness):
            continue

        nest = [int(x) for x in data_str.strip().split()]
        
        # 将提取的数据存储为一个字典列表，方便后续排序
        all_extracted_structures.append({
            'original_id': lattice_idx,
            'fitness': fitness,
            'nest_data': nest
        })

    if not all_extracted_structures:
        print("\n在指定条件下未找到任何结构。")
        return

    print(f"共提取了 {len(all_extracted_structures)} 个符合初步筛选条件的结构记录。")

    # 2. 按照 Fitness 从小到大排序
    all_extracted_structures.sort(key=lambda x: x['fitness'])

    # 3. 截取用户请求的前 top_n 个结构
    if top_n is not None and top_n > 0:
        structures_to_process = all_extracted_structures[:top_n]
        print(f"将截取并生成 Fitness 最小的前 {len(structures_to_process)} 个结构。")
    else:
        structures_to_process = all_extracted_structures
        print("未指定限制数量或数量无效，将处理所有提取的结构。")

    # 4. 生成文件并准备 CSV 数据
    os.makedirs(output_folder, exist_ok=True)
    print(f"输出文件夹 '{output_folder}' 已就绪。")

    csv_data = []
    structures_generated = 0
    num_types = len(element_labels)

    for index, struct_data in enumerate(structures_to_process):
        # 使用排名 (1, 2, 3...) 作为新的文件名，这样最优秀的结构就是 1.vasp
        file_index = index + 1 
        filename = f"{file_index}.vasp"
        filepath = os.path.join(output_folder, filename)

        # 写入POSCAR (.vasp) 文件
        success = write_poscar(struct_data['nest_data'], latt_obj, num_types, element_labels, filepath)

        if success:
            structures_generated += 1
            # 记录数据供CSV使用，包括排名、原ID、Fitness
            csv_data.append([filename, struct_data['original_id'], struct_data['fitness']])

    # 5. 写入 CSV 总结文件
    if structures_generated > 0:
        with open(outcsv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Generated Filename', 'Original Lattice ID', 'Root Mean Square Error (Fitness)'])
            writer.writerows(csv_data)

        print(f"\n处理完成！")
        print(f"成功在 '{output_folder}' 目录中生成了 {structures_generated} 个 .vasp 文件。")
        print(f"文件命名从 1.vasp 到 {structures_generated}.vasp，Fitness依次递增。")
        print(f"一份详细摘要已保存至 '{outcsv}'。")
    else:
        print("\n未能成功生成任何文件。")


if __name__ == '__main__':
    # --- 主程序执行 ---

    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="从evolution_data.txt解析数据，按Fitness排序，提取最佳结构并生成VASP POSCAR文件。")
    
    parser.add_argument("config", type=str,
                        help="配置文件路径 (例如 config.yaml)")
    parser.add_argument("input_file", type=str,
                        help="包含演化数据的文本文件")
    parser.add_argument("-n", "--num_structures", type=int, required=True,
                        help="需要提取和生成的最佳结构的个数 (必须指定)")
    parser.add_argument('--min_fitness', type=float, default=0.0,
                        help='要考虑的最低fitness值 (默认: 0.0)')
    parser.add_argument('--max_fitness', type=float, default=float('inf'),
                        help='要考虑的最高fitness值 (默认: 无穷大)')
    parser.add_argument('--outdir', type=str, default="best_structures",
                        help='保存生成结构的输出文件夹名称 (默认: best_structures)')
    parser.add_argument('--outcsv', type=str, default="top_results.csv",
                        help='保存结果汇总的CSV文件名称 (默认: top_results.csv)')
    
    args = parser.parse_args()

    CONFIG_FILENAME = args.config
    INPUT_FILENAME = args.input_file
    OUTPUT_FOLDER = args.outdir
    OUT_CSV = args.outcsv
    TOP_N = args.num_structures

    # 参数校验
    if TOP_N <= 0:
        print("错误: 请求的结构数量 (-n/--num_structures) 必须大于 0。")
        sys.exit(1)

    # 3. 从配置文件加载晶格信息
    try:
        with open(CONFIG_FILENAME, 'r') as f:
            config = yaml.safe_load(f)
        structure_file = config.get('structure')
        if not structure_file:
            raise KeyError("未在配置文件中找到 'structure' 键。")
    except FileNotFoundError:
        print(f"错误: 配置文件 '{CONFIG_FILENAME}' 未找到。")
        sys.exit(1)
    except (yaml.YAMLError, KeyError) as e:
        print(f"错误: 解析 '{CONFIG_FILENAME}' 时出错。错误信息: {e}")
        sys.exit(1)

    latt_obj, element_labels = read_poscar_for_lattice(structure_file)

    if latt_obj is None:
        sys.exit(1)

    # 4. 运行脚本
    print(f"设定目标: 提取 Fitness 最小的前 {TOP_N} 个结构。")
    if args.min_fitness > 0.0 or args.max_fitness != float('inf'):
         print(f"(应用预过滤范围: {args.min_fitness} <= fitness <= {args.max_fitness})")

    parse_and_generate_files(INPUT_FILENAME, OUTPUT_FOLDER, latt_obj, element_labels, args.min_fitness, args.max_fitness, OUT_CSV, TOP_N)


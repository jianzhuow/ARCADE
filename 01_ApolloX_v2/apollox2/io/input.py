import os
import yaml
import numpy as np
from apollox2.utils.device import get_gpu_info
from apollox2.utils.logger import logger

class input_config:
    """
    A class to handle configuration input from a YAML file.

    Attributes
    ----------
    file_path : str
        The path to the YAML configuration file.
    config : dict
        The configuration data loaded from the YAML file.
    structure_data : dict
        The structure data loaded from the POSCAR file.

    Methods
    -------
    element:
        Returns the list of elements from the configuration.
    cell_dim:
        Returns the cell dimensions from the configuration.
    solutions:
        Returns the number of solutions from the configuration.
    mc_step:
        Returns the number of Monte Carlo steps from the configuration.
    total_iter:
        Returns the total number of iterations from the configuration.
    global_iter:
        Returns the number of global iterations from the configuration.
    max_shell_num:
        Returns the maximum shell number from the configuration.
    weight:
        Returns the weight list from the configuration.
    structure:
        Returns the structure from the configuration.
    _read_structure:
        Reads and validates the POSCAR structure file.
    """

    # Allowed keys in the configuration file
    ALLOWED_KEYS = {
        'type', 'element', 'cell_dim','cutoff', 'pbc', 'solutions', 'device', 'total_iter', 'weight',
        'parallel_task', 'converge_depth', 'max_shell_num', 'structure', 'output', 'target_sro','subgroup','cut_iter','srogroup','mc_num_lattice_per_batch','mc_num_tasks','mc_search_depth','mc_fitness_threshold','mc_initial_temperature','mc_cooling_rate','mc_annealing_steps','mc_batch_num'
    }

    # Supported output formats
    SUPPORTED_FORMATS = {
        'vasp/poscar': 'VASP POSCAR format (default)',
        'lammps/lmp': 'LAMMPS data format'
    }

    def __init__(self, file_path):
        self.file_path = file_path
        self.config = self._read_yaml()
        self.structure = self._read_structure()
        self._validate_config()

    def _read_yaml(self):
        try:
            with open(self.file_path, 'r') as file:
                data = yaml.safe_load(file)
                if data is None or not isinstance(data, dict):
                    raise ValueError(f"Invalid or empty YAML file: {self.file_path}")
                if not all(key in self.ALLOWED_KEYS for key in data):
                    raise ValueError(f"Invalid keys in YAML file: {self.file_path}")
                return data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"YAML file not found: {self.file_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {self.file_path}") from e

    def _read_structure(self):
        """
        Reads and validates the POSCAR structure file.

        Raises
        ------
        ValueError:
            If the POSCAR file is malformed or contains invalid data.
        """

        structure_file = self.config.get('structure', None)
        structure_data = {}
        current_dir = os.getcwd()
        structure_file = os.path.join(current_dir, structure_file)
        try:
            with open(structure_file, 'r') as file:
                # Read all lines and filter out empty lines and comments
                lines = [line.split('#')[0].strip() for line in file if line.strip()]
                # Extracting relevant data from the POSCAR file
                structure_data['comment'] = lines[0].strip()
                structure_data['lattice_constant'] = float(lines[1].strip())

                structure_data['lattice_vectors'] = [
                    list(map(float, lines[2].split())),
                    list(map(float, lines[3].split())),
                    list(map(float, lines[4].split()))
                ]
                element_counts = np.array(list(map(int, lines[6].split())))
    # 原子总数（加和）
                num_atoms = int(np.sum(element_counts))

                structure_data['num_atoms'] = num_atoms
                structure_data['coordinate_type'] = lines[7].strip().lower()
                if structure_data['coordinate_type'] not in ['direct', 'cartesian']:
                    raise ValueError("Coordinate type must be either 'direct' or 'cartesian'.")

                if len(lines[8:]) != structure_data['num_atoms']:
                    raise ValueError("Number of atomic positions does not match declared atom count")

                positions = []
                for line in lines[8:]:
                    if len(line.split()) != 3:
                        raise ValueError("Invalid atomic position format")
                    positions.append(list(map(float, line.split())))
                structure_data['positions'] = positions

        except FileNotFoundError as e:
            raise FileNotFoundError(f"POSCAR file not found: {structure_file}") from e

        self._validate_structure(structure_data)
        return structure_data

    def _validate_config(self):
        """Validate the configuration data.

        Raises
        ------
        ValueError:
            If any configuration values are invalid.
        """
        # Validate device configuration
        device = self.config.get('device', 'cpu').lower()
        if device not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid device type '{device}'. Must be either 'cpu' or 'gpu'")

        # Check device availability
        gpu_info = get_gpu_info()
        gpu_available = bool(gpu_info)
        
        if device == 'gpu' and not gpu_available:
            logger.warning("GPU device requested but no GPU is available. Falling back to CPU.")
            self.config['device'] = 'cpu'
            device = self.config.get('device', 'cpu').lower()
        device_name = 'CPU' if device == 'cpu' else gpu_info[0]['name']
        
        if device == 'cpu' and gpu_available:
            logger.warning(f"Set CPU device but {gpu_info[0]['name']} GPU is available. Inefficient configuration!!")
        logger.info(f"Running with device: {device_name}")

        # Validate target SRO file
        target_sro_file = self.config.get('target_sro')
        if target_sro_file:
            if not os.path.exists(target_sro_file):
                raise FileNotFoundError(f"Target SRO file not found: {target_sro_file}")
        #self._validate_sro_file(target_sro_file)


    def _validate_structure(self, structure_data):
        """Validate the structure data from POSCAR.

        Parameters
        ----------
        structure_data : dict
            The structure data to validate.

        Raises
        ------
        ValueError:
            If the structure data is invalid.
        """
        # Check number of atoms matches coordinates
        if len(structure_data.get('positions', [])) != structure_data.get('num_atoms', 0):
            raise ValueError("Number of atomic positions does not match declared atom count")
    @property
    def cut_iter(self):
        return self.config['cut_iter']

    @property
    def type(self):
        """
        返回元素种类数（POSCAR 中元素符号的数量）
        """
        structure_file = self.config['structure']
        if not os.path.isfile(structure_file):
            raise FileNotFoundError(f"POSCAR file not found: {structure_file}")

        with open(structure_file, 'r') as file:
            lines = [line.split('#')[0].strip() for line in file if line.strip()]

        # VASP5 格式，第6行是元素符号
        element_symbols = lines[5].split()
        return len(element_symbols)
    @property
    def element_type(self):
        """
        返回元素种类数（POSCAR 中元素符号的数量）
        """
        structure_file = self.config['structure']
        if not os.path.isfile(structure_file):
            raise FileNotFoundError(f"POSCAR file not found: {structure_file}")

        with open(structure_file, 'r') as file:
            lines = [line.split('#')[0].strip() for line in file if line.strip()]

        # VASP5 格式，第6行是元素符号
        element_symbols = lines[5].split()
        return element_symbols
        
    @property
    def species(self):
        """
        从POSCAR文件中读取并生成一个包含每个原子元素符号的完整列表。

        Args:
            poscar_filepath (str): POSCAR文件的路径。

        Returns:
            list of str: 例如 ['Ni', 'Ni', ..., 'Cr']
        """
        poscar_filepath=self.config['structure']
        if not os.path.isfile(poscar_filepath):
            raise FileNotFoundError(f"POSCAR file not found: {poscar_filepath}")

        with open(poscar_filepath, 'r') as file:
            lines = [line.strip() for line in file]

        # VASP 5+ 格式:
        # 第6行 (index 5) 是元素符号
        # 第7行 (index 6) 是每个元素的原子数量
        element_symbols = lines[5].split()
        atom_counts = [int(count) for count in lines[6].split()]

        if len(element_symbols) != len(atom_counts):
            raise ValueError("The number of element symbols does not match the number of atom counts in the POSCAR file.")

        # 使用列表推导式高效地展开列表
        full_species_list = [
            symbol
            for symbol, count in zip(element_symbols, atom_counts)
            for _ in range(count)
        ]

        return full_species_list


    @property
    def element(self):
        """
        返回每种元素的原子数列表
        """
        structure_file = self.config['structure']
        if not os.path.isfile(structure_file):
            raise FileNotFoundError(f"POSCAR file not found: {structure_file}")

        with open(structure_file, 'r') as file:
            lines = [line.split('#')[0].strip() for line in file if line.strip()]

        # 第7行是原子数
        counts = list(map(int, lines[6].split()))
        return counts
    # @property
    # def cell_dim(self):
    #     return self.config.get('cell_dim', [])

    @property
    def solutions(self):
        return self.config.get('solutions', 0)

    @property
    def parallel_task(self):
        return self.config.get('parallel_task', 0)

    @property
    def converge_depth(self):
        return self.config.get('converge_depth', 0)

    @property
    def total_iter(self):
        return self.config.get('total_iter', 0)

    @property
    def max_shell_num(self):
        if self.config.get('weight', 0) == 0:
            return 0
        else:
            return len(self.config.get('weight', []))

    @property
    def weight(self):
        return self.config.get('weight', [])
    
    @property
    def device(self):
        return self.config.get('device', 'cpu')

    @property
    def latt_const(self):
        return self.structure.get('lattice_constant', 0)
    
    @property
    def latt_vectors(self):
        return self.structure.get('lattice_vectors', [])
    
    @property
    def position(self):
        return self.structure.get('positions')
    
    @property
    def natoms(self):
        return self.structure.get('num_atoms')
    @property
    def cutoff(self):
        return self.config.get('cutoff')
    
    @property
    def pbc(self):
        return self.config.get('pbc')
    @property
    def mc_num_lattice_per_batch(self):
        return self.config.get('mc_num_lattice_per_batch',1)
    @property
    def mc_batch_num(self):
        return self.config.get('mc_batch_num',1)
    @property
    def mc_num_tasks(self):
        return self.config.get('mc_num_tasks',1)
    @property
    def mc_search_depth(self):
        return self.config.get('mc_search_depth',1)
    @property
    def mc_fitness_threshold(self):
        return self.config.get('mc_fitness_threshold',0.01)
    @property
    def mc_initial_temperature(self):
        return self.config.get("mc_initial_temperature",1)
    @property
    def mc_cooling_rate(self):
        return self.config.get("mc_cooling_rate",0.9)
    @property
    def mc_annealing_steps(self):
        return self.config.get("mc_annealing_steps",1)
    @property
    def output_format(self):
        """Get the output format configuration.

        Returns:
            str: The output format (default: 'poscar')
        """
        output_config = self.config.get('output', {})
        format_name = output_config.get('format', 'vasp/poscar').lower()
        
        if format_name not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {format_name}. "
                           f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS.keys())}")
        return format_name

    @property
    def output_name(self):
        """Get the output name.
        
        @return str Output name (default: 'output')
        """
        output_config = self.config.get('output', {})
        return output_config.get('name', 'output')
    
#     @property
#     def swap_group(self):
#         """
#         Safely reads atom swap groups from the subgroup file.
#         - Skips empty lines and comment lines (starting with '#').
#         - Ignores lines that contain non-integer values.
#         - Raises an error if the file is not found.
#         """
#         subgroup_file = self.config.get("subgroup", "")
#         if not subgroup_file or not os.path.exists(subgroup_file):
#             # This function requires the file to exist, raises an error if not found.
#             raise FileNotFoundError(f"Subgroup file '{subgroup_file}' not found or not specified in config.")

#         swap_list = []
#         # Use 'with' to ensure the file is always closed properly
#         with open(subgroup_file, 'r') as f:
#             for i, line_text in enumerate(f, 1):
#                 line = line_text.strip()
#                 # Skip empty lines and comment lines
#                 if not line or line.startswith('#'):
#                     continue
                
#                 try:
#                     # Attempt to convert the line to a list of integers
#                     indices = list(map(int, line.split()))
#                     swap_list.append(indices)
#                 except ValueError:
#                     # If conversion fails, print a warning and skip the line
#                     print(f"Warning: Could not parse line {i} in {subgroup_file}, skipping: '{line}'")

#         return swap_list
    @property
    def swap_group(self):
        """
        Safely reads atom swap groups from the subgroup file.

        Behavior:
        - If 'subgroup' is not specified in config, return None.
        - If 'subgroup' is specified but the file does not exist, raise FileNotFoundError.
        - Skip empty lines and comment lines (starting with '#').
        - Ignore lines that contain non-integer values.
        """

        subgroup_file = self.config.get("subgroup", None)

        # User did not specify subgroup -> do not use this feature
        if not subgroup_file:
            return [] 

        # User specified subgroup but file not found -> raise error
        if not os.path.exists(subgroup_file):
            raise FileNotFoundError(
                f"Subgroup file '{subgroup_file}' specified in config but not found."
            )

        swap_list = []

        with open(subgroup_file, 'r') as f:
            for i, line_text in enumerate(f, 1):
                line = line_text.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                try:
                    indices = list(map(int, line.split()))
                    swap_list.append(indices)

                except ValueError:
                    print(
                        f"Warning: Could not parse line {i} in "
                        f"{subgroup_file}, skipping: '{line}'"
                    )

        return swap_list
    @property
    def sro_group(self):
        """
        Safely reads atom group indices from the srogroup file.
        - Automatically collects any atom indices not defined in the file into a final group.
        - Skips empty lines and comment lines (starting with '#').
        - Ignores lines that contain non-integer values.
        """
        srogroup_file = self.config.get("srogroup", "")
        
        sro_list = []
        assigned_indices = set()

        # Step 1: Read the explicitly defined groups from the file, if it exists.
        if srogroup_file and os.path.exists(srogroup_file):
            with open(srogroup_file, 'r') as f:
                for i, line_text in enumerate(f, 1):
                    line = line_text.strip()
                    # Skip empty lines and comment lines
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        indices = list(map(int, line.split()))
                        sro_list.append(indices)
                        # Keep track of which atoms have been assigned to a group
                        assigned_indices.update(indices)
                    except ValueError:
                        print(f"Warning: Could not parse line {i} in {srogroup_file}, skipping: '{line}'")
        
        # Step 2: Find all atoms that were not in any of the defined groups.
        total_atoms = self.natoms
        unassigned = [i for i in range(total_atoms) if i not in assigned_indices]
        
        # Step 3: If there are any unassigned atoms, add them as a new, final group.
        if unassigned:
            sro_list.append(unassigned)
            
        # Edge Case: If the file was empty/missing and natoms > 0,
        # sro_list will now correctly contain one group with all atoms.
        # If natoms is 0, it will correctly return an empty list.
        return sro_list
    @property
    def target_sro(self):
        """
        从 target_sro 文件读取 SRO 矩阵（每个壳层为完整矩阵，不做补全）
        返回:
            numpy.ndarray: shape=(num_shells, num_types, num_types)
        """
        num_types = self.type
        target_sro_file = self.config.get('target_sro')

        if target_sro_file is None:
        # 没有文件则返回一个形状正确但第一维为0的空数组
            return np.empty((0, 0, num_types * num_types))

        target_sro_path = os.path.join(os.getcwd(), target_sro_file)
        if not os.path.isfile(target_sro_path):
            raise FileNotFoundError(f"Target SRO file not found: {target_sro_path}")

        shells = []
        current_matrix = []

        groups = []
        current_group_shells = []
        current_matrix = []

        with open(target_sro_path, 'r') as f:
            for line in f:
                line = line.strip().lower()  # Use lower() for case-insensitive matching
                if not line or line.startswith('#'):
                    continue

                if line.startswith("group"):
                    # 遇到新的 group 之前，保存上一个 group 的完整数据
                    if current_matrix:
                        if len(current_matrix) != num_types:
                            raise ValueError(f"Matrix for group {len(groups)+1}, shell {len(current_group_shells)+1} has incorrect row count. Expected {num_types}, got {len(current_matrix)}")
                        current_group_shells.append(np.array(current_matrix, dtype=float))

                    if current_group_shells:
                        groups.append(current_group_shells)

                    # 为新的 group 重置
                    current_group_shells = []
                    current_matrix = []
                    continue

                if line.startswith("shell"):
                    # 遇到新的 shell 之前，保存上一个矩阵
                    if current_matrix:
                        if len(current_matrix) != num_types:
                            raise ValueError(f"Matrix for group {len(groups)+1}, shell {len(current_group_shells)+1} has incorrect row count. Expected {num_types}, got {len(current_matrix)}")
                        current_group_shells.append(np.array(current_matrix, dtype=float))

                    # 为新的 shell 重置
                    current_matrix = []
                    continue

                # 普通行，解析成浮点数列表
                try:
                    values = list(map(float, line.split()))
                except ValueError:
                    raise ValueError(f"Could not parse line to floats: '{line}'")

                if len(values) != num_types:
                    raise ValueError(f"Matrix for group {len(groups)+1}, shell {len(current_group_shells)+1} has incorrect column count. Expected {num_types}, got {len(values)}")
                current_matrix.append(values)

        # 文件结束后，保存最后一个正在处理的矩阵和 group
        if current_matrix:
            if len(current_matrix) != num_types:
                raise ValueError(f"Final matrix in file has incorrect row count. Expected {num_types}, got {len(current_matrix)}")
            current_group_shells.append(np.array(current_matrix, dtype=float))

        if current_group_shells:
            groups.append(current_group_shells)

        if not groups:
            return np.empty((0, 0, num_types * num_types))

        # 验证所有 group 是否有相同数量的 shell
        num_shells_per_group = [len(g) for g in groups]
        if len(set(num_shells_per_group)) > 1:
            raise ValueError(f"All groups in the target_sro file must have the same number of shells. Found counts: {num_shells_per_group}")

        # 将嵌套列表转换为 4D numpy 数组: (num_groups, num_shells, num_types, num_types)
        arr4d = np.array(groups)

        # 获取维度信息
        num_groups, num_shells, _, _ = arr4d.shape

        # 重塑为最终需要的 3D 数组
        final_arr = arr4d.reshape((num_groups, num_shells, -1)) # -1 会自动计算为 num_types * num_types

        return final_arr
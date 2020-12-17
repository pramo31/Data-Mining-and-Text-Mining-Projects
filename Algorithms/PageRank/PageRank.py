import copy
import os


class PageRank:
    __initial_transition_probability_matrix = []
    __stochastic_transition_probability_matrix = []
    __augmented_transition_probability_matrix = []
    __vertices = 0
    __page_rank = []

    def __init__(self, input_data, damping_factor=1.0, num_iterations=100):
        parsed_input = [
            [int(i) for i in line.replace(")", "").replace("(", "").strip().split("\t")] for line in
            input_data.split("\n") if line.strip() != ""]
        self.__initial_transition_probability_matrix = [[float(0) if (sum(row) == 0) else i / sum(row) for i in row]
                                                        for row in parsed_input]

        self.__vertices = len(self.__initial_transition_probability_matrix)
        self.__damping_factor = damping_factor
        self.__page_rank = transpose_matrix([[1 / self.__vertices] * self.__vertices])

        self.__make_matrix_stochastic()
        self.__make_matrix_irreducible()
        self.__compute_page_rank(num_iterations)

    def __make_matrix_stochastic(self):
        for row in self.__initial_transition_probability_matrix:
            out_links = sum(row)
            if out_links == 0:
                self.__stochastic_transition_probability_matrix.append(
                    [1 / self.__vertices] * self.__vertices)
            else:
                self.__stochastic_transition_probability_matrix.append(
                    [i / out_links for i in row])

    def __make_matrix_irreducible(self):
        if not is_graph_strongly_connected(self.__stochastic_transition_probability_matrix):
            e_part = [[(1 - self.__damping_factor) / self.__vertices] * self.__vertices for i in range(self.__vertices)]
            a_part = [[i * self.__damping_factor for i in row] for row in
                      self.__stochastic_transition_probability_matrix]
            self.__augmented_transition_probability_matrix = add_matrices(e_part, a_part)
        else:
            self.__augmented_transition_probability_matrix = copy.deepcopy(
                self.__stochastic_transition_probability_matrix)

    def __compute_page_rank(self, num_iterations):
        for i in range(num_iterations):
            self.__page_rank = matrices_product(transpose_matrix(self.__augmented_transition_probability_matrix),
                                                self.__page_rank)

    def get_augmented_transition_probability_matrix(self):
        return self.__augmented_transition_probability_matrix

    def get_stochastic_transition_probability_matrix(self):
        return copy.deepcopy(self.__stochastic_transition_probability_matrix)

    def get_initial_transition_probability_matrix(self):
        return copy.deepcopy(self.__initial_transition_probability_matrix)

    def get_page_rank(self):
        print(self.__page_rank)
        return transpose_matrix(self.__page_rank)


def add_matrices(matrix_1, matrix_2):
    sum_matrix = []
    for row1, row2 in zip(matrix_1, matrix_2):
        row = []
        for col1, col2 in zip(row1, row2):
            row.append(col1 + col2)
        sum_matrix.append(row)
    return sum_matrix


def matrices_product(left_matrix, right_matrix):
    product_matrix = []
    for r in left_matrix:
        row = []
        for c in transpose_matrix(right_matrix):
            product_sum = 0
            for x, y in zip(r, c):
                product_sum += x * y
            row.append(product_sum)
        product_matrix.append(row)
    return product_matrix


def scalar_multiply(scalar, matrix):
    return [[scalar * i for i in row] for row in matrix]


def transpose_matrix(matrix):
    t_matrix = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i == 0):
                t_matrix.append([])
            t_matrix[j].append(matrix[i][j])
    return t_matrix


def is_graph_strongly_connected(adj_matrix):
    num_vertices = len(adj_matrix)

    # Run DFS on adj matrix to check if all vertices are visited
    # Transpose adj_matrix Run DFS to check if all vertices are visited
    return (num_vertices == num_traversible_nodes(adj_matrix)) and (
            num_vertices == num_traversible_nodes(adj_matrix))


def num_traversible_nodes(matrix):
    visited = set()
    queue = [0]
    while (queue):
        node = queue.pop()
        if (node not in visited):
            visited.add(node)
            queue += [i for i, link in enumerate(matrix[node]) if link != 0]

    return len(visited)


def process_list_for_output(output_list):
    separator = " "
    test_output = "("
    for i, line in enumerate(output_list):
        test_output += separator.join([str(round(j, round_val)) for j in line])
        if i != len(output_list) - 1:
            test_output += "\n"
    test_output += ")"
    return test_output


def write_file(output, file_name='output.txt', open_mode='w'):
    with open(file_name, open_mode) as file:
        file.write(output)
    file.close()


def read_file(file_name):
    with open(file_name, encoding='utf-8-sig') as file:
        read_data = file.read()
    return read_data


if __name__ == '__main__':
    data_set_num = 2
    round_val = 2
    num_power_iterations = 50
    damping_factor = 0.9
    input_file = f'./input/dataset{data_set_num}.txt'
    output_file = f'./output/result-{data_set_num}.txt'
    input_data_as_text = read_file(input_file)

    obj = PageRank(input_data_as_text, damping_factor, num_power_iterations)

    write_file(f"{process_list_for_output(obj.get_initial_transition_probability_matrix())}\n", output_file)
    write_file(f"{process_list_for_output(obj.get_stochastic_transition_probability_matrix())}\n", output_file, "a")
    write_file(f"{process_list_for_output(obj.get_augmented_transition_probability_matrix())}\n", output_file, "a")
    write_file(f"{process_list_for_output(obj.get_page_rank())}\n", output_file, "a")

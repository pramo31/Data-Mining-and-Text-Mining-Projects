import itertools


class MSApriori:
    __transactions = []  # T
    __tran_count = 0
    __mis_dict = {}  # MS
    __sorted_item_mis_list = []  # M
    __SDC = 0.0  # phi
    __support_count = {}  # S

    def __init__(self, transactions, parameters):
        self.__process_data(transactions, parameters)

    def __process_data(self, transactions, parameters):
        self.__transactions = process_transactions(transactions)
        self.__tran_count = len(transactions)
        processed_parameters = process_parameters(parameters)

        mis_values = {x[0]: float(x[1]) for x in processed_parameters}
        self.__SDC = mis_values["SDC"] if "SDC" in mis_values.keys() else 1.0

        item_set = set()
        for tran in self.__transactions:
            item_set.update(tran)
        item_set = sorted(list(item_set))

        for item in item_set:
            self.__mis_dict[item] = mis_values["MIS(" + str(item) + ")"] if "MIS(" + str(
                item) + ")" in mis_values.keys() else mis_values["MIS(rest)"]

        self.__sorted_item_mis_list = sorted(self.__mis_dict.items(), key=lambda x: (x[1]))

    def __init_pass(self, sorted_item_set, transactions):
        # Step 1 Generate Support count
        self.__support_count = {x[0]: 0 for x in sorted_item_set}
        for tran in transactions:
            for item in tran:
                self.__support_count[item] += 1

        # Step 2
        # First item meeting min support criteria in sorted M
        # Generate L used in generating 2-itemset candidate generation
        L = []
        found_lowest_mis_item = False
        for (item, mis_val) in sorted_item_set:
            if not found_lowest_mis_item and (self.__support_count[item] / self.__tran_count >= mis_val):
                L.append(item)
                mis = mis_val
                found_lowest_mis_item = True
            elif found_lowest_mis_item:
                if self.__support_count[item] / self.__tran_count >= mis:
                    L.append(item)
        return L

    def __level2_candidate_gen(self, L):
        c_2 = []
        for i in range(len(L)):
            item = L[i]
            if self.__support_count[item] / self.__tran_count >= self.__mis_dict[item]:
                for second_item in L[i + 1:]:
                    if self.__support_count[second_item] / self.__tran_count >= self.__mis_dict[item] and abs(
                            self.__support_count[second_item] / self.__tran_count - self.__support_count[
                                item] / self.__tran_count) <= self.__SDC:
                        c_2.append((item, second_item))
        return c_2

    def __MSCandidate_Gen(self, F, k):
        c_k = []
        for i, f1 in enumerate(F):
            for f2 in F[i + 1:]:
                if f1[:-1] == f2[:-1] and abs(self.__support_count[f1[-1]] / self.__tran_count - self.__support_count[
                    f2[-1]] / self.__tran_count) <= self.__SDC:
                    c = f1 + f2[-1:]
                    c_k.append(c)
                    for subset in list(itertools.combinations(c, k - 1)):
                        if c[0] in subset or self.__mis_dict[c[1]] == self.__mis_dict[c[2]]:
                            if subset not in F:
                                c_k.remove(c)
        return c_k

    def get_frequent_itemsets(self):
        L = self.__init_pass(self.__sorted_item_mis_list, self.__transactions)
        frequent_itemset = [[],
                            [l for l in L if self.__support_count[l] / self.__tran_count >= self.__mis_dict[l]]]
        candidates = [[], []]
        k = 2
        while frequent_itemset[k - 1]:
            if k == 2:
                candidates.append(self.__level2_candidate_gen(L))
            else:
                candidates.append(self.__MSCandidate_Gen(frequent_itemset[k - 1], k))
            for t in self.__transactions:
                for c in candidates[k]:
                    if c not in self.__support_count:
                        self.__support_count[c] = 0
                    if set(c).issubset(t):
                        self.__support_count[c] += 1
            frequent_itemset.append(
                [c for c in candidates[k] if self.__support_count[c] / self.__tran_count >= self.__mis_dict[c[0]]])
            k += 1
        return frequent_itemset


def process_transactions(transactions):
    transactions = ([x.strip('\n,},{,(,),[,]') for x in transactions])
    transactions = ([x.split(',') for x in transactions])
    transactions = ([[int(x.strip()) for x in tran_line] for tran_line in transactions])
    return transactions


def process_parameters(parameters):
    parameters = ([x.strip('\n') for x in parameters])
    parameters = ([x.split('=') for x in parameters])
    parameters = [list(map(lambda x: x.strip(), x)) for x in parameters]
    return parameters


def write_file(output, file_name='output.txt', open_mode='w'):
    with open(file_name, open_mode) as file:
        for i in range(len(output)):
            if output[i]:
                file.write("(Length-" + str(i) + " " + str(len(output[i])) + "\n")
                for j in output[i]:
                    file.write("\t(" + str(j).strip('()').replace(',', '') + ")" + "\n")
                file.write(")" + "\n")

def read_file(file_name):
    with open(file_name, encoding='utf-8-sig') as file:
        read_data = file.readlines()
    return read_data


if __name__ == '__main__':
    transaction_file = "./input/data-1/data-1.txt"
    parameters_file = "./input/data-1/para-1.txt"
    output_file = './output/1_1_result.txt'
    '''
    transaction_file = "./input/data-2/data-2.txt"
    parameters_file = "./input/data-2/para-2.txt"
    output_file = './output/1_2_result.txt'
    '''
    obj = MSApriori(read_file(transaction_file), read_file(parameters_file))
    result = obj.get_frequent_itemsets()
    print("Frequent Item Sets : " + str(result))
    write_file(result, output_file)

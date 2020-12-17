import math


class InformationGain:
    __attribute_list_set = []
    __class_set = set()
    __attribute_list = []
    __classified_list = []
    __data_count = 0
    __attribute_count = 0

    def __init__(self, input_data):
        self.__process_input(input_data)

    def __process_input(self, input_data):
        lines = [x for x in input_data.split("\n")]
        self.__attribute_count = len(lines[0].split(",")) - 1
        for i in range(self.__attribute_count):
            self.__attribute_list.append([])
            self.__attribute_list_set.append(set())
        for line in lines:
            if line.strip() == "":
                continue
            counter = 0
            self.__data_count += 1
            for each in line.split(","):
                counter += 1
                if counter > self.__attribute_count:
                    self.__class_set.add(each.strip())
                    self.__classified_list.append(each.strip())
                else:
                    self.__attribute_list[counter - 1].append(each.strip())
                    self.__attribute_list_set[counter - 1].add(each.strip())

    def get_info_gain(self, attribute_column):
        root_entropy = self.__get_entropy(self.__classified_list)
        att_entropy = self.__get_attribute_entropy(self.__attribute_list[attribute_column - 1])
        return round(root_entropy - att_entropy, round_val)

    def __get_attribute_entropy(self, attribute_list):
        count_map = {}
        att_class_map = {}
        data_count = len(attribute_list)
        for i in range(len(attribute_list)):
            att = attribute_list[i]
            if att in count_map:
                count_map[att] += 1
                att_class_map[att].append(self.__classified_list[i])
            else:
                count_map[att] = 1
                att_class_map[att] = [self.__classified_list[i]]

        entropy_att = 0
        for key, value in count_map.items():
            entropy_att += (value / data_count) * self.__get_entropy(att_class_map[key])
        return entropy_att

    def __get_entropy(self, class_list):
        count_map = {}
        data_count = len(class_list)
        for c in class_list:
            if c in count_map:
                count_map[c] += 1
            else:
                count_map[c] = 1
        entropy = 0
        for key, value in count_map.items():
            pr_c = value / data_count
            entropy += -1 * (pr_c * math.log(pr_c, 2))
        return entropy


def write_file(output, file_name='output.txt', open_mode='w'):
    with open(file_name, open_mode) as file:
        file.writelines(output)
    file.close()


def read_file(file_name):
    with open(file_name, encoding='utf-8-sig') as file:
        read_data = file.read()
    return read_data


if __name__ == '__main__':
    round_val = 5
    input_file = './input/2-data-1.txt'
    output_file = f'./output/2.txt'
    input_data_as_text = read_file(input_file)
    obj = InformationGain(input_data_as_text)
    attribute_number = int(input("Which attribute column do you want to calculate IG for : "))
    info_gain = str(obj.get_info_gain(attribute_number))
    print(f"Information Gain for Attribute in Column {attribute_number} is {info_gain}")
    output = f"(IG {info_gain})"

    write_file(output, output_file)

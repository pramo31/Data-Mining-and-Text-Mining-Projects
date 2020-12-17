class ClassifierEvaluator:
    __test_results = []
    data_count = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    def __init__(self, input_data):
        self.__process_input(input_data)
        self.__process_confusion_matrix(self.__test_results)

    def __process_input(self, input_data):
        lines = [x for x in input_data.split("\n")]
        for line in lines:
            if line.strip() == "":
                continue
            self.data_count += 1
            result = [x.strip() for x in line.split(",")]
            self.__test_results.append((int(result[0]), float(result[1]), self.classify(float(result[1])),
                                        int(result[2])))

    def __process_confusion_matrix(self, test_results):
        for result in test_results:
            if result[2] == 1:
                if result[3] == 1:
                    self.true_positive += 1
                else:
                    self.false_positive += 1
            else:
                if result[3] == 1:
                    self.false_negative += 1
                else:
                    self.true_negative += 1

    def classify(self, score, threshold=0.5):
        if score >= threshold:
            return 1
        else:
            return 0

    def get_accuracy(self, true_positive, true_negative, data_count):
        return round((true_positive + true_negative) / data_count, round_val)

    def get_precision(self, true_positive, false_positive):
        return round(true_positive / (true_positive + false_positive), round_val)

    def get_recall(self, true_positive, false_negative):
        return round(true_positive / (true_positive + false_negative), round_val)

    def get_f1_score(self, true_positive, false_positive, false_negative):
        precision = self.get_precision(true_positive, false_positive)
        recall = self.get_recall(true_positive, false_negative)
        return round((2 * precision * recall) / (precision + recall), round_val)

    def get_true_positive_rate(self, true_positive, false_negative):
        return self.get_recall(true_positive, false_negative)

    def get_false_positive_rate(self, false_positive, true_negative):
        return round(false_positive / (false_positive + true_negative), round_val)

    def get_sensitivity(self, true_positive, false_negative):
        return round(self.get_recall(true_positive, false_negative), round_val)

    def get_specificity(self, false_positive, true_negative):
        return round(1 - self.get_false_positive_rate(false_positive, true_negative), round_val)

    # AUC using the single calculated TPR and FPR for the default threshold of 0.5
    def get_auc(self, true_positive, false_positive, true_negative, false_negative):
        tpr = self.get_true_positive_rate(true_positive, false_negative)
        fpr = self.get_false_positive_rate(false_positive, true_negative)
        return round(0.5 - (fpr / 2) + (tpr / 2), round_val)

    def get_auc_multiple_threshold(self):
        points = [(0, 0)]
        scores = [x[1] for x in self.__test_results]
        for threshold in sorted(scores, reverse=True):
            tp, fp, tn, fn = self.get_fpr_tpr_for_threshold(self.__test_results, threshold)
            points.append((self.get_false_positive_rate(fp, tn), self.get_true_positive_rate(tp, fn)))
        return round(self.calculate_auc(points), round_val)

    def get_fpr_tpr_for_threshold(self, test_results, threshold):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for result in test_results:
            model_class = self.classify(result[1], threshold)
            if model_class == 1:
                if result[3] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if result[3] == 1:
                    fn += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def calculate_auc(self, points):
        sorted_points = sorted(points, key=lambda x: x[0])
        auc = 0
        for i in range(1, len(sorted_points)):
            x1, y1 = sorted_points[i - 1]
            x2, y2 = sorted_points[i]
            auc += (x2 - x1) * y1 + 0.5 * (x2 - x1) * (y2 - y1)
        return auc


def write_file(output, file_name='output.txt', open_mode='w'):
    with open(file_name, open_mode) as file:
        file.write(output)
    file.close()


def read_file(file_name):
    with open(file_name, encoding='utf-8-sig') as file:
        read_data = file.read()
    return read_data


if __name__ == '__main__':
    round_val = 5
    data_set_num = 5
    input_file = f'./input/3-data-{data_set_num}.txt'
    output_file = f'./output/{data_set_num}_result.txt'
    input_data_as_text = read_file(input_file)
    obj = ClassifierEvaluator(input_data_as_text)
    output = "(\n"
    output += f"(Accuracy {obj.get_accuracy(obj.true_positive, obj.true_negative, obj.data_count)})\n"
    output += f"(Precision {obj.get_precision(obj.true_positive, obj.false_positive)})\n"
    output += f"(Recall {obj.get_recall(obj.true_positive, obj.false_negative)})\n"
    output += f"(F1 {obj.get_f1_score(obj.true_positive, obj.false_positive, obj.false_negative)})\n"
    output += f"(TPR {obj.get_true_positive_rate(obj.true_positive, obj.false_negative)})\n"
    output += f"(FPR {obj.get_false_positive_rate(obj.false_positive, obj.true_negative)})\n"
    output += f"(Specificity {obj.get_specificity(obj.false_positive, obj.true_negative)})\n"
    output += f"(Sensitivity {obj.get_sensitivity(obj.true_positive, obj.false_negative)})\n"
    # output += f"(AUC {obj.get_auc(obj.true_positive, obj.false_positive, obj.true_negative, obj.false_negative)})\n"
    output += f"(AUC {obj.get_auc_multiple_threshold()})\n"
    output += ")"
    print("Classifier Characteristics :\n" + output)
    write_file(output, output_file)

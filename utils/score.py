#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def cm_submission(gold_labels, test_labels):
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return cm


def print_confusion_matrix(cm,type=None):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))

    filename = "./results/confusion_matrix.txt" if type == None else "./results/confusion_matrix_{}.txt".format(type)
    with open(filename, "w+") as f:
        f.write('\n'.join(lines))
    f.close()

def report_score(actual,predicted, is_full_set, type=None):
    score = get_f1_score(actual,predicted,is_full_set)
    best_score = get_f1_score(actual,actual,is_full_set)
    cm = cm_submission(actual,predicted)

    print_confusion_matrix(cm,type)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

def detailed_score(actual, predicted, is_full_set):

    assert len(actual) == len(predicted)

    def output_score(label):
        actual_out = [1 if x != "unrelated" else 0 for x in actual] if label == 'related' else [1 if x == label else 0 for x in actual]
        predicted_out = [1 if x != "unrelated" else 0 for x in predicted] if label == 'related' else [1 if x == label else 0 for x in predicted]

        actual_count = actual_out.count(1)
        predict_count = predicted_out.count(1)

        precision = precision_score(actual_out, predicted_out)
        recall = recall_score(actual_out, predicted_out)
        output = []
        output.append("--------------------")
        output.append("actual {} count {}".format(label, actual_count))
        output.append("predicted {} count {}".format(label, predict_count))
        output.append("precision: {}".format(precision))
        output.append("recall: {}".format(recall))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        output.append("F1 score: " + str(f1))
        print('\n'.join(output))
        with open("./results/{}_evaluation.txt".format(label), "w+") as f:
            f.write('\n'.join(output))
        f.close()
        return f1

    the_f1 = []
    the_f1.append(output_score('agree'))
    the_f1.append(output_score('disagree'))
    the_f1.append(output_score('discuss'))
    if is_full_set:
        the_f1.append(output_score('unrelated'))
    output_score('related')
    print('Average F1 score is ' + str(sum(the_f1) / len(the_f1)))

def get_f1_score(actual, predicted, is_full_set):

    assert len(actual) == len(predicted)

    def output_score(label):
        actual_out = [1 if x != "unrelated" else 0 for x in actual] if label == 'related' else [1 if x == label else 0 for x in actual]
        predicted_out = [1 if x != "unrelated" else 0 for x in predicted] if label == 'related' else [1 if x == label else 0 for x in predicted]
        precision = precision_score(actual_out, predicted_out)
        recall = recall_score(actual_out, predicted_out)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return f1

    the_f1 = []
    the_f1.append(output_score('agree'))
    the_f1.append(output_score('disagree'))
    the_f1.append(output_score('discuss'))
    if is_full_set:
        the_f1.append(output_score('unrelated'))
    return sum(the_f1) / len(the_f1)
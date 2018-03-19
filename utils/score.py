#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

import pdb

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


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

    filename = "confusion_matrix.txt" if type == None else "confusion_matrix_{}.txt".format(type)
    with open(filename, "w") as f:
        f.write('\n'.join(lines))
    f.close()

def report_score(actual,predicted,type=None):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm,type)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

def detailed_score(actual, predicted):

    def output_score(label):
        actual_count = actual.count('agree') + actual.count('disagree') + actual.count('discuss') if label == 'related' else actual.count(label)
        predict_count = predicted.count('agree') + predicted.count('disagree') + predicted.count('discuss') if label == 'related' else predicted.count(label)
        
        precision = predict_count / len(predicted)
        recall = predict_count / actual_count
        output = []
        output.append("--------------------")
        output.append("actual {} count {}".format(label, actual_count))
        output.append("predicted {} count {}".format(label, predict_count))
        output.append("precision: {}".format(precision))
        output.append("recall: {}".format(recall))
        output.append("F1: {}".format(2 * precision * recall / (precision + recall)))
        print('\n'.join(output))
        with open("{}_evaluation.txt".format(label), "w") as f:
            f.write('\n'.join(output))
        f.close()

    output_score('agree')
    output_score('disagree')
    output_score('discuss')
    output_score('unrelated')
    output_score('related')

if __name__ == "__main__":
    actual = [0,0,0,0,1,1,0,3,3]
    predicted = [0,0,0,0,1,1,2,3,3]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])

    detailed_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])

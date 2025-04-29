import json
import argparse
from tabulate import tabulate


tasks = {
    "count": ("4_count.json", "4_count", "video"),
    "ego": ("3_ego.json", "3_ego", "video"),
    "needle": ("2_needle.json", "2_needle", "video"),
    "order": ("5_order.json", "5_order", "video"),
    "plotQA": ("1_plotQA.json", "1_plotQA", "video"),
    "anomaly_reco": ("6_anomaly_reco.json", "6_anomaly_reco", "video"),
    "topic_reasoning": ("7_topic_reasoning.json", "7_topic_reasoning", "video")
}


def main():
    args = parse_args()
    res = [eval(x.strip()) for x in open(args.pred_path, 'r').readlines()]
    task_types = tasks.keys()
    task_acc = {x: [] for x in task_types}
    acc = []
    for i, x in enumerate(res):
        value = 1
        if x['pred'] != x['gt']:
            value = 0
        acc.append(value)
        task_acc[x['task_type']].append(value)
    acc = sum(acc) * 100 / len(acc)
    task_acc = {x: sum(task_acc[x]) * 100 / len(task_acc[x]) for x in task_acc}
    print(f"{args.pred_path}:", acc)
    task_names = list(tasks.keys())
    
    table_data = []
    for i in range(len(task_names) // 4):
        row_task_names = task_names[i * 4: (i + 1) * 4]
        row_task_acc = [task_acc[x] for x in row_task_names]
        table_data.append(row_task_names)
        table_data.append(row_task_acc)
    print(tabulate(table_data, floatfmt=".2f"), '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video captioning.")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

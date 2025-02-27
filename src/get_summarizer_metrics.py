import os
import json

import pandas as pd

folder = "/Users/odunayoogundepo/Desktop/jubilant-sniffle/runs/xl_sum/google_gemma-2_9b_it"
all_results = []


for file in os.listdir(folder):
    language = file.split('_')[2]
    file_summary = {"bleu": [], "rouge1": [], "rouge2": [], "rougeL": [], "bertscore_precision": [], "bertscore_recall": [], "bertscore_f1": []}
    if file.endswith(".json"):
        with open(os.path.join(folder, file)) as f:
            data = json.load(f)
        
        for k, v in data.items():
            for sample in v:
                file_summary["bleu"].append(sample['metrics']["bleu"]['bleu'])
                file_summary["rouge1"].append(sample['metrics']["rouge"]['rouge1'])
                file_summary["rouge2"].append(sample['metrics']["rouge"]['rouge2'])
                file_summary["rougeL"].append(sample['metrics']["rouge"]['rougeL'])
                file_summary["bertscore_precision"].append(sample['metrics']["bertscore"]['precision'][0])
                file_summary["bertscore_recall"].append(sample['metrics']["bertscore"]['recall'][0])
                file_summary["bertscore_f1"].append(sample['metrics']["bertscore"]['f1'][0])
            
            for metric, list in file_summary.items():
                file_summary[metric] = sum(list) / len(list)
            
            file_summary['prompt_type'] = k.split('.txt')[0]
            file_summary['language'] = language
            all_results.append(file_summary)


file_name = "google_gemma-2_9b_it_xl_sum.csv"
pd.DataFrame(all_results).to_csv(file_name, index=False)


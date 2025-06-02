import argparse
import json
import os

def fetch_classes_from_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('_oracle.txt'):
            # Remove the "_oracle.txt" part
            base = filename[:-11]
            # Split into class1 and class2
            if '_' in base:
                parts = base.split('_', 1)  # Only split at the first underscore
                class1, class2 = parts[0], parts[1]
                results.append((class1, class2, filename))
        else:
            print(filename)
    return results

def run(args):
    work_dir = args.work_dir
    orcale_summaries_dir = args.orcale_summaries_dir

    key_count = 1
    quries_dict = {}
    for i in range(1,5):
        gt_dir = f'{orcale_summaries_dir}/P0{i}'
        class_pair = fetch_classes_from_files(gt_dir)
        for (class1, class2, gt_file_name) in class_pair:
            VidQry = f'VidQry_{key_count}'
            video_id = f'P0{i}'
            query = f'Focus on scenes containing {class1} and {class2}'
            quries_dict[VidQry] = {
                'video_id' : video_id,
                'query' : query,
                'gt_file' : gt_file_name
            }
            key_count += 1
    
    results_file = f'{work_dir}/QFVS_mapping_new.json'
    with open(results_file, 'w') as json_file :
        json.dump(quries_dict, json_file, indent=4)
    return

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QFVS Mapping')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--orcale_summaries_dir", type=str, help='Ground truth file path')
    args = parser.parse_args()
    run(args)

"""
python data/QFVS/QFVS_mapping.py 
--work_dir /root/data/QFVS 
--orcale_summaries_dir /root/Datasets/QFVS/data/origin_data/Query-Focused_Summaries/Oracle_Summaries
"""
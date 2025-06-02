import json
import argparse

def run(args):
    mapping_file = args.mapping_file
    output_file = args.output_file

    mapping = None
    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)
    
    splits = []
    for i in range(1,5,1):
        test_keys = []
        train_keys = []
        for vidQry in mapping.keys():
            if mapping[vidQry]['video_id'] == f'P0{i}':
                test_keys.append(vidQry)
            else:
                train_keys.append(vidQry)
        split = {'train_kets' : train_keys, 'test_keys' : test_keys, 'test_video_id' : i}
        splits.append(split)

    with open(output_file, 'w') as json_file:
        json.dump(splits, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QFVS Construct splits (split containes 1 video)')
    parser.add_argument("--mapping_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    run(args)

"""
python /root/data/QFVS/GFVS_splits.py \
--mapping_file /root/data/QFVS/QFVS_mapping_new.json \
--output_file /root/data/QFVS/QFVS_splits.json 
"""
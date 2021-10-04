import argparse
import json
import os

'''
    function:
        check label including none class or not
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_json',type=str, help='path/to/label_need_check')
    args = parser.parse_args()
    return args


def main():
    args=get_args()
    for video in sorted(os.listdir(args.label_json)):
        for file in sorted(os.listdir(os.path.join(args.label_json,video))):
            path = os.path.join(args.label_json, video, file)
            with open(path, 'r') as _:
                f = json.load(_)
            for label in f['labels']:
                if label['label_class'] == None:
                    print('video_{}: file:{} is wrong'.format(video,file))


if __name__ == '__main__':
    main()
''' 
    function:
        To demmostraing scrips function
    
    scripts:
        train.sh
            arg1: path/to/config-file
            arg2: num_gpus
            arg3: path/to/output_dir
        png2mrcnn_label.sh
            arg1: path/to/mask_folder
            arg2: path/to/train.json
        json2mask.sh
            arg1: path/to/mask_folder
            arg2: path/to/label_json_folder



'''

#TODO: shell cript changing content of file
# https://stackoverflow.com/questions/14643531/changing-contents-of-a-file-through-shell-script/28559815



'''
    dataset information:
        TSL:
            train:
                frames: /home/Datasets/TSL/train/frames/
                json:   /home/Datasets/TSL/train/train.json
            val:
                frames: /home/Datasets/TSL/val/frames/
                json:   /home/Datasets/TSL/val/val.json

        ASL:
            train:
                men:
                    student_label_well:
                        frames: /home/Datasets/ASL/train/0915ASL_men/frame/ 
                        json:   /home/Datasets/ASL/train/0915ASL_men/train_0915ASL_men.json
                    all
                        frames:
                        json:

                women:
                    long_sleeve:
                        frams:
                        json:
                    short_sleeve:
                        frames:
                        json:
            val:
                men:
                    frames: /home/Datasets/ASL/train/0901spreadthesign_men/val/
                    json:   /home/Datasets/ASL/train/0901spreadthesign_men/val_0901spreadthesign_men.json

''' 

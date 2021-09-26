PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

'''
function:
    convert modified json file to png
args:
    $1: path/to/mask_folder
    $2: path/to/label_json_folder
'''

echo "====> start convert mask to json"
cd /home/detectron2/detectron2/tool
python png2json.py $1 $2
echo "====> create label json done."

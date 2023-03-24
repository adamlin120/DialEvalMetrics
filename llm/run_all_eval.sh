# Run all eval for each json data file under "./data/" excluding "dstc6.json"
# e.g. python eval.py data/topicalchat_usr.json cache/0-100/topicalchat_usr.json 0-100
# e.g. python eval.py data/topicalchat_usr.json cache/0-5/topicalchat_usr.json 0-5

data_dir=./data
cache_dir=./cache

# for loop
for file in $data_dir/*.json; do
    filename=$(basename $file)
    filename="${filename%.*}"
    if [ $filename != "dstc6" ]; then
        for dir in $cache_dir/*; do
            dirname=$(basename $dir)
            echo $filename
            echo $dirname
            python eval.py $data_dir/$filename.json $cache_dir/$dirname/$filename.json $dirname
        done
    fi
done

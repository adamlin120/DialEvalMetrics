# Run all eval for each json data file under "./data/" excluding "dstc6.json"
# e.g. python eval.py data/topicalchat_usr.json cache/0-100/topicalchat_usr.json 0-100
# e.g. python eval.py data/topicalchat_usr.json cache/0-5/topicalchat_usr.json 0-5
# Init prompt styles: ["0-100", "0-100_ref", "0-5", "0-5_ref"]
prompt_styles=("0-100" "0-100_ref" "0-5" "0-5_ref")


data_dir=./data
cache_dir=./cache

# for loop
for file in $data_dir/*.json; do
    filename=$(basename $file)
    filename="${filename%.*}"
    if [ $filename != "dstc6" ]; then
        for dir in $cache_dir/*; do
          for prompt_style in ${prompt_styles[@]}; do
            echo $filename
            echo $prompt_style
            python eval.py $data_dir/$filename.json $cache_dir/$prompt_style/$filename.json $prompt_style
          done
        done
    fi
done

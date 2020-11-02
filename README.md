#  A Clarifying Question Selection System from NTES\_ALONG in Convai3 Challenge


The system solution from team NTES\_ALONG in EMNLP 2020 workshop Convai3 Challenge.

## How to run
First download the model file from <https://drive.google.com/file/d/1_XO94D6_4Psf5214pVxqenJVgHb43FSA/view?usp=sharing> to models/*
then decompress the models
```shell script
mkdir models

cd models
# download `convai3_system_models.zip` from <https://drive.google.com/file/d/1_XO94D6_4Psf5214pVxqenJVgHb43FSA/view?usp=sharing> here

unzip convai3_system_models.zip
```

Take "dev\_tiny\_synthetic.pkl" as an input example
Remember to replace the ${work\_path} to your abs path

```shell script
# first make dir "data_dir"
mkdir data_dir

# then put the input pkl file in "data_dir"
cp dev_tiny_synthetic.pkl ./data_dir/.

# third build docker image
docker build -t convai3_ntes_along:v20201028 ./

# run docker container
docker run --runtime=nvidia -d -v ${work_path}/data_dir:/workspace/data_dir convai3_ntes_along:v20201028 python -u Interface.py --input_file_path /workspace/data_dir/dev_tiny_synthetic.pkl --output_file_path /workspace/data_dir/dev_tiny.result
# here will output a container id
>> ${container_id}

# watch the running progress
docker logs -f ${container_id}

# finally the result is in "data_dir"
head -n 5 data_dir/dev_tiny.result
```


## Citing
```
@misc{ou2020clarifying,
      title={A Clarifying Question Selection System from NTES_ALONG in Convai3 Challenge}, 
      author={Wenjie Ou and Yue Lin},
      year={2020},
      eprint={2010.14202},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

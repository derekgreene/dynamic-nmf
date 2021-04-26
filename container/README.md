## Building a Docker image to run the preprocessing

-----

#### HISTORY

* 4/25/21 mbod - initial setup and testing

----

### Notes

The `Dockerfile` creates an image based on the Docker `python:3.7-slim-buster` container and then installs the required module for the
preprocessing script `congress_pre_process.py`.

* `congress_pre_process.py` is in the `script` folder
* There is a local module `text` that gets copied to the container in this same folder
* The `data` folder has `input` and `output`
	* In `input` place a pair of congress files, e.g. `speeches_110.txt` and `descr_110.txt`
	* Also script looks for a JSON file passed in the `omit_tokens` arg, a dummy empty file is also in `input`

### Building and running

* Assuming Docker is installed locally, to build:
```
# from the container directory
docker build -t preproc .
```

* To run:
```
docker run -v '{ABSPATH to container}/data:/data' \
           -v "{ABSPATH to container}/working" \
           preproc /working/congress_pre_process.py /data/input 110 /data/input/omit_tokens.json /data/output
```

* Replace `{ABSPATH to container}` to the absolute path to the `container` folder on local system.
* This maps the local `data` folder to `/data` in the container and `script` to `/working`
* Two output files should be written to `data/output`


### Next steps

* Create a build and deploy script to upload the docker image to EC2
* Test on SageMaker

### Refs

* https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html
	* https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own/container

* https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html

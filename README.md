Paper Accepted to WSDM'22
=============

* ### Title: Linear, or Non-Linear, That is the Question!


How to Use
=============

* ### Docker environments

```
docker pull pytorch/pytorch
```

* ### Docker run

```
docker run --gpus all -it --rm --privileged -v {local_path}:/HMLET pytorch/pytorch bash -c "pip install pandas && pip install scipy && pip install sklearn && pip install tensorboardX && pip install openpyxl && cd /HMLET && {train_model_command}"
```

* ### Train model

```
python train.py --dataset {dataset_name} --model {model_variants}
```

Methods Proposal Background and Purpose
=============

* ### Which embedding propagation (linear & non-linear) is more appropriate to recommender systems?

Methods
=============

* ### HMLET (Hybrid-Method-of-Linear-and-non-linEar-collaborative-filTering-method)
  * ##### Dynamically selecting the best propagation method for each node in a layer using gating networks.


* ### Four variants of HMLET: HMLET (End), HMLET (Middle), HMLET (Front), HMLET (All)
  * ##### Four variants of HMLET in terms of the location of the non-linear propagation.
  <p align="center">
    <img src="https://user-images.githubusercontent.com/52263269/141878827-d40a2844-8fad-4d75-aae3-0f693bb1034c.png" width="550px" height="350px"></img>
  </p>  


* ### HMLET (End)
  * ##### HMLET (End) shows best performance among these variants
  * ##### Focusing on gating in the third and fourth layers
  * ##### The detailed workflow of HMLET (End)
  <p align="center">
    <img src="https://user-images.githubusercontent.com/52263269/141666368-71bff1c9-f4a4-4ffd-b6ca-f0ecbdf5f845.png" width="1100px" height="350px"></img>
  </p>
  

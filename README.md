#### This repository is extension of official repository `https://github.com/facebookresearch/low-shot-shrink-hallucinate`, with added functions to run the code on any generic dataset. 


## Running the Code


### Cleaning data
Remove all the corrput images. 
Use the script data_cleaning.py.

    python data_cleaning.py --dataset_folder "dataset_path"


### Split classes into base and novel
Split all classes into - 2. Threshold parameter decides how many images a class should countain to be called as base class. 
Use the script tuple_generator.py

    python tuple_generator.py --dataset_folder "dataset_path" --threshold 150


### Creating Validation dataset
Create a separate folder for validation data. Inside the folder create folder for each classes and put the validation data. 
Use the script prepare_validation_data.py

    python prepare_validation_data.py --dataset_folder "dataset_path" --test_folder "test_dataset_path" --test_images 50


### Training a ConvNet representation
To train the ConvNet, we first need to specify the training and validation sets.
The training and validation datasets, together with data-augmentation and preprocessing steps, are specified through yaml files: see `base_classes_train_template.yaml` and `base_classes_val_template.yaml`.
You will need to specify the path to the directory containing dataset in each file.

The main entry point for training a ConvNet representation is `main.py`. For example, to train a ResNet10 representation with the sgm loss, run:

    mkdir -p checkpoints/ResNet10_sgm
    python ./main.py --model ResNet10 \
      --traincfg base_classes_train_template.yaml \
      --valcfg base_classes_val_template.yaml \
      --print_freq 10 --save_freq 10 \
      --aux_loss_wt 0.02 --aux_loss_type sgm \
      --checkpoint_dir checkpoints/ResNet10_sgm
      
Here, `aux_loss_type` is the kind of auxilliary loss to use (`sgm` or `l2` or `batchsgm`), `aux_loss_wt` is the weight attached to this auxilliary loss, and `checkpoint_dir` is a cache directory to save the checkpoints. 

The model checkpoints will be saved as epoch-number.tar. Training by default runs for 90 epochs, so the final model saved will be `89.tar`.


### Saving features from the ConvNet
The next step is to save features from the trained ConvNet. This is fairly straightforward: first, create a directory to save the features in , and then save the features for the train set and the validation set. Specify the root folders for train and valid set in `train_save_data.yaml` and `val_save_data.yaml` files. Thus, for the ResNet10 model trained above:
    
    mkdir -p features/ResNet10_sgm
    python ./save_features.py \
      --cfg train_save_data.yaml \
      --outfile features/ResNet10_sgm/train.hdf5 \
      --modelfile checkpoints/ResNet10_sgm/89.tar \
      --model ResNet10

### Finetuning last layer for novel classes
The next step is fine tune last layer with novel classes data.

    python fine_tuning_last_layer.py \
    --trainfile features/new-folder/train.hdf5 \
    --maxiters 10000
    


### Testing
Final step is testing on single image.

    python testing_on_single_image.py \
    --model ResNet10
    --config testing.yaml \
    --modelfile checkpoints/ResNet10_sgm/89.tar \ 
    --num_classes 10378\
    --image_path image.jpg \
    

